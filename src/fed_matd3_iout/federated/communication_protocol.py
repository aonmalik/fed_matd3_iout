"""Asynchronous federated communication protocol over intermittent acoustic links.

Per client:  local learning (always) -> update ready -> feasible contact -> UPLOAD (fragmented over windows, fading losses)
-> last bit leaves the AUV (IN_FLIGHT) -> USV receives after propagation (BUFFERED, or REJECTED at receipt) ->
aggregation (AGGREGATED, or REJECTED/EXPIRED) -> control messages travel back on the downlink.  Independently the USV
pushes new global versions via download transfers when the client's link is open.

Update lifecycle (details: update_packet.py)
    created -> uploading -> in_flight -> buffered -> aggregated | rejected | expired
  * ONE outstanding update per client for its whole life: no new update is created until the client has processed the
    terminal outcome of the previous one (``FederatedClient.outstanding``).
  * Two kinds of downlink messages: 'receipt' (TRANSPORT ACK, informational) and the final 'commit' / 'reject' / 'expire'.
    Only 'commit' advances the client's reference model.  Control messages of a client are queued in the server outbox and
    sent as a list over ONE downlink transfer at a time, so they can never overwrite each other.
  * The update's window statistics are owned by the update (``pkt.stats``) and settled EXACTLY ONCE in ``_settle``:
        AGGREGATED                         -> consumed (they were used for the relevance weight)
        REJECTED / EXPIRED, same episode   -> returned to the environment window (nothing was committed, nothing is lost)
        REJECTED / EXPIRED, other episode  -> discarded (never restored into a different episode)
  * Settlement happens when the authoritative terminal state is reached (server side / client abandon), not when the client
    hears about it, so the accounting does not depend on downlink luck.
"""
from __future__ import annotations

import numpy as np

from ..communication.capacity_arbiter import arbitrate
from ..communication.contact_scheduler import ContactScheduler
from ..communication.model_transfer import ModelTransfer
from .federated_client import FederatedClient
from .update_packet import UpdateStatus
from .usv_server import USVServer

_ACC_KEYS = ("coll_mbit", "owned_loss_mbit", "V_mature", "U_mature")


class FederatedProtocol:
    def __init__(self, clients: list[FederatedClient], server: USVServer, fcfg, usv_cfg, dt: float):
        self.clients, self.server, self.f = clients, server, fcfg
        self.share = float(usv_cfg["model_share"])                 # POOL of airtime for ALL model traffic of one AUV (2.2)
        self.duplex = str(usv_cfg.get("duplex", "half"))           # 'half': one shared pool; 'full': per-direction pools
        if self.duplex not in ("half", "full"):
            raise ValueError(f"acoustic.usv_link.duplex must be 'half' or 'full', got {self.duplex!r}")
        self.last_alloc: dict[int, dict[str, float]] = {}          # per-AUV grants of the latest step (tests / diagnostics)
        self.alloc_violations = 0                                  # steps where a pool was exceeded (must stay 0)
        self.alloc_steps = 0
        self.accept_new = True                                     # False during the final drain: no NEW update is created
        self.frag = bool(fcfg["allow_fragmentation"])
        self.sched = ContactScheduler(len(clients), dt, self.frag)
        self.dt = dt
        self.inflight: list[tuple[ModelTransfer, int]] = []       # uploads whose last bit is in the air
        self.comm_log: list[dict] = []
        self.link_up_steps = 0
        self.link_total_steps = 0
        self.ack_bytes = float(fcfg["ack_bytes"])
        self.model_bytes = clients[0].model_params * float(fcfg["model_bytes_per_param"])
        # ---- exactly-once accounting
        self.n_created = 0
        self.n_transmitted = 0                                     # reached IN_FLIGHT (last bit left the AUV)
        self.n_status = {"aggregated": 0, "rejected": 0, "expired": 0}
        self.stats_disposition = {"consumed": 0, "returned": 0, "discarded_episode": 0}
        self.stats_mass = {d: {k: 0.0 for k in _ACC_KEYS} for d in self.stats_disposition}
        self.update_log: list[dict] = []                           # one row per settled update
        self.rejected_bytes = 0.0                                  # uplink bytes of updates that ended REJECTED
        self.expired_bytes = 0.0                                   # uplink bytes (possibly partial) of EXPIRED updates

    # ------------------------------------------------------------------------------------------
    def _settle(self, env, pkt, t: float) -> None:
        """Settle a terminal update exactly once: count its outcome and dispose of its window statistics."""
        if pkt.stats_disposition != "held":
            return                                                 # already settled (defence against double processing)
        if not pkt.terminal:
            raise RuntimeError(f"{pkt.uid} settled before reaching a terminal state ({pkt.status.value})")
        if pkt.status is UpdateStatus.AGGREGATED:
            disp = "consumed"
        elif env.return_window_stats(pkt.client_id, pkt.stats, pkt.episode_id):
            disp = "returned"
        else:
            disp = "discarded_episode"
        pkt.stats_disposition = disp
        self.stats_disposition[disp] += 1
        for k in _ACC_KEYS:
            self.stats_mass[disp][k] += float(pkt.stats.get(k, 0.0))
        self.n_status[pkt.status.value] += 1
        if pkt.status is UpdateStatus.REJECTED:
            self.rejected_bytes += pkt.bytes_sent
        elif pkt.status is UpdateStatus.EXPIRED:
            self.expired_bytes += pkt.bytes_sent
        self.update_log.append({"uid": pkt.uid, "client": pkt.client_id, "status": pkt.status.value, "reason": pkt.reason,
                                "disposition": disp, "episode": pkt.episode_id, "t_created": pkt.t_created, "t_end": t,
                                "base_version": pkt.base_version})

    # ------------------------------------------------------------------------------------------
    def _demand(self, tr: ModelTransfer | None, link) -> float:
        """Airtime (fraction of one step at the full link rate) the transfer still needs; 0 if it cannot send now."""
        if tr is None or tr.done_tx or not link.up or not link.success:
            return 0.0
        rate = link.rate_mbps * 1e6 / 8.0
        return float("inf") if rate <= 0 else (tr.total_bytes - tr.progress_bytes) / (rate * self.dt)

    def _grants(self, c: FederatedClient, link, env=None) -> dict[str, float]:
        """Per-transfer airtime grants for this step (sum <= pool; half duplex: one pool, full duplex: per direction)."""
        d = {"upload": self._demand(c.upload, link), "ctrl": self._demand(c.ctrl, link),
             "download": self._demand(c.download, link)}
        # Half duplex: airtime already used by the sensor-data uplink in THIS step's time slot (the environment steps first) is
        # not available to model traffic, so sensor data + model traffic never exceed one step of airtime on the AUV's USV-link
        # modem.  In steady state the sensor uplink is held to 1 - pool by the reservation, so the full pool remains.
        arr = getattr(env, "sensor_uplink_airtime_s", None)
        used = float(arr[c.id]) / self.dt if arr is not None else 0.0
        free = max(0.0, 1.0 - used)
        if self.duplex == "half":
            pool = min(self.share, free)
            g = arbitrate({k: min(v, 1.0) for k, v in d.items()}, pool)
            pools = [(g, pool)]
        else:                                                      # full duplex: only the uplink direction competes with sensor data
            up_pool, dn_pool = min(self.share, free), self.share
            g = arbitrate({"upload": min(d["upload"], 1.0)}, up_pool)
            g.update(arbitrate({k: min(d[k], 1.0) for k in ("ctrl", "download")}, dn_pool))
            pools = [({"upload": g["upload"]}, up_pool), ({k: g[k] for k in ("ctrl", "download")}, dn_pool)]
        self.alloc_steps += 1
        if any(sum(x.values()) > pool + 1e-9 for x, pool in pools):
            self.alloc_violations += 1
        return g

    def step(self, env, t: float) -> list[dict]:
        """Advance all transfers by one step of global time ``t`` (seconds). Returns aggregation records."""
        links = env.links
        self.sched.observe(links)
        self.link_up_steps += sum(1 for l in links if l.up)
        self.link_total_steps += len(links)
        recs = []
        sync = self.f["mode"] == "sync"
        for c in self.clients:
            i = c.id
            l = links[i]
            tx_s = rx_s = 0.0
            # ---- start a new upload: only when NO update of this client is unresolved
            ready = self.accept_new and c.outstanding is None and c.upload is None and c.update_ready() and not (sync and c.awaiting_global)
            if ready and not c.retry_allows(t):                    # bounded back-off after failed updates: defer, never drop
                c.stats["retry_deferred_steps"] += 1
                if not c._defer_open:
                    c._defer_open = True
                    c.stats["retry_deferred"] += 1                 # one deferral episode
            elif ready:
                if self.sched.may_start(i, self.model_bytes, l, self.share):
                    if c.consecutive_failures > 0:
                        c.stats["retries"] += 1                    # an upload attempted after >= 1 consecutive failure
                    c._defer_open = False
                    pkt = c.make_update(t, env.pop_window_stats(i), env.episode_id)
                    pkt.transition(UpdateStatus.UPLOADING, t)
                    pkt.t_upload_start = t
                    c.upload = ModelTransfer("upload", pkt.size_bytes, pkt, t_start=t)
                    c.stats["attempted"] += 1
                    self.n_created += 1
                    c.awaiting_global = True
                elif l.up:
                    c.stats["deferred"] += 1
            # ---- open the downlink transfers that will compete for airtime this step
            if c.ctrl is None:
                msgs = self.server.pop_outbox(i)
                if msgs:
                    c.ctrl = ModelTransfer("ack", self.ack_bytes * len(msgs), msgs, t_start=t)
            if c.download is None and c.base_version < self.server.version and l.up:
                v = self.server.version
                c.download = ModelTransfer("download", self.model_bytes, (v, self.server.theta.clone()), t_start=t)
            # ---- per-AUV airtime arbitration (communication/capacity_arbiter.py): grants never exceed the pool
            grants = self._grants(c, l, env)
            self.last_alloc[i] = grants
            # ---- advance upload
            if c.upload is not None:
                tx_s += c.upload.step(t, self.dt, l, grants.get("upload", 0.0), self.frag)
                if c.upload.done_tx:
                    pkt = c.upload.payload
                    pkt.bytes_sent = float(c.upload.total_bytes)
                    pkt.transition(UpdateStatus.IN_FLIGHT, t)      # from here on only the server decides the outcome
                    self.inflight.append((c.upload, i))
                    self.n_transmitted += 1
                    c.stats["bytes_up"] += c.upload.total_bytes
                    c.upload = None
                elif t - c.upload.t_start > float(self.f["upload_timeout_s"]):
                    c.stats["timeouts"] += 1
                    c.stats["bytes_up"] += c.upload.progress_bytes          # bytes actually sent but wasted
                    c.upload.payload.bytes_sent = float(c.upload.progress_bytes)
                    pkt = c.expire_upload(t, "upload_timeout")
                    c.stats["failed"] += 1
                    c.upload = None
                    c.awaiting_global = False
                    self._settle(env, pkt, t)
            # ---- downlink control messages (receipt / commit / reject / expire): one transfer at a time, list payload
            if c.ctrl is not None:
                rx_s += c.ctrl.step(t, self.dt, l, grants.get("ctrl", 0.0), self.frag)
                if c.ctrl.done_tx and t >= c.ctrl.t_arrival:
                    c.stats["bytes_down"] += c.ctrl.total_bytes
                    for m in c.ctrl.payload:                        # in send order
                        c.handle_control(m, t)
                    c.ctrl = None
            # ---- advance download of a new global model
            if c.download is not None:
                rx_s += c.download.step(t, self.dt, l, grants.get("download", 0.0), self.frag)
                if c.download.done_tx and t >= c.download.t_arrival:
                    v, theta = c.download.payload
                    c.install_global(theta, v, float(self.f["download_blend"]))
                    c.stats["bytes_down"] += c.download.total_bytes
                    c.stats["sync_latencies"].append(t - self.server.t_version.get(v, t))
                    c.download = None
                    c.awaiting_global = False
            # A transfer that has sent its LAST bit and only waits for propagation (done_tx, t < t_arrival) occupies no airtime and
            # must not reserve any: only transfers with bytes still to send count (half duplex: any; full duplex: an upload).
            on_air = lambda tr: tr is not None and not tr.done_tx
            env.model_traffic_active[i] = on_air(c.upload) or on_air(c.download) or on_air(c.ctrl)
            env.model_uplink_active[i] = on_air(c.upload)           # direction-aware reservation of the sensor-data uplink
            if tx_s or rx_s:
                env.add_comm_energy(i, tx_s, rx_s)
                c.stats["tx_s"] += tx_s
                c.stats["rx_s"] += rx_s
        # ---- deliveries that have arrived at the USV
        still = []
        for tr, i in self.inflight:
            if t >= tr.t_arrival:
                pkt = tr.payload
                self.server.receive(pkt, t)
                self.clients[i].stats["latencies"].append(t - pkt.t_upload_start)
            else:
                still.append((tr, i))
        self.inflight = still
        rec = self.server.maybe_aggregate(t)
        if rec is not None:
            recs.append(rec)
        for pkt in self.server.drain_resolved():                    # terminal outcomes decided by the server this step
            self._settle(env, pkt, t)
        return recs

    # ------------------------------------------------------------------------------------------
    def pending_report(self) -> dict:
        """Census of unresolved updates by lifecycle state (read-only; used at the training cutoff and after the drain)."""
        by = {"uploading": 0, "in_flight": 0, "buffered": 0, "awaiting_terminal_outcome_at_client": 0}
        for c in self.clients:
            p = c.outstanding
            if p is None:
                continue
            if p.status is UpdateStatus.UPLOADING:
                by["uploading"] += 1
            elif p.status is UpdateStatus.IN_FLIGHT:
                by["in_flight"] += 1
            elif p.status is UpdateStatus.BUFFERED:
                by["buffered"] += 1
            else:                                               # server already decided (aggregated/rejected/expired); client not yet told
                by["awaiting_terminal_outcome_at_client"] += 1
        return {**by, "unresolved_total": sum(by.values()),
                "control_transfers_active": sum(1 for c in self.clients if c.ctrl is not None),
                "control_messages_queued": sum(len(v) for v in self.server.outbox.values()),
                "downloads_active": sum(1 for c in self.clients if c.download is not None),
                "global_version": self.server.version}

    def drain_active(self) -> bool:
        """True while something already created can still resolve (an unresolved update or a transfer in progress)."""
        r = self.pending_report()
        return bool(r["unresolved_total"] or r["control_transfers_active"] or r["control_messages_queued"] or r["downloads_active"])

    # ------------------------------------------------------------------------------------------
    def statistics_accounting(self) -> dict:
        """Conservation check: every created update is either settled once or still unresolved (held)."""
        held = [c.outstanding for c in self.clients if c.outstanding is not None and not c.outstanding.terminal]
        settled = sum(self.n_status.values())
        unsettled_terminal = [c.outstanding for c in self.clients
                              if c.outstanding is not None and c.outstanding.terminal and c.outstanding.stats_disposition == "held"]
        return {"created": self.n_created, "settled": settled, "unresolved": len(held),
                "terminal_unsettled": len(unsettled_terminal), "n_status": dict(self.n_status),
                "stats_disposition": dict(self.stats_disposition), "stats_mass": self.stats_mass,
                "balanced": self.n_created == settled + len(held) + len(unsettled_terminal)
                            and len(self.update_log) == settled}

    def summary(self) -> dict:
        s = self.server
        lat = [x for c in self.clients for x in c.stats["latencies"]]
        syn = [x for c in self.clients for x in c.stats["sync_latencies"]]
        stales = [d for r in s.history for d in r["staleness"]]
        agg, rej, exp = (self.n_status[k] for k in ("aggregated", "rejected", "expired"))
        return {
            "link_up_fraction": self.link_up_steps / max(self.link_total_steps, 1),
            # created -> transmitted -> received -> aggregated / rejected / expired (all server-authoritative)
            "attempted_updates": self.n_created,                    # = created (name kept for compatibility)
            "transmitted_updates": self.n_transmitted,
            "received_updates": s.n_received,
            "accepted_updates": s.n_accepted,                       # = aggregated (meaning unchanged)
            "aggregated_updates": agg,
            "rejected_updates": rej,
            "expired_updates": exp,
            "failed_updates": rej + exp,                            # CHANGED: rejected + expired (was: rejected + upload timeouts)
            "unresolved_updates": sum(1 for c in self.clients if c.outstanding is not None and not c.outstanding.terminal),
            "duplicate_deliveries_ignored": s.n_duplicates,
            "stats_consumed": self.stats_disposition["consumed"],
            "stats_returned": self.stats_disposition["returned"],
            "stats_discarded_episode": self.stats_disposition["discarded_episode"],
            "retry_attempts": int(sum(c.stats["retries"] for c in self.clients)),
            "retry_deferred_updates": int(sum(c.stats["retry_deferred"] for c in self.clients)),
            "retry_deferred_steps": int(sum(c.stats["retry_deferred_steps"] for c in self.clients)),
            "rejected_bytes": self.rejected_bytes,
            "expired_bytes": self.expired_bytes,
            "wasted_uplink_bytes": self.rejected_bytes + self.expired_bytes,
            "wasted_uplink_fraction": (self.rejected_bytes + self.expired_bytes) / max(sum(c.stats["bytes_up"] for c in self.clients), 1e-12),
            "backoff_s_total": float(sum(c.stats["backoff_s"] for c in self.clients)),
            "timeouts": sum(c.stats["timeouts"] for c in self.clients),
            "deferred_updates": int(sum(c.stats["deferred"] for c in self.clients)),
            "aggregation_rounds": s.version,
            "avg_staleness": float(np.mean(stales)) if stales else float("nan"),
            "avg_local_steps": float(np.mean([c.total_steps for c in self.clients])),
            "bytes_up": sum(c.stats["bytes_up"] for c in self.clients),
            "bytes_down": sum(c.stats["bytes_down"] for c in self.clients),
            "bytes_total": sum(c.stats["bytes_up"] + c.stats["bytes_down"] for c in self.clients),
            "comm_tx_s": sum(c.stats["tx_s"] for c in self.clients),
            "comm_rx_s": sum(c.stats["rx_s"] for c in self.clients),
            "upload_latency_mean_s": float(np.mean(lat)) if lat else float("nan"),
            "sync_latency_mean_s": float(np.mean(syn)) if syn else float("nan"),
        }
