"""Asynchronous federated communication protocol over intermittent acoustic links (steps 1-7 of the spec).

Per client:  local learning (always)  ->  update ready  ->  wait for feasible contact (contact scheduler)  ->
upload (fragmented over windows, loses packets to fading)  ->  USV receives (after last bit + propagation)  ->
ACK download  ->  continue.  Independently the USV pushes new global versions via download transfers when the
client's link is open.  Nothing is aggregated before it is successfully received.

Update creation: the delta is snapshotted at the moment its transmission STARTS (freshest local model); the
base version is that of the last installed global model.  One outstanding update per client.
"""
from __future__ import annotations

import numpy as np

from ..communication.contact_scheduler import ContactScheduler
from ..communication.model_transfer import ModelTransfer
from .federated_client import FederatedClient
from .usv_server import USVServer


class FederatedProtocol:
    def __init__(self, clients: list[FederatedClient], server: USVServer, fcfg, usv_cfg, dt: float):
        self.clients, self.server, self.f = clients, server, fcfg
        self.share = float(usv_cfg["model_share"])
        self.frag = bool(fcfg["allow_fragmentation"])
        self.sched = ContactScheduler(len(clients), dt, self.frag)
        self.dt = dt
        self.inflight: list[tuple[ModelTransfer, int]] = []       # transfers whose last bit is in the air
        self.comm_log: list[dict] = []
        self.link_up_steps = 0
        self.link_total_steps = 0
        self.ack_bytes = float(fcfg["ack_bytes"])
        self.model_bytes = clients[0].model_params * float(fcfg["model_bytes_per_param"])

    # ------------------------------------------------------------------------------------------
    def step(self, env, t: float) -> list[dict]:
        """Advance all transfers by one step of global time ``t`` (seconds). Returns aggregation records."""
        links = env.links
        self.sched.observe(links)
        self.link_up_steps += sum(1 for l in links if l.up)
        self.link_total_steps += len(links)
        recs = []
        for c in self.clients:
            i = c.id
            l = links[i]
            tx_s = rx_s = 0.0
            # ---- start a new upload when an update is ready and the contact is feasible
            if c.upload is None and c.ack is None and c.update_ready() and not (self.f["mode"] == "sync" and c.awaiting_global):
                if self.sched.may_start(i, self.model_bytes, l, self.share):
                    pkt = c.make_update(t, env.pop_window_stats(i))
                    pkt.t_upload_start = t
                    c.upload = ModelTransfer("upload", pkt.size_bytes, pkt, t_start=t)
                    c.stats["attempted"] += 1
                    c.awaiting_global = True
                elif l.up:
                    c.stats["deferred"] += 1
            # ---- advance upload
            if c.upload is not None:
                tx_s += c.upload.step(t, self.dt, l, self.share, self.frag)
                if c.upload.done_tx:
                    self.inflight.append((c.upload, i))
                    c.stats["bytes_up"] += c.upload.total_bytes
                    c.upload = None
                elif t - c.upload.t_start > float(self.f["upload_timeout_s"]):
                    c.stats["timeouts"] += 1
                    c.stats["failed"] += 1
                    c.stats["bytes_up"] += c.upload.progress_bytes          # bytes actually sent but wasted
                    pkt = c.upload.payload
                    # restore: the window statistics of an unsent update are not lost
                    env.win_restore(i, pkt.stats) if hasattr(env, "win_restore") else None
                    c.upload = None
                    c.awaiting_global = False
            # ---- advance ACK
            if c.ack is not None:
                rx_s += c.ack.step(t, self.dt, l, self.share, self.frag)
                if c.ack.done_tx and t >= c.ack.t_arrival:
                    pkt = c.ack.payload
                    c.stats["bytes_down"] += c.ack.total_bytes
                    if c.base_version == pkt.base_version:                 # no global installed in between
                        c.reference = pkt.snapshot.clone()
                        c.steps_since = 0
                        c.trans_since = 0
                    c.ack = None
                    c.awaiting_global = c.awaiting_global and self.f["mode"] == "sync"
            # ---- advance download of a new global model
            if c.download is None and c.base_version < self.server.version and l.up:
                v = self.server.version
                c.download = ModelTransfer("download", self.model_bytes, (v, self.server.theta.clone()), t_start=t)
            if c.download is not None:
                rx_s += c.download.step(t, self.dt, l, self.share, self.frag)
                if c.download.done_tx and t >= c.download.t_arrival:
                    v, theta = c.download.payload
                    c.install_global(theta, v, float(self.f["download_blend"]))
                    c.stats["bytes_down"] += c.download.total_bytes
                    c.stats["sync_latencies"].append(t - self.server.t_version.get(v, t))
                    c.download = None
                    c.awaiting_global = False
            env.model_traffic_active[i] = (c.upload is not None) or (c.download is not None) or (c.ack is not None)
            if tx_s or rx_s:
                env.add_comm_energy(i, tx_s, rx_s)
                c.stats["tx_s"] += tx_s
                c.stats["rx_s"] += rx_s
        # ---- deliveries that have arrived at the USV
        still = []
        for tr, i in self.inflight:
            if t >= tr.t_arrival:
                pkt = tr.payload
                c = self.clients[i]
                ok = self.server.receive(pkt, t)
                c.stats["succeeded" if ok else "failed"] += 1
                c.stats["latencies"].append(t - pkt.t_upload_start)
                c.ack = ModelTransfer("ack", self.ack_bytes, pkt, t_start=t)
            else:
                still.append((tr, i))
        self.inflight = still
        rec = self.server.maybe_aggregate(t)
        if rec is not None:
            recs.append(rec)
        return recs

    # ------------------------------------------------------------------------------------------
    def summary(self) -> dict:
        s = self.server
        att = sum(c.stats["attempted"] for c in self.clients)
        lat = [x for c in self.clients for x in c.stats["latencies"]]
        syn = [x for c in self.clients for x in c.stats["sync_latencies"]]
        stales = [d for r in s.history for d in r["staleness"]]
        return {
            "link_up_fraction": self.link_up_steps / max(self.link_total_steps, 1),
            "attempted_updates": att,
            "received_updates": s.n_received, "accepted_updates": s.n_accepted, "rejected_updates": s.n_rejected,
            "failed_updates": sum(c.stats["failed"] for c in self.clients),
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
