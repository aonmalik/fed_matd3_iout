"""USV federated server: buffers RECEIVED updates, gates staleness, aggregates, logs every decision.

The server is the single authority for an update's state from IN_FLIGHT on.  Every state change goes through
``UpdatePacket.transition`` (terminal states have no outgoing edge => each outcome is produced exactly once) and queues a
``ControlMessage`` in the per-client outbox; the protocol delivers outbox messages over the acoustic downlink.

  receive()          IN_FLIGHT -> BUFFERED  (+ 'receipt' = transport ACK)   or  -> REJECTED (+ 'reject')
  aggregate_now()    BUFFERED  -> AGGREGATED (+ 'commit', carries the new global version)  or -> REJECTED (+ 'reject')
  expire_buffer()    BUFFERED  -> EXPIRED  (+ 'expire') after ``max_buffer_age_s``
A repeated delivery of the same update id is ignored (counted in ``n_duplicates``) and can never be aggregated twice.
Resolved packets are exposed once through ``drain_resolved`` so the protocol can settle their statistics exactly once.
"""
from __future__ import annotations

import numpy as np
import torch

from .aggregation import aggregate, l2_clip, normalise_with_cap, raw_weights
from .relevance import relevance
from .staleness import is_acceptable, staleness, staleness_factor
from .update_packet import ControlMessage, UpdatePacket, UpdateStatus


class USVServer:
    def __init__(self, init_params: torch.Tensor, fcfg, n_clients: int, use_semantics: bool = True):
        self.theta = init_params.clone()
        self.version = 0
        self.t_version = {0: 0.0}               # creation time of each global version (sync-latency metric)
        self.fcfg = fcfg
        self.M = n_clients
        self.use_sem = use_semantics
        self.buffer: list[UpdatePacket] = []
        self.round_start_t = 0.0
        self.history: list[dict] = []           # one record per aggregation round
        self.events: list[dict] = []            # every receive / reject / expire decision
        self.outbox: dict[int, list[ControlMessage]] = {i: [] for i in range(n_clients)}
        self._resolved: list[UpdatePacket] = []
        self.seen_uids: set[str] = set()
        self.n_received = self.n_accepted = self.n_rejected = self.n_expired = self.n_duplicates = 0
        self.max_buffer_age_s = float(fcfg.get("max_buffer_age_s", float("inf")))
        kind = fcfg["aggregation"]
        self.kind = kind
        clip = fcfg["clip_norm"]
        self.clip = None if clip is None else float(clip)

    # -------------------------------------------------------------- helpers
    def _send(self, kind: str, pkt: UpdatePacket, t: float, reason: str = "") -> None:
        self.outbox[pkt.client_id].append(ControlMessage(kind, pkt.uid, pkt.client_id, pkt.update_id, self.version,
                                                         reason, t))

    def _finish(self, pkt: UpdatePacket, status: UpdateStatus, kind: str, t: float, reason: str) -> None:
        pkt.transition(status, t, reason)
        self._send(kind, pkt, t, reason)
        self._resolved.append(pkt)

    def pop_outbox(self, client_id: int) -> list[ControlMessage]:
        msgs, self.outbox[client_id] = self.outbox[client_id], []
        return msgs

    def drain_resolved(self) -> list[UpdatePacket]:
        out, self._resolved = self._resolved, []
        return out

    # -------------------------------------------------------------- receive
    def receive(self, pkt: UpdatePacket, t: float) -> bool:
        """Called ONLY when the last bit of the update has arrived (after transmission + propagation)."""
        if pkt.uid in self.seen_uids or pkt.status is not UpdateStatus.IN_FLIGHT:
            self.n_duplicates += 1                       # duplicate delivery: no state change, no second outcome
            self.events.append({"t": t, "client": pkt.client_id, "update": pkt.update_id, "ok": False,
                                "reason": "duplicate", "server_version": self.version})
            return False
        self.seen_uids.add(pkt.uid)
        self.n_received += 1
        pkt.t_received = t
        ok, reason = True, "buffered"
        if pkt.delta.numel() != self.theta.numel():
            ok, reason = False, "shape_mismatch"
        elif not is_acceptable(staleness(self.version, pkt.base_version), int(self.fcfg["max_staleness"])):
            ok, reason = False, "stale_at_receipt"
        elif not torch.isfinite(pkt.delta).all():
            ok, reason = False, "non_finite"
        self.events.append({"t": t, "client": pkt.client_id, "update": pkt.update_id, "base": pkt.base_version,
                            "server_version": self.version, "ok": ok, "reason": reason})
        if ok:
            pkt.transition(UpdateStatus.BUFFERED, t, reason)
            self.buffer.append(pkt)
            self._send("receipt", pkt, t)                # transport ACK only; the client reference does NOT move
        else:
            self.n_rejected += 1
            self._finish(pkt, UpdateStatus.REJECTED, "reject", t, reason)
        return ok

    # -------------------------------------------------------------- buffer expiry
    def expire_buffer(self, t: float) -> int:
        keep, n = [], 0
        for p in self.buffer:
            if (t - p.t_received) > self.max_buffer_age_s:
                n += 1
                self.n_expired += 1
                self.events.append({"t": t, "client": p.client_id, "update": p.update_id, "ok": False,
                                    "reason": "expired_in_buffer", "server_version": self.version})
                self._finish(p, UpdateStatus.EXPIRED, "expire", t, "expired_in_buffer")
            else:
                keep.append(p)
        self.buffer = keep
        return n

    # -------------------------------------------------------------- aggregation trigger
    def ready(self, t: float) -> bool:
        if not self.buffer:
            return False
        if self.fcfg["mode"] == "sync":
            have = {p.client_id for p in self.buffer if p.base_version == self.version}
            return len(have) >= self.M or (t - self.round_start_t) >= float(self.fcfg["sync_timeout_s"])
        oldest = min(p.t_received for p in self.buffer)
        return len(self.buffer) >= int(self.fcfg["agg_buffer_size"]) or (t - oldest) >= float(self.fcfg["agg_timeout_s"])

    def maybe_aggregate(self, t: float) -> dict | None:
        self.expire_buffer(t)
        if not self.ready(t):
            return None
        return self.aggregate_now(t)

    def aggregate_now(self, t: float) -> dict | None:
        f = self.fcfg
        accepted, rejected = [], []
        for p in self.buffer:
            d = staleness(self.version, p.base_version)
            (accepted if is_acceptable(d, int(f["max_staleness"])) else rejected).append((p, d))
        self.buffer = []
        for p, d in rejected:
            self.n_rejected += 1
            self.events.append({"t": t, "client": p.client_id, "update": p.update_id, "base": p.base_version,
                                "server_version": self.version, "ok": False, "reason": "stale_at_aggregation", "staleness": d})
            self._finish(p, UpdateStatus.REJECTED, "reject", t, "stale_at_aggregation")
        if not accepted:                        # empty received-update set => no aggregation, version unchanged
            return None
        kind = self.kind
        n = [p.n_samples for p, _ in accepted]
        stale = [staleness_factor(d, f["kappa"]) if kind in ("stale", "semantic", "service", "s3a") else 1.0
                 for _, d in accepted]
        rel_kind = {"semantic": "semantic", "service": "service", "s3a": "s3a"}.get(kind, "none")
        rel = [relevance(p.stats, rel_kind, float(f["beta_sem"]), self.use_sem) for p, _ in accepted]
        alpha = float(f["alpha"]) if kind != "fedavg" else 1.0
        eta = float(f["eta"]) if rel_kind != "none" else 0.0
        z = raw_weights(n, stale, rel, alpha, eta, float(f["eps_m"]))
        w = normalise_with_cap(z, float(f["max_client_weight"]))
        diag = self._relevance_counterfactual(accepted, n, stale, alpha, rel_kind, w)       # DIAGNOSTIC ONLY (never applied)
        self.theta, scales = aggregate(self.theta, [p.delta for p, _ in accepted], w, float(f["server_lr"]), self.clip)
        self.version += 1
        self.t_version[self.version] = t
        self.round_start_t = t
        self.n_accepted += len(accepted)
        for p, _ in accepted:                   # definitive success: AGGREGATED + commit (carries the NEW version)
            self._finish(p, UpdateStatus.AGGREGATED, "commit", t, "aggregated")
        rec = {"round": self.version, "t": t, "kind": kind,
               "clients": [p.client_id for p, _ in accepted], "updates": [p.update_id for p, _ in accepted],
               "staleness": [d for _, d in accepted], "n": n, "stale_factor": stale, "relevance": rel, "Z": z.tolist(),
               "w": w.tolist(), "norms": [p.norm for p, _ in accepted], "clip_scale": scales, **diag,
               "rejected": [(p.client_id, d) for p, d in rejected], "theta_norm": float(self.theta.norm())}
        self.history.append(rec)
        return rec

    # -------------------------------------------------------------- diagnostics (no effect on the learning dynamics)
    def _relevance_counterfactual(self, accepted, n, stale, alpha, rel_kind, w) -> dict:
        """How much does the relevance factor change this round?  Counterfactual weights use relevance exponent 0 (factor == 1)
        with the SAME n, staleness and cap.  Reported: total-variation distance between the two weight vectors and the relative
        L2 size of the resulting parameter-update difference.  The counterfactual is NEVER used to update the model."""
        if rel_kind == "none":
            return {"cf_rel_tv": 0.0, "cf_rel_effect": 0.0}
        f = self.fcfg
        z0 = raw_weights(n, stale, np.ones(len(n)), alpha, 0.0, float(f["eps_m"]))
        w0 = normalise_with_cap(z0, float(f["max_client_weight"]))
        diff = torch.zeros_like(self.theta)
        act = torch.zeros_like(self.theta)
        for (p, _), a, b in zip(accepted, w, w0):
            c, _s = l2_clip(p.delta, self.clip)
            diff += float(a - b) * c
            act += float(a) * c
        return {"cf_rel_tv": float(0.5 * np.abs(np.asarray(w) - w0).sum()),
                "cf_rel_effect": float(diff.norm() / max(float(act.norm()), 1e-12)), "w_norel": w0.tolist()}
