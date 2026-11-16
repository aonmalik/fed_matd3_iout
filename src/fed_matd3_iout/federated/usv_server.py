"""USV federated server: buffers RECEIVED updates, gates staleness, aggregates, logs every decision."""
from __future__ import annotations

import math

import numpy as np
import torch

from .aggregation import aggregate, normalise_with_cap, raw_weights
from .relevance import relevance
from .staleness import is_acceptable, staleness, staleness_factor
from .update_packet import UpdatePacket


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
        self.events: list[dict] = []            # every receive / reject decision
        self.n_received = self.n_accepted = self.n_rejected = 0
        kind = fcfg["aggregation"]
        self.kind = kind
        clip = fcfg["clip_norm"]
        self.clip = None if clip is None else float(clip)

    # -------------------------------------------------------------- receive
    def receive(self, pkt: UpdatePacket, t: float) -> bool:
        """Called ONLY when the last bit of the update has arrived (after transmission + propagation)."""
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
            self.buffer.append(pkt)
        else:
            self.n_rejected += 1
        return ok

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
        self.theta, scales = aggregate(self.theta, [p.delta for p, _ in accepted], w, float(f["server_lr"]), self.clip)
        self.version += 1
        self.t_version[self.version] = t
        self.round_start_t = t
        self.n_accepted += len(accepted)
        rec = {"round": self.version, "t": t, "kind": kind,
               "clients": [p.client_id for p, _ in accepted], "updates": [p.update_id for p, _ in accepted],
               "staleness": [d for _, d in accepted], "n": n, "stale_factor": stale, "relevance": rel, "Z": z.tolist(),
               "w": w.tolist(), "norms": [p.norm for p, _ in accepted], "clip_scale": scales,
               "rejected": [(p.client_id, d) for p, d in rejected], "theta_norm": float(self.theta.norm())}
        self.history.append(rec)
        return rec
