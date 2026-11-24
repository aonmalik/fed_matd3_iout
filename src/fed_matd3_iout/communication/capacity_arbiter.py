"""Deterministic per-AUV airtime arbitration for model traffic (Protocol 2.2).

Why: in Protocol 2.1 an AUV's model upload, its control-message transfer and the global-model download each advanced with the
FULL ``model_share`` of the link, so up to 3 x model_share of the link could be consumed in one step while the sensor-data
uplink was reduced only by one ``model_share``.  The arbiter makes ``model_share`` a *pool* that all model transfers of one
AUV share, so that the airtime handed out in a step never exceeds the pool.

Model (all quantities are fractions of ONE step's airtime on the AUV<->USV channel, 0 <= x <= 1):
  * pool        = ``model_share``  (the configured fraction of link capacity given to model traffic when any is active)
  * demand_k    = airtime transfer k still needs, as a fraction of the step, at the full link rate (>= 0)
  * grant_k     = what k receives;   sum_k grant_k <= pool   (conservation, tested)   and   grant_k <= demand_k
Policy (deterministic, no randomness):
  1. 'ctrl' (tiny acknowledgements) is served first, up to its demand, but never more than HALF the pool while another transfer
     is waiting: acknowledgements are a few bytes (so they normally finish in one step) and serving them first keeps the update
     lifecycle from stalling behind a model transfer; the half-pool cap guarantees that no download/upload is ever starved.
  2. The remaining pool is shared between the other active transfers by max-min fairness (water-filling): equal split,
     any transfer needing less than its equal split releases the surplus to the others.
Starvation freedom: while the pool is positive every transfer with positive demand receives a strictly positive grant each step.
"""
from __future__ import annotations

PRIORITY = ("ctrl",)          # served first; everything else is max-min fair


def arbitrate(demands: dict[str, float], pool: float) -> dict[str, float]:
    """Return per-transfer grants (same keys as ``demands``), sum(grants) <= pool, grant <= demand."""
    pool = max(float(pool), 0.0)
    grants = {k: 0.0 for k in demands}
    left = pool
    others_waiting = any(demands[k] > 0.0 for k in demands if k not in PRIORITY)
    for k in PRIORITY:
        if k in demands:
            cap = left / 2.0 if others_waiting else left            # priority can never take more than half the pool from others
            g = min(max(demands[k], 0.0), cap)
            grants[k] = g
            left -= g
    rest = sorted(k for k in demands if k not in PRIORITY and demands[k] > 0.0)     # sorted => deterministic order
    # water-filling: smallest demands first
    rest.sort(key=lambda k: (demands[k], k))
    n = len(rest)
    for idx, k in enumerate(rest):
        fair = left / (n - idx)
        g = min(demands[k], fair)
        grants[k] = g
        left -= g
    return grants
