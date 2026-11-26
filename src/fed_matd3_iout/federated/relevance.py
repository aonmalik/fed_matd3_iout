"""Semantic-service relevance M_i in [0,1] from MEASURED, batch-attributed rollout statistics (Eq. F4, class C: proposed).

Inputs are the statistics frozen into an update when it was created (``UpdatePacket.stats``):

  Semantic part -- from the batch ledger (semantic/attribution.py).  Over the set S_i of MATURE batches that AUV i
  collected (fully delivered, or older than ln(1/eps)/mu so that waiting longer cannot change their value by more
  than a fraction eps):
      U_mature = sum_S  delivered  c sigma exp(-mu age)        realised utility of exactly those batches
      V_mature = sum_S  collected  c sigma                      undecayed value of exactly those batches
      rho_sem  = U_mature / V_mature                            in [0, 1] BY CONSTRUCTION (age >= 0, delivered <= collected)
  Numerator and denominator describe the SAME batches, so delayed delivery is credited to the collecting AUV and to the
  window in which the data was collected.  Immature batches are neither counted nor treated as failures.

  Service part -- same time window, both quantities are sensor-side events of the same AUV:
      rho_svc = C / (C + L_own)      C: Mbit collected by the AUV;  L_own: overflow Mbit at sensors it owned (was assigned)

  M_i (kind 's3a')  = beta rho_sem + (1-beta) rho_svc      ('semantic' -> rho_sem ; 'service' -> rho_svc)

Undefined components.  If a component has no evidence (V_mature = 0, or C + L_own = 0) it is *undefined*, not zero: 's3a'
then uses the defined component alone; if both are undefined, or a single-component kind is undefined, M_i = 0 (and the
weight floor eps_M still gives the update a positive weight).  This avoids punishing a client for batches that simply have
not matured yet.

No clipping.  The ratio is never clipped to hide inconsistent accounting: a value above 1 (beyond numerical tolerance) means
the ledger is broken and raises ``ValueError``.  The final ``min`` only removes floating-point noise (<= 1e-9).

Semantics-off ablation: rho_sem uses delivered/collected UNITS of the same mature batches.

Known limitations (class D): rho_svc depends on the allocator (overflow at an AUV's own sensors is not only its policy);
the maturity horizon is a modelling choice (eps); batches delivered after being claimed by age are not re-attributed.
"""
from __future__ import annotations

_TOL = 1e-9


def _ratio(num: float, den: float, what: str) -> float | None:
    if den <= 0.0:
        return None
    if num < -_TOL or num > den * (1.0 + _TOL):
        raise ValueError(f"inconsistent attribution for {what}: numerator {num} outside [0, {den}]")
    return min(max(num / den, 0.0), 1.0)          # floating-point guard only (validated above)


def rho_sem(stats: dict, use_semantics: bool = True) -> float | None:
    if use_semantics:
        return _ratio(stats.get("U_mature", 0.0), stats.get("V_mature", 0.0), "rho_sem (utility)")
    return _ratio(stats.get("units_del_mature", 0.0), stats.get("units_coll_mature", 0.0), "rho_sem (units)")


def rho_svc(stats: dict) -> float | None:
    c, l = stats.get("coll_mbit", 0.0), stats.get("owned_loss_mbit", 0.0)
    return float(c / (c + l)) if (c + l) > 0 else None


def relevance(stats: dict, kind: str, beta: float, use_semantics: bool = True) -> float:
    s, v = rho_sem(stats, use_semantics), rho_svc(stats)
    if kind == "semantic":
        return 0.0 if s is None else float(s)
    if kind == "service":
        return 0.0 if v is None else float(v)
    if kind == "s3a":
        if s is None and v is None:
            return 0.0
        if s is None:
            return float(v)
        if v is None:
            return float(s)
        return float(beta * s + (1.0 - beta) * v)
    return 1.0                                   # fedavg / stale: no relevance term
