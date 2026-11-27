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


# ----------------------------------------------------------------------------------------------------- diagnostics (never used by the update rule)
SEM_NO_COLLECTION = "no_collection"
SEM_COLLECTED_NO_MATURE = "collection_no_mature_evidence"
SEM_MATURE_EVIDENCE = "mature_attribution_evidence"   # neutral: semantic-utility OR unit-based attribution, see "attribution_mode"
SVC_NO_FLOW = "no_owned_service_flow"
SVC_EVIDENCE = "service_evidence"
SEM_STATUSES = (SEM_NO_COLLECTION, SEM_COLLECTED_NO_MATURE, SEM_MATURE_EVIDENCE)
SVC_STATUSES = (SVC_NO_FLOW, SVC_EVIDENCE)


def relevance_status(stats: dict, use_semantics: bool = True) -> dict:
    """DIAGNOSTIC classification of WHY a relevance component is defined or undefined.  It reads the same frozen ``UpdatePacket.stats``
    as ``relevance`` and never feeds back into any weight; it does not alter rho_sem / rho_svc / M.

    Semantic status (mutually exclusive, evaluated in this order):
      mature_attribution_evidence    rho_sem is defined (mature batches with positive value exist in this update's claim).  The label is
                                     deliberately neutral: with ``use_semantics=True`` the evidence is semantic-utility attribution
                                     (U_mature / V_mature); with ``use_semantics=False`` (semantics-off ablation) it is UNIT-based
                                     (units_del_mature / units_coll_mature) and carries no semantic information.  The field
                                     ``attribution_mode`` ('semantic_utility' | 'unit_based') says which.
      collection_no_mature_evidence  rho_sem undefined, but data was collected in the window (coll_mbit > 0) or collected batches are still
                                     open (n_immature_batches > 0, i.e. waiting for delivery/maturity -- NOT a failure)
      no_collection                  rho_sem undefined and neither collected data in the window nor open batches
    Service status:
      service_evidence               rho_svc defined (C + L_own > 0); its value MAY be 0 (collected nothing while owned sensors overflowed)
      no_owned_service_flow          rho_svc undefined (C + L_own == 0)
    ``n_immature_batches`` counts the client's open batches at claim time (including batches collected in earlier windows); stats that
    lack the key (older runs) are treated as 0 immature batches.  An undefined component is never reported as zero relevance here.
    """
    s, v = rho_sem(stats, use_semantics), rho_svc(stats)
    if s is not None:
        sem = SEM_MATURE_EVIDENCE
    elif stats.get("coll_mbit", 0.0) > 0.0 or stats.get("n_immature_batches", 0.0) > 0.0:
        sem = SEM_COLLECTED_NO_MATURE
    else:
        sem = SEM_NO_COLLECTION
    return {"sem": sem, "svc": SVC_EVIDENCE if v is not None else SVC_NO_FLOW,
            "attribution_mode": "semantic_utility" if use_semantics else "unit_based",
            "rho_sem_defined": s is not None, "rho_svc_defined": v is not None,
            "svc_zero_with_evidence": v is not None and v == 0.0,
            "coll_mbit": float(stats.get("coll_mbit", 0.0)), "owned_loss_mbit": float(stats.get("owned_loss_mbit", 0.0)),
            "n_mature_batches": float(stats.get("n_mature_batches", 0.0)), "n_immature_batches": float(stats.get("n_immature_batches", 0.0))}
