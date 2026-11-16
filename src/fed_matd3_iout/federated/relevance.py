"""Semantic-service relevance M_i in [0,1] from MEASURED rollout statistics (Eq. F4, class C: proposed).

Window = interval since this client's previous update. Quantities are accumulated by the simulator:
  U_del   : decayed, criticality-weighted utility of data DELIVERED by AUV i to the USV (each packet delivered once)
  V_coll  : undecayed value sum(c*sigma*units) of data COLLECTED by AUV i from sensors
  C       : volume collected by AUV i [Mbit];   L_own : overflow at sensors OWNED (assigned) by AUV i [Mbit]
  rho_sem = clip(U_del / V_coll, 0, 1)     delivery fidelity of the semantic value the AUV picked up
  rho_svc = C / (C + L_own)                service effectiveness at its own sensors (collection vs. loss)
  M_i     = beta*rho_sem + (1-beta)*rho_svc          ('s3a');  'semantic' -> rho_sem;  'service' -> rho_svc
No data is added twice: rho_sem measures *value* fidelity after collection, rho_svc compares collection with
loss at owned sensors; delivered volume does not enter rho_svc.  KNOWN LIMITATION (class D): U_del and V_coll are
windowed independently, so a delivery lag can make rho_sem exceed 1 (clipped) -- documented, not hidden.
Empty window (V_coll = 0 and C + L_own = 0) => M_i = 0 (floored by eps_M in the weight).
"""
from __future__ import annotations


def rho_sem(stats: dict, use_semantics: bool = True) -> float:
    if use_semantics:
        num, den = stats.get("U_del", 0.0), stats.get("V_coll", 0.0)
    else:                                    # semantics-off ablation: volume fidelity
        num, den = stats.get("units_del", 0.0), stats.get("units_coll", 0.0)
    return float(min(max(num / den, 0.0), 1.0)) if den > 0 else 0.0


def rho_svc(stats: dict) -> float:
    c, l = stats.get("coll_mbit", 0.0), stats.get("owned_loss_mbit", 0.0)
    return float(c / (c + l)) if (c + l) > 0 else 0.0


def relevance(stats: dict, kind: str, beta: float, use_semantics: bool = True) -> float:
    if kind == "semantic":
        return rho_sem(stats, use_semantics)
    if kind == "service":
        return rho_svc(stats)
    if kind == "s3a":
        return float(beta * rho_sem(stats, use_semantics) + (1.0 - beta) * rho_svc(stats))
    return 1.0                               # fedavg / stale: no relevance term
