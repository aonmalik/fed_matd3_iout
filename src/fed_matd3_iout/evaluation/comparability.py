"""Which method pairs are like-for-like, and which differences (declared non-equivalences N1-N10) a comparison inherits.

A paired test on matched training seeds says nothing about WHY two methods differ.  Every row written by
``report_generator.comparisons`` therefore carries the declared non-equivalences that separate the two methods, so that a
difference can never be reported as 'the effect of semantic weighting' when the pair also differs in deployment or trainer.

Headline policy per method (what ``--which global`` scores):
  federated methods                -> ONE global actor deployed to every AUV            (N5; local actors reported separately)
  other MATD3-family methods       -> each AUV's own actor
  MAPPO                            -> one shared actor
"""
from __future__ import annotations

FEDERATED = ("fedavg_matd3", "stale_fedmatd3", "semantic_fedmatd3", "service_fedmatd3", "s3a_fedmatd3")
ATTR = {  # deployment, trainer, allocator, critic information, TD3 stabilisers
    **{m: ("global_federated_actor", "matd3", "sta", "ctde", True) for m in FEDERATED},
    "centralized_matd3": ("per_auv_local_actors", "matd3", "sta", "ctde", True),
    "greedy_matd3": ("per_auv_local_actors", "matd3", "greedy", "ctde", True),
    "hungarian_matd3": ("per_auv_local_actors", "matd3", "hungarian", "ctde", True),
    "independent_td3": ("per_auv_local_actors", "matd3", "sta", "independent", True),
    "maddpg": ("per_auv_local_actors", "matd3", "sta", "ctde", False),
    "mappo": ("shared_actor", "mappo", "sta", "ctde", True),
}


def deployment(method: str) -> str:
    return ATTR[method][0]


def differences(a: str, b: str) -> list[str]:
    """Declared non-equivalences (ids refer to validation.fairness.NON_EQUIVALENT) that separate methods a and b."""
    (da, ta, la, ca, sa), (db, tb, lb, cb, sb) = ATTR[a], ATTR[b]
    out = []
    if da != db and "global_federated_actor" in (da, db):
        out.append("N5 deployment: global federated actor vs non-federated actors (and N7 model-traffic cost, N6 initialisation)")
    if (ta == "mappo") != (tb == "mappo"):
        out.append("N2/N3/N4 MAPPO: different update budget, policy class and parameter sharing")
    if la != lb:
        out.append("N1 task allocator differs")
    if ca != cb:
        out.append("N9 critic information differs (independent critic)")
    if sa != sb:
        out.append("N10 TD3 stabilisers differ (MADDPG)")
    return out


def comparability(a: str, b: str) -> dict:
    d = differences(a, b)
    both_fed = a in FEDERATED and b in FEDERATED
    return {"class": "like_for_like_weighting_only" if both_fed and not d else ("like_for_like" if not d else "NOT_like_for_like"),
            "inherited_non_equivalences": d, "deployment_a": deployment(a), "deployment_b": deployment(b)}
