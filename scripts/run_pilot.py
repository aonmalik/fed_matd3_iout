"""Pilot: a few methods, one training seed, short budget, + goal-seeking reference. NOT evidence of superiority."""
from _common import *
from fed_matd3_iout.baselines.reference_controller import goal_seeking_policies
from fed_matd3_iout.evaluation.evaluator import evaluate, summarise
from fed_matd3_iout.experiment import run_one
import json
p = base_parser(__doc__)
p.add_argument("--methods", nargs="*", default=["centralized_matd3", "s3a_fedmatd3"])
a = p.parse_args()
cfg = cfg_from(a)
ref = summarise(evaluate(cfg, goal_seeking_policies(cfg.env.n_auvs)))
print("REFERENCE goal-seeking:", {k: round(ref[k], 3) for k in ("U_total", "delivered_mbit", "overflow_mbit", "collection_ratio")}, flush=True)
for m in a.methods:
    for s in a.seeds:
        r = run_one(cfg, m, s, Path(a.out) / "pilot", n_episodes=a.episodes, verbose=False)
        g = summarise(r["global"])
        print(m, "seed", s, {k: round(g[k], 3) for k in ("U_total", "delivered_mbit", "overflow_mbit", "collection_ratio")}, "comm:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r["comm"].items() if k in ("received_updates", "aggregation_rounds", "avg_staleness")}, flush=True)
