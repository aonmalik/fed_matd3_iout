"""Aggregation diagnostics from a SHORT real training probe (default 6 episodes, one seed, one method).  NOT a pilot, NOT a comparison.
Writes docs/aggregation_diagnostics.json.  Early-training statistics: later in training relevance may behave differently."""
from _common import *
import argparse, json
from fed_matd3_iout.validation.lifecycle_scenarios import training_probe
p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=6)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--method", default="s3a_fedmatd3")
p.add_argument("--out", default=str(ROOT / "docs" / "aggregation_diagnostics.json"))
a = p.parse_args()
cfg = load_config()
r = training_probe(cfg, a.episodes, a.method, a.seed, with_diagnostics=True)
from fed_matd3_iout.validation.report import _git, source_tree_hash
from fed_matd3_iout.utils.config import config_hash
r["provenance"] = {"commit": _git(ROOT, "rev-parse", "HEAD"), "branch": _git(ROOT, "rev-parse", "--abbrev-ref", "HEAD"),
                   "working_tree_clean": _git(ROOT, "status", "--porcelain") == "", "config_hash": config_hash(cfg),
                   "protocol_version": str(cfg.experiment.protocol_version), "source_tree_hash": source_tree_hash(ROOT),
                   "episodes": a.episodes, "seed": a.seed, "method": a.method}
r["note"] = "short probe at main-protocol episode length; descriptive diagnostics only; no performance claim"
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(r, open(a.out, "w"), indent=2, default=float)
d = r["aggregation_diagnostics"]
print(json.dumps({k: d[k] for k in ("n_rounds", "single_update_round_fraction", "multi_update_rounds", "participants_per_round",
                                    "relevance_factor_g", "weight_variance_shares_multi_rounds", "counterfactual_relevance_effect", "verdict")}, indent=1, default=float))
print("retry counters:", r["retry_counters"])
