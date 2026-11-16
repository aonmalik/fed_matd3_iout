"""Ablations of the proposed method (definitions: baselines/registry.py::ABLATIONS)."""
from _common import *
from fed_matd3_iout.baselines.registry import ABLATIONS
from fed_matd3_iout.evaluation import report_generator as rg
from fed_matd3_iout.experiment import run_one
from fed_matd3_iout.utils.reproducibility import atomic_write_json

p = base_parser(__doc__)
p.add_argument("--only", nargs="*", default=list(ABLATIONS))
a = p.parse_args()
cfg = cfg_from(a)
res = {}
for name in a.only:
    res[name] = {}
    for s in a.seeds:
        r = run_one(cfg, "s3a_fedmatd3", s, Path(a.out) / "runs", overrides=ABLATIONS[name], tag=f"abl_{name}",
                    n_episodes=a.episodes, verbose=False)
        res[name][s] = r["global"]
        print(f"ablation {name} seed {s} done", flush=True)
d = save_results(res, a.out, "ablation")
if "full" in res and len(a.seeds) > 1:
    atomic_write_json(d / "paired_tests.json", rg.comparisons(res, "full"))
print("results:", d)
