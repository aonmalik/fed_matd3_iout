"""All 11 methods x training seeds on matched evaluation missions -> results.json, table, paired tests."""
from _common import *
from fed_matd3_iout.baselines.registry import METHODS
from fed_matd3_iout.evaluation import report_generator as rg
from fed_matd3_iout.experiment import run_one
from fed_matd3_iout.utils.reproducibility import atomic_write_json

p = base_parser(__doc__)
p.add_argument("--methods", nargs="*", default=list(METHODS))
p.add_argument("--which", choices=["global", "local"], default="global")
a = p.parse_args()
cfg = cfg_from(a)
res, comm = {}, {}
for m in a.methods:
    res[m], comm[m] = {}, {}
    for s in a.seeds:
        r = run_one(cfg, m, s, Path(a.out) / "runs", n_episodes=a.episodes, verbose=False)
        res[m][s] = r[a.which]
        comm[m][s] = r["comm"]
        print(f"{m} seed {s} done", flush=True)
d = save_results(res, a.out, "all_methods")
atomic_write_json(d / "comm.json", comm)
if "s3a_fedmatd3" in res and len(a.seeds) > 1:
    atomic_write_json(d / "paired_tests.json", rg.comparisons(res, "s3a_fedmatd3"))
else:
    print("NOTE: paired tests need >= 2 training seeds; none computed. Do not claim significance from this run.")
print("results:", d)
