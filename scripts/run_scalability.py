"""Network-size scaling with exactly ONE USV. Sizes from configs/experiment_manifest.json (or --sizes M:N ...)."""
from _common import *
import json
from fed_matd3_iout.experiment import run_one

p = base_parser(__doc__)
p.add_argument("--methods", nargs="*", default=["centralized_matd3", "fedavg_matd3", "s3a_fedmatd3"])
p.add_argument("--sizes", nargs="*", default=None, help="M:N pairs, e.g. 1:10 3:20 10:40")
a = p.parse_args()
cfg = cfg_from(a)
man = json.load(open(ROOT / "configs" / "experiment_manifest.json"))
sizes = [tuple(map(int, s.split(":"))) for s in a.sizes] if a.sizes else [(x["n_auvs"], x["n_sensors"]) for x in man["scalability"]]
res = {}
for M, N in sizes:
    for m in a.methods:
        tag = f"{m}@{M}auv_{N}sen_1usv"
        res[tag] = {}
        for s in a.seeds:
            r = run_one(cfg, m, s, Path(a.out) / "runs", overrides={"env.n_auvs": M, "env.n_sensors": N}, tag=tag.replace("@", "_"),
                        n_episodes=a.episodes, verbose=False)
            res[tag][s] = r["global"]
            print(tag, "seed", s, "done", flush=True)
d = save_results(res, a.out, "scalability")
print("results:", d)
