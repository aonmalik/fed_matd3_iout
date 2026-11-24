"""Centralised / non-federated baselines (centralized_matd3, independent_td3, maddpg, mappo, greedy_matd3, hungarian_matd3)."""
from _common import *
from fed_matd3_iout.experiment import run_one

p = base_parser(__doc__)
p.add_argument("--method", default="centralized_matd3")
a = p.parse_args()
cfg = cfg_from(a)
for s in a.seeds:
    r = run_one(cfg, a.method, s, a.out, n_episodes=a.episodes)
    print("run dir:", r["run_dir"])
