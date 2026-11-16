"""Federated training. Usage: python scripts/train_federated.py --method s3a_fedmatd3|fedavg_matd3|... [--config smoke.yaml]"""
from _common import *
from fed_matd3_iout.experiment import run_one

p = base_parser(__doc__)
p.add_argument("--method", default="s3a_fedmatd3")
a = p.parse_args()
cfg = cfg_from(a)
for s in a.seeds:
    r = run_one(cfg, a.method, s, a.out, n_episodes=a.episodes)
    print("run dir:", r["run_dir"], "| comm:", r["comm"])
