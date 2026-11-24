"""End-to-end smoke test (tiny CPU config). Demonstrates plumbing; NOT a scientific experiment.
Verifies each required capability and exits non-zero if any check fails."""
from _common import *
import numpy as np, torch
from fed_matd3_iout.baselines.registry import build_cfg
from fed_matd3_iout.experiment import run_one
from fed_matd3_iout.environment.multi_auv_iout_env import MultiAUVIoUTEnv
from fed_matd3_iout.training.checkpoint import load_checkpoint

p = base_parser(__doc__)
a = p.parse_args()
cfg = load_config(["smoke.yaml"] + a.config, parse_overrides(a.set))
ck = {}
env = MultiAUVIoUTEnv(cfg)
env.reset(0)
ck["1 env init (3 AUV, 1 USV, 10 sensors)"] = env.M == 3 and env.N == 10 and env.usv is not None
ck["2 sensor data generation"] = sum(s.queue.generated for s in env.sensors) > 0
r = run_one(cfg, "s3a_fedmatd3", 0, a.out, n_episodes=a.episodes, verbose=True)
rd = r["run_dir"]
import json, pathlib
from fed_matd3_iout.environment.multi_auv_iout_env import MultiAUVIoUTEnv as E
agg = json.load(open(next(pathlib.Path(rd).glob("aggregation_history_final.json"))))
comm = r["comm"]
g = r["global"][0]
# plumbing check of the physical layer with a goal-seeking controller (the 3-episode policy is untrained)
env.reset(0)
for _ in range(cfg.env.episode_steps):
    act = np.zeros((env.M, 2))
    for i in range(env.M):
        gp = env.goal_pos(i)
        if gp is not None:
            v = gp - env.auvs[i].pos
            act[i] = [1.0, np.arctan2(v[1], v[0]) / np.pi]
    env.step(act)
acc = env.accounting()
ck["3 task allocation events (>0)"] = env.allocator.events > 0
ck["4 AUV motion (travelled>0)"] = float(env.travel_by_auv.sum()) > 0
ck["5 sensor service (collected>0)"] = acc["collected"] > 0
ck["6 semantic utility computed (delivered, U>0, mass balance)"] = env.ledger.U > 0 and abs(acc["e2e_balance_err"]) < 1e-9
ck["6b learning-run evaluation reports semantic metrics"] = "U_norm" in g
ck["7 local MATD3 update ran"] = comm.get("avg_local_steps", 0) > 0
ck["8 intermittent communication (0 < link-up fraction < 1)"] = 0.0 < comm.get("link_up_fraction", 1.0) < 1.0
ck["9 successful federated upload"] = comm.get("received_updates", 0) > 0
ck["10 USV aggregation"] = len(agg) > 0 and comm.get("aggregation_rounds", 0) > 0
ck["11 global actor synchronisation"] = comm.get("bytes_down", 0) > 0
ck["12 checkpoint saved"] = pathlib.Path(r["checkpoint"]).exists() and "global_actor" in load_checkpoint(r["checkpoint"])
ck["13 evaluation"] = len(r["global"]) == cfg.experiment.eval_scenarios and g["mass_balance_err"] < 1e-6
print("\n=== SMOKE TEST ===")
for k, v in ck.items():
    print(("PASS " if v else "FAIL ") + k)
print("comm summary:", comm)
sys.exit(0 if all(ck.values()) else 1)
