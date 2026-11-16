"""Evaluate a saved run on matched scenarios. Usage: python scripts/evaluate.py --run-dir results/<run> [--ckpt final] [--n 20]
Writes a NEW evaluation file; never modifies the run."""
from _common import *
import json, torch
from fed_matd3_iout.agents.actor import Actor, load_flat_params
from fed_matd3_iout.evaluation.evaluator import evaluate, summarise
from fed_matd3_iout.training.checkpoint import load_checkpoint
from fed_matd3_iout.utils.config import _wrap
from fed_matd3_iout.utils.reproducibility import atomic_write_json, new_run_dir
from fed_matd3_iout.environment.multi_auv_iout_env import MultiAUVIoUTEnv

p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True)
p.add_argument("--ckpt", default="final")
p.add_argument("--n", type=int, default=None)
a = p.parse_args()
rd = Path(a.run_dir)
man = json.load(open(rd / "run_manifest.json"))
cfg = _wrap(man["config"])
ck = load_checkpoint(rd / "checkpoints" / f"{a.ckpt}.pt")
env = MultiAUVIoUTEnv(cfg)
def mk(flat):
    net = Actor(env.obs_dim, env.act_dim, int(cfg.matd3.hidden)); load_flat_params(net, flat)
    return lambda o, net=net: net(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))[0].detach().numpy()
out = {}
if "actors" in ck:
    out["local"] = evaluate(cfg, [mk(f) for f in ck["actors"]], a.n)
    if ck.get("global_actor") is not None:
        out["global"] = evaluate(cfg, [mk(ck["global_actor"])] * env.M, a.n)
else:
    print("checkpoint has no per-agent actors (e.g. MAPPO): evaluate via run_all_experiments.")
for k, rows in out.items():
    s = summarise(rows)
    print(f"[{k}] U_total={s['U_total']:.1f} U_norm={s['U_norm']:.4f} delivered={s['delivered_mbit']:.3f} Mbit overflow={s['overflow_mbit']:.3f}")
d = new_run_dir(rd, "reeval")
atomic_write_json(d / "evaluation.json", out)
print("saved", d)
