"""PILOT (Protocol 2.2 candidate): all 11 methods x 2 training seeds x a few episodes.  NOT publication evidence.

SAFE BY DEFAULT: without --approved this only PRINTS the plan and the estimated cost and exits 0.  Nothing is trained.
    python scripts/run_pilot_2seeds.py                      # dry run (plan + cost estimate)
    python scripts/run_pilot_2seeds.py --approved           # actually runs (only after the author's explicit approval)

Purpose (plumbing / cost / dynamics, never performance): every method trains and evaluates end to end on the CURRENT Protocol 2.2
candidate configuration (configs/*.yaml + configs/experiment_manifest.json; Protocol 2.1 stays frozen in configs/protocols/v2.1 and is NOT what runs here;
2.2 results are not comparable with 2.1 results, and this is a single-USV project, not a reproduction of the Paper 2 setup); communication, retry/back-off, rejected-bytes and aggregation diagnostics are recorded; cost per method is measured.
Reduced on purpose and recorded in the output: train episodes (--episodes, default 5) and evaluation scenarios (--eval-scenarios, default 5).
With 2 seeds NO significance test is possible (exact Wilcoxon minimum p = 0.5; a paired t-test on 2 pairs has 1 degree of freedom):
none is computed or reported.  Outputs go to results/pilot_protocol22/ and are stamped PILOT (the script refuses a Protocol 2.1-named output directory for a 2.2 run).
"""
from _common import *
import argparse, json, time
from fed_matd3_iout.baselines.registry import ABLATIONS, METHODS
from fed_matd3_iout.evaluation.evaluator import summarise
from fed_matd3_iout.experiment import run_one
from fed_matd3_iout.utils.config import config_hash
from fed_matd3_iout.utils.manifest import validate_manifest
from fed_matd3_iout.utils.reproducibility import atomic_write_json

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--episodes", type=int, default=5)
p.add_argument("--eval-scenarios", type=int, default=5)
p.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
p.add_argument("--methods", nargs="*", default=list(METHODS))
p.add_argument("--out", default=str(ROOT / "results" / "pilot_protocol22"))
p.add_argument("--approved", action="store_true", help="REQUIRED to train; set only after the author has approved the pilot")
a = p.parse_args()

cfg = load_config(overrides={"experiment.eval_scenarios": a.eval_scenarios})
import yaml
man = json.load(open(ROOT / "configs" / "experiment_manifest.json"))
errs = validate_manifest(man, load_config(), METHODS, ABLATIONS, yaml.safe_load(open(ROOT / "configs" / "ablations.yaml"))["ablations"])
import re
version = str(man["protocol_version"])
if version.startswith("2.2") and re.search(r"(protocol[_-]?2[._-]?1|v2[._]1)", Path(a.out).name.lower()):
    print(f"REFUSED: output directory {a.out!r} is Protocol 2.1-named but the selected protocol is {version}; use a protocol-2.2 name.")
    sys.exit(2)
run_hash = config_hash(cfg)
runs = len(a.methods) * len(a.seeds)
est_s = runs * (a.episodes * 12.0 + a.eval_scenarios * 2.0 * 1.5)        # 12 s/episode MEASURED for s3a_fedmatd3; eval ~1.2-1.5 s/scenario
print(f"PILOT plan: protocol {man['protocol_version']} (manifest config_hash {man['config_hash']}), {len(a.methods)} methods x seeds {a.seeds}, "
      f"{a.episodes} episodes (protocol: {cfg.experiment.train_episodes}), {a.eval_scenarios} eval scenarios (protocol: 20)")
print(f"  runs: {runs};  estimated wall time ~{est_s/60:.0f} min sequential on CPU (PREDICTION from a 12 s/episode measurement for one method; "
      f"other methods not yet timed)")
print(f"  selected protocol: {version}   manifest config_hash: {man['config_hash']}   effective run config_hash: {run_hash} "
      f"(differs from the manifest hash only by the pilot override experiment.eval_scenarios={a.eval_scenarios})")
print(f"  output directory: {a.out}   drain: federated.final_drain_s={cfg.federated.final_drain_s} s (provisional, D8)   federated mode: {cfg.federated.mode}")
print(f"  outputs: {a.out}   manifest validation: {'OK' if not errs else 'INVALID: ' + '; '.join(errs)}")
if errs:
    sys.exit(1)
if not a.approved:
    print("DRY RUN: nothing was trained. Re-run with --approved after the author's approval.")
    sys.exit(0)

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rows = []
for m in a.methods:
    for s in a.seeds:
        t0 = time.time()
        r = run_one(cfg, m, s, out / "runs", n_episodes=a.episodes, verbose=False)
        g, l = summarise(r["global"]), summarise(r["local"])
        c = r["comm"]
        rows.append({"method": m, "seed": s, "wall_s": round(time.time() - t0, 1), "headline_policy": "global" if c else "own/shared actors",
                     "U_total_global": g["U_total"], "U_total_local": l["U_total"], "delivered_mbit_global": g["delivered_mbit"],
                     "comm": {k: c.get(k) for k in ("attempted_updates", "aggregated_updates", "rejected_updates", "expired_updates", "retry_attempts",
                                                    "retry_deferred_updates", "rejected_bytes", "expired_bytes", "bytes_total", "aggregation_rounds")} if c else None,
                     "single_update_round_fraction": (r["aggregation_diagnostics"] or {}).get("single_update_round_fraction"),
                     "relevance_verdict": ((r["aggregation_diagnostics"] or {}).get("verdict") or {}).get("reading")})
        print(m, "seed", s, f"{rows[-1]['wall_s']}s", flush=True)
atomic_write_json(out / "pilot_summary.json", {
    "STAMP": "PILOT - plumbing/cost/dynamics only; NOT publication evidence; NO significance test is valid with 2 seeds",
    "protocol_version": version, "manifest_config_hash": man["config_hash"], "config_hash": run_hash,
    "pilot_overrides": {"experiment.eval_scenarios": a.eval_scenarios, "episodes": a.episodes, "seeds": a.seeds},
    "decisions_status": "D1-D6 open; D7-D10 provisional (docs/PROTOCOL_2.2_DECISION_LOG.md); compute energy not_measured", "episodes": a.episodes, "eval_scenarios": a.eval_scenarios,
    "note": "federated methods: headline = ONE global actor; non-federated: own actors (non-equivalence N5). Not like-for-like across those classes.",
    "rows": rows})
print("pilot summary:", out / "pilot_summary.json")
