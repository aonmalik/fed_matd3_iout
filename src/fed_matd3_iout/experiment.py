"""One experiment run: train one (method, training seed), evaluate global & local policies on matched scenarios."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .baselines.registry import build_cfg, make_trainer
from .evaluation.evaluator import evaluate, summarise
from .utils.logger import RunLogger
from .utils.reproducibility import atomic_write_json, new_run_dir, save_run_manifest, select_device


def run_one(base_cfg, method: str, seed: int, root: str | Path, overrides: dict | None = None,
            tag: str | None = None, verbose: bool = True, n_episodes: int | None = None) -> dict:
    cfg = build_cfg(base_cfg, method, overrides)
    device = select_device(cfg["run"]["device"])
    run_dir = new_run_dir(root, f"{tag or method}_s{seed}")
    (run_dir / "checkpoints").mkdir()
    save_run_manifest(run_dir, cfg, {"method": method, "tag": tag or method, "train_seed": seed, "device": str(device)})
    log = RunLogger(run_dir, verbose=verbose)
    log.info(f"run {run_dir.name} device={device}")
    tr = make_trainer(cfg, device, method, run_dir, seed, log)
    ee = int(cfg["experiment"]["eval_every"])
    curve = tr.train(n_episodes, eval_fn=(lambda t: summarise(evaluate(cfg, t.global_policies(), 3))) if ee else None,
                     eval_every=ee, ckpt_every=int(cfg["experiment"]["checkpoint_every"]))
    termination = tr.finalize() if hasattr(tr, "finalize") else {"federated": False}      # cutoff / drain / evaluation start
    ckpt = tr.save("final")
    glob = evaluate(cfg, tr.global_policies())
    fw_rows = None
    if bool(cfg["experiment"].get("fixed_workload_eval", False)):       # D10: separate fixed-workload measurement (global policies)
        from .evaluation.fixed_workload import evaluate_fixed_workload
        fw_rows = evaluate_fixed_workload(cfg, tr.global_policies())
        for row, fw in zip(glob, fw_rows):                              # numeric keys join the per-scenario rows (seed-level aggregation)
            row.update({k: v for k, v in fw.items() if isinstance(v, (int, float))})
    loc = evaluate(cfg, tr.local_policies()) if tr.server is not None else glob
    comm = tr.proto.summary() if tr.proto is not None else {}
    diag = None
    if tr.server is not None and tr.server.history:              # descriptive aggregation diagnostics (no effect on learning)
        from .validation.aggregation_diagnostics import diagnose_rounds
        diag = diagnose_rounds(tr.server.history, cfg["federated"])
    envd = getattr(getattr(tr, "env", None), "episode_boundary_diagnostics", None)
    out = {"method": method, "tag": tag or method, "seed": seed, "global": glob, "local": loc, "comm": comm, "aggregation_diagnostics": diag, "episode_boundary_diagnostics": (envd() if envd else None), "termination": termination, "fixed_workload": fw_rows, "training_compute": (tr.compute_report() if hasattr(tr, "compute_report") else None),
           "curve": curve, "checkpoint": str(ckpt), "run_dir": str(run_dir), "device": str(device)}
    atomic_write_json(run_dir / "evaluation.json", out)
    log.info(f"done: U_total(global)={summarise(glob)['U_total']:.1f}")
    log.close()
    return out
