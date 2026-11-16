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
    ckpt = tr.save("final")
    glob = evaluate(cfg, tr.global_policies())
    loc = evaluate(cfg, tr.local_policies()) if tr.server is not None else glob
    comm = tr.proto.summary() if tr.proto is not None else {}
    out = {"method": method, "tag": tag or method, "seed": seed, "global": glob, "local": loc, "comm": comm,
           "curve": curve, "checkpoint": str(ckpt), "run_dir": str(run_dir), "device": str(device)}
    atomic_write_json(run_dir / "evaluation.json", out)
    log.info(f"done: U_total(global)={summarise(glob)['U_total']:.1f}")
    log.close()
    return out
