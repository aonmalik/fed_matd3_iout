"""Safe, never-overwriting checkpoints (write to temp file, fsync via os.replace, unique names)."""
from __future__ import annotations

import os
from pathlib import Path

import torch


def save_checkpoint(dirpath: str | Path, name: str, payload: dict) -> Path:
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    k = 0
    while True:
        p = d / (f"{name}.pt" if k == 0 else f"{name}_{k}.pt")
        if not p.exists():
            break
        k += 1
    tmp = p.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, p)
    return p


def load_checkpoint(path: str | Path, device="cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=False)
