"""Software versions, git commit, never-overwrite run directories, atomic JSON writes."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .config import to_plain, config_hash


def software_versions() -> dict:
    import numpy, scipy, torch, yaml
    v = {"python": sys.version.split()[0], "platform": platform.platform(),
         "numpy": numpy.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
         "pyyaml": yaml.__version__, "cuda_available": bool(torch.cuda.is_available())}
    try:
        import gymnasium
        v["gymnasium"] = gymnasium.__version__
    except Exception:  # pragma: no cover
        pass
    return v


def git_commit(path: str | Path | None = None) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(path or Path.cwd()),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def atomic_write_json(path: str | Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_default)
    os.replace(tmp, path)


def _default(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # pragma: no cover
        pass
    return str(o)


def new_run_dir(root: str | Path, name: str) -> Path:
    """Create a fresh run directory. Never reuses or overwrites an existing one."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    k = 0
    while True:
        d = root / f"{name}_{stamp}" if k == 0 else root / f"{name}_{stamp}_{k}"
        try:
            d.mkdir(parents=False, exist_ok=False)
            return d
        except FileExistsError:
            k += 1


def save_run_manifest(run_dir: str | Path, cfg, extra: dict | None = None) -> dict:
    man = {"config": to_plain(cfg), "config_hash": config_hash(cfg),
           "software": software_versions(), "git_commit": git_commit(),
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "extra": extra or {}}
    atomic_write_json(Path(run_dir) / "run_manifest.json", man)
    return man


def select_device(pref: str = "auto"):
    import torch
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")
