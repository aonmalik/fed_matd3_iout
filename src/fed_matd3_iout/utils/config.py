"""YAML configuration loading with deep-merge, dotted overrides and a dot-access dict."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULT_FILES = (
    "default.yaml", "environment.yaml", "acoustic_channel.yaml", "semantic_model.yaml",
    "task_allocation.yaml", "matd3.yaml", "federated_learning.yaml", "reward.yaml",
)


class AttrDict(dict):
    """dict with attribute access (read) for nested configuration."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:  # pragma: no cover
            raise AttributeError(k) from e
        return v

    def __setattr__(self, k, v):
        self[k] = v


def _wrap(x):
    if isinstance(x, dict):
        return AttrDict({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def to_plain(x):
    if isinstance(x, dict):
        return {k: to_plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_plain(v) for v in x]
    return x


def deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def set_by_path(cfg: dict, path: str, value: Any) -> None:
    keys = path.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            raise KeyError(f"override path '{path}': '{k}' is not a config section")
        d = d[k]
    if keys[-1] not in d:
        raise KeyError(f"override path '{path}': unknown key '{keys[-1]}' (refusing to create silently)")
    d[keys[-1]] = value


def config_dir() -> Path:
    env = os.environ.get("FED_MATD3_CONFIG_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "configs"


def load_config(extra_files: Iterable[str] = (), overrides: dict | None = None,
                cfg_dir: str | Path | None = None) -> AttrDict:
    """Load all default config files, then extra files (e.g. smoke.yaml), then dotted overrides."""
    d = Path(cfg_dir) if cfg_dir else config_dir()
    cfg: dict = {}
    for f in list(DEFAULT_FILES) + list(extra_files):
        p = Path(f) if os.path.isabs(str(f)) else d / f
        with open(p, "r") as fh:
            deep_update(cfg, yaml.safe_load(fh) or {})
    for k, v in (overrides or {}).items():
        set_by_path(cfg, k, v)
    return _wrap(cfg)


def config_hash(cfg: dict) -> str:
    s = json.dumps(to_plain(cfg), sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]
