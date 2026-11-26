"""Reproducibility manifest validation (configs/experiment_manifest.json vs the merged YAML configuration).

Every mismatch listed here is CRITICAL: a publication-scale run must not start while the frozen manifest and the
configuration the code actually uses disagree.  ``validate_manifest`` returns a list of error strings (empty == consistent).
The manifest carries ``config_hash`` (sha256[:16] of the merged default configuration, see ``utils.config.config_hash``):
any later change to any YAML value changes the hash and therefore REQUIRES a new ``protocol_version`` + changelog entry.
"""
from __future__ import annotations

from .config import config_hash

REQUIRED = ("protocol_version", "protocol_changelog", "architecture", "seeds", "methods", "scalability", "train_episodes",
            "eval_scenarios", "eval_every", "n_train_seeds", "checkpoint_every", "episode_steps", "config_hash")


def validate_manifest(m: dict, cfg, methods: dict, ablations_registry: dict, ablations_yaml: dict) -> list[str]:
    errs: list[str] = []
    miss = [k for k in REQUIRED if k not in m]
    if miss:
        return [f"manifest missing required keys: {miss}"]
    if m["architecture"].get("n_usv") != 1:
        errs.append("n_usv must be 1")
    for x in m["methods"]:
        if x not in methods:
            errs.append(f"unknown method {x}")
    if set(methods) - set(m["methods"]):
        errs.append(f"methods missing from manifest: {sorted(set(methods) - set(m['methods']))}")
    if len(set(m["methods"])) != len(m["methods"]):
        errs.append("duplicate methods in manifest")
    s = m["seeds"]
    if len({s[k] for k in ("layout", "arrivals", "fading", "comm")}) != 4:
        errs.append("stream seeds must be distinct")
    if any(x.get("n_usv", 1) != 1 for x in m["scalability"]):
        errs.append("scalability entry with n_usv != 1")
    sizes = {x["n_auvs"] for x in m["scalability"]}
    if not {1, 3, 10} <= sizes:
        errs.append("scalability must include 1, 3, 10 AUVs")
    if set(ablations_yaml) != set(ablations_registry):
        errs.append("ablations.yaml and registry disagree on names")
    else:
        for k in ablations_yaml:
            if dict(ablations_yaml[k]) != dict(ablations_registry[k]):
                errs.append(f"ablation {k} differs between yaml and registry")
    # ---- manifest vs YAML (critical)
    ex, sd = cfg["experiment"], cfg["seeds"]
    for k in ("layout", "arrivals", "fading", "comm", "eval_scenario_base"):
        if s[k] != sd[k]:
            errs.append(f"seeds.{k}: manifest {s[k]} != YAML {sd[k]}")
    if list(s["train_seeds"]) != list(range(int(ex["n_train_seeds"]))):
        errs.append(f"train_seeds {s['train_seeds']} != range(experiment.n_train_seeds={ex['n_train_seeds']})")
    for mk, ck in (("train_episodes", "train_episodes"), ("eval_scenarios", "eval_scenarios"), ("eval_every", "eval_every"),
                   ("n_train_seeds", "n_train_seeds"), ("checkpoint_every", "checkpoint_every")):
        if m[mk] != ex[ck]:
            errs.append(f"{mk}: manifest {m[mk]} != YAML experiment.{ck} {ex[ck]}")
    if m["episode_steps"] != cfg["env"]["episode_steps"]:
        errs.append(f"episode_steps: manifest {m['episode_steps']} != YAML {cfg['env']['episode_steps']}")
    if str(m["protocol_version"]) != str(ex.get("protocol_version")):
        errs.append(f"protocol_version: manifest {m['protocol_version']} != YAML {ex.get('protocol_version')}")
    log = m["protocol_changelog"]
    if not log or str(log[-1].get("version")) != str(m["protocol_version"]):
        errs.append("protocol_changelog must end with an entry for the current protocol_version")
    h = config_hash(cfg)
    if m["config_hash"] != h:
        errs.append(f"config_hash: manifest {m['config_hash']} != merged YAML {h} (a YAML value changed: bump protocol_version, "
                    "add a changelog entry, then re-freeze the hash)")
    return errs
