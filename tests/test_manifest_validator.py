"""Reproducibility manifest: the validator must FAIL on any critical mismatch (final-validation item 1)."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from fed_matd3_iout.baselines.registry import ABLATIONS, METHODS
from fed_matd3_iout.utils.config import load_config
from fed_matd3_iout.utils.manifest import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
M = json.load(open(ROOT / "configs" / "experiment_manifest.json"))
Y = yaml.safe_load(open(ROOT / "configs" / "ablations.yaml"))["ablations"]


def errs(m, cfg):
    return validate_manifest(m, cfg, METHODS, ABLATIONS, Y)


def test_committed_manifest_is_consistent(cfg):
    assert errs(M, cfg) == []
    assert M["train_episodes"] == cfg.experiment.train_episodes == 500        # Paper 2 Table I


def test_script_exit_code_zero_when_consistent():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_manifest.py")], capture_output=True, text=True)
    assert r.returncode == 0 and "MANIFEST OK" in r.stdout


@pytest.mark.parametrize("key,val,needle", [
    ("train_episodes", 200, "train_episodes"), ("eval_scenarios", 5, "eval_scenarios"), ("eval_every", 7, "eval_every"),
    ("n_train_seeds", 3, "n_train_seeds"), ("episode_steps", 10, "episode_steps"), ("protocol_version", "9.9", "protocol_version"),
    ("config_hash", "0" * 16, "config_hash")])
def test_scalar_mismatch_is_reported(cfg, key, val, needle):
    m = copy.deepcopy(M); m[key] = val
    assert any(needle in e for e in errs(m, cfg))


def test_seed_and_structure_mismatch_reported(cfg):
    m = copy.deepcopy(M); m["seeds"]["fading"] = 1
    assert any("seeds.fading" in e for e in errs(m, cfg))
    m = copy.deepcopy(M); m["architecture"]["n_usv"] = 2
    assert any("n_usv" in e for e in errs(m, cfg))
    m = copy.deepcopy(M); del m["config_hash"]
    assert any("missing required" in e for e in errs(m, cfg))
    m = copy.deepcopy(M); m["protocol_changelog"] = m["protocol_changelog"][:-1]
    assert any("changelog" in e for e in errs(m, cfg))


def test_any_yaml_change_forces_a_new_protocol_version(cfg):
    c2 = copy.deepcopy(cfg); c2.matd3.gamma = 0.5                             # a silent hyper-parameter change ...
    assert any("config_hash" in e for e in errs(M, c2))                        # ... is caught


def test_script_exit_nonzero_on_mismatch(tmp_path):
    cfgdir = tmp_path / "configs"; cfgdir.mkdir()
    for f in (ROOT / "configs").iterdir():
        if f.is_file():
            (cfgdir / f.name).write_text(f.read_text())
    d = yaml.safe_load((cfgdir / "default.yaml").read_text()); d["experiment"]["train_episodes"] = 200
    (cfgdir / "default.yaml").write_text(yaml.safe_dump(d))
    import os
    env = dict(os.environ, FED_MATD3_CONFIG_DIR=str(cfgdir))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_manifest.py")], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "MANIFEST INVALID" in r.stdout and "train_episodes" in r.stdout


def test_protocol_20_is_preserved_and_still_reproducible():
    """Protocol 2.0 (5 seeds) stays reproducible from its frozen copy; the current manifest is protocol 2.1 (10 seeds)."""
    from fed_matd3_iout.utils.config import config_hash, load_config
    frozen = ROOT / "configs" / "protocols" / "v2.0"
    m20 = json.load(open(frozen / "experiment_manifest.json"))
    c20 = load_config(cfg_dir=frozen)
    assert m20["protocol_version"] == "2.0" and m20["n_train_seeds"] == 5 and m20["config_hash"] == config_hash(c20) == "7f4a0e4f66569528"
    assert validate_manifest(m20, c20, METHODS, ABLATIONS, yaml.safe_load(open(frozen / "ablations.yaml"))["ablations"]) == []
    assert M["protocol_version"] == "2.1" and M["n_train_seeds"] == 10 and M["seeds"]["train_seeds"] == list(range(10))
    assert [e["version"] for e in M["protocol_changelog"]][-3:] == ["1.0", "2.0", "2.1"]
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_manifest.py"), "--protocol", "2.0"], capture_output=True, text=True)
    assert r.returncode == 0 and "protocol 2.0" in r.stdout
    assert c20.federated.get("retry") is None and load_config().federated.retry.enabled      # 2.0 behaviour == retry absent
