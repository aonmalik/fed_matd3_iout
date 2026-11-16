import json
import numpy as np
import pytest
import torch
from fed_matd3_iout.baselines.registry import build_cfg
from fed_matd3_iout.training.checkpoint import load_checkpoint, save_checkpoint
from fed_matd3_iout.training.trainer import Trainer
from fed_matd3_iout.utils.config import config_hash, load_config, set_by_path
from fed_matd3_iout.utils.reproducibility import new_run_dir, save_run_manifest, software_versions
from fed_matd3_iout.utils.seed import make_rng


def test_named_streams_deterministic_and_distinct():
    a, b = make_rng(1000, "layout", 3).random(5), make_rng(1000, "layout", 3).random(5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, make_rng(1000, "arrivals", 3).random(5))
    assert not np.array_equal(a, make_rng(1000, "layout", 4).random(5))
    assert not np.array_equal(a, make_rng(1001, "layout", 3).random(5))


def test_config_hash_and_unknown_override(cfg):
    h = config_hash(cfg)
    assert h == config_hash(load_config()) and isinstance(h, str)
    with pytest.raises(KeyError):
        set_by_path(cfg, "env.not_a_key", 1)                     # typos in overrides never create silent new keys
    set_by_path(cfg, "env.n_auvs", 4)
    assert config_hash(cfg) != h


def test_run_dirs_never_overwritten(tmp_path):
    d = {new_run_dir(tmp_path, "x") for _ in range(5)}
    assert len(d) == 5 and all(p.exists() for p in d)


def test_checkpoints_never_overwritten(tmp_path):
    p1 = save_checkpoint(tmp_path, "final", {"a": torch.ones(2)})
    p2 = save_checkpoint(tmp_path, "final", {"a": torch.zeros(2)})
    assert p1 != p2 and torch.equal(load_checkpoint(p1)["a"], torch.ones(2))


def test_manifest_contents(tmp_path, smoke_cfg):
    save_run_manifest(tmp_path, smoke_cfg, {"method": "x"})
    m = json.loads((tmp_path / "run_manifest.json").read_text())
    assert {"config", "config_hash", "software", "git_commit", "extra"} <= set(m) and "torch" in m["software"]
    assert m["config"]["seeds"]["layout"] == 1000 and "torch" in software_versions()


def test_training_is_reproducible(smoke_cfg):
    outs = []
    for _ in range(2):
        cfg = build_cfg(smoke_cfg, "s3a_fedmatd3")
        t = Trainer(cfg, torch.device("cpu"), None, 7)
        c = t.train(2)
        outs.append(([r["return_mean"] for r in c], t.server.theta.clone(), t.proto.summary()["bytes_total"]))
    assert outs[0][0] == outs[1][0] and torch.equal(outs[0][1], outs[1][1]) and outs[0][2] == outs[1][2]


def test_different_training_seed_differs(smoke_cfg):
    cfg = build_cfg(smoke_cfg, "s3a_fedmatd3")
    a = Trainer(cfg, torch.device("cpu"), None, 1); a.train(2)
    b = Trainer(cfg, torch.device("cpu"), None, 2); b.train(2)
    assert not torch.equal(a.server.theta, b.server.theta)
