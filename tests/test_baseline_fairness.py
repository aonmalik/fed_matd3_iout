"""Final validation item 5: baseline fairness -- equivalences are verified numerically, non-equivalences are declared."""
import copy

import pytest

from fed_matd3_iout.baselines import registry
from fed_matd3_iout.validation import fairness as F


def test_equivalences_hold_for_all_eleven_methods(cfg):
    r = F.run_audit(cfg)
    assert len(r["rows"]) == 11
    assert all(r["checks"].values()), {k: v for k, v in r["checks"].items() if not v}


def test_non_equivalences_are_declared_and_quantified(cfg):
    r = F.run_audit(cfg)
    ids = {n["id"] for n in r["non_equivalent"]}
    assert {"N1", "N2", "N5", "N7"} <= ids                                  # allocation, MAPPO budget, deployed policy, model traffic
    assert r["gradient_budget_ratio_matd3_to_mappo_minibatches"] > 10          # the MAPPO update-budget gap is real and reported


def test_each_method_changes_only_its_declared_keys(cfg):
    rows = {x["method"]: x for x in F.run_audit(cfg)["rows"]}
    assert rows["s3a_fedmatd3"]["changed_keys"] == []                         # proposed method == base configuration
    assert rows["fedavg_matd3"]["changed_keys"] == ["federated.aggregation", "federated.clip_norm", "federated.max_client_weight"]
    assert set(rows["centralized_matd3"]["changed_keys"]) == {"federated.enabled"}


def test_mutation_method_that_changes_the_environment_is_detected(cfg, monkeypatch):
    bad = copy.deepcopy(registry.METHODS)
    bad["independent_td3"]["overrides"]["env.n_sensors" if "n_sensors" in cfg.env else "env.episode_steps"] = (
        cfg.env.n_sensors + 1 if "n_sensors" in cfg.env else cfg.env.episode_steps - 1)
    monkeypatch.setattr(registry, "METHODS", bad)
    monkeypatch.setattr(F, "METHODS", bad)
    r = F.run_audit(cfg)
    assert not all(r["checks"].values())
