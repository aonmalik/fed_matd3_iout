"""D8 -- Drain Semantics and Continued Mission Dynamics.  The documented behaviour is: a drain step is a FULL env.step (sensor arrivals,
allocation, AUV motion, service, sensor uplink, link process, energy continue) while learning is frozen (no gradient step, no replay
write, no new federated update); communication keeps progressing; drain energy is reported separately and is NOT in episode or evaluation
energy; evaluation starts after the drain on freshly reset scenarios.  These tests make the real implementation match that text.
A REAL Trainer on the smoke configuration is used; ``drain_active`` is scripted True for a fixed number of steps so the drain runs
deterministically (whether a pending update exists at cutoff is a property of the run, not of the drain mechanics)."""
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from fed_matd3_iout.evaluation.evaluator import evaluate
from fed_matd3_iout.training import trainer as trainer_mod
from fed_matd3_iout.training.trainer import DRAIN_SEMANTICS, Trainer

ROOT = Path(__file__).resolve().parents[1]
K = 5


def _trained(cfg, drain_s=1000.0):
    cfg["federated"]["final_drain_s"] = drain_s
    tr = Trainer(cfg, torch.device("cpu"), None, 0)
    tr.train(1)
    return tr


def _script_drain(tr, k=K):
    n = {"left": k}
    real = tr.proto.step

    def drain_active():
        return n["left"] > 0

    def step(env, t):
        n["left"] -= 1
        return real(env, t)
    tr.proto.drain_active = drain_active
    tr.proto.step = step
    return n


def test_no_gradient_no_replay_write_no_new_update_during_drain(smoke_cfg, monkeypatch):
    tr = _trained(smoke_cfg)
    _script_drain(tr)
    calls = {"opt": 0, "local_update": 0, "store_add": 0, "make_update": 0, "accept_new_true_in_step": 0}
    real_opt = torch.optim.Optimizer.step
    monkeypatch.setattr(torch.optim.Optimizer, "step", lambda self, *a, **k: (calls.__setitem__("opt", calls["opt"] + 1), real_opt(self, *a, **k))[1])
    monkeypatch.setattr(trainer_mod, "local_update", lambda *a, **k: calls.__setitem__("local_update", calls["local_update"] + 1))
    real_add = tr.store.add
    monkeypatch.setattr(tr.store, "add", lambda *a, **k: (calls.__setitem__("store_add", calls["store_add"] + 1), real_add(*a, **k))[1])
    for c in tr.clients:
        real_mu = c.make_update
        monkeypatch.setattr(c, "make_update", lambda *a, _r=real_mu, **k: (calls.__setitem__("make_update", calls["make_update"] + 1), _r(*a, **k))[1])
    inner = tr.proto.step

    def spy(env, t):
        calls["accept_new_true_in_step"] += int(tr.proto.accept_new)
        return inner(env, t)
    tr.proto.step = spy
    t = tr.finalize()
    assert t["drain_steps"] == K
    assert calls == {"opt": 0, "local_update": 0, "store_add": 0, "make_update": 0, "accept_new_true_in_step": 0}
    assert tr.proto.accept_new is True                                         # restored afterwards


def test_communication_is_stepped_every_drain_step(smoke_cfg):
    tr = _trained(smoke_cfg)
    n = _script_drain(tr)
    t0 = tr.global_time
    t = tr.finalize()
    assert n["left"] == 0 and t["drain_steps"] == K                            # proto.step ran once per drain step
    assert tr.global_time == pytest.approx(t0 + K * tr.env.dt)


def test_mission_dynamics_continue_and_documented_semantics_match(smoke_cfg):
    tr = _trained(smoke_cfg)
    s0, g0 = tr.env.step_idx, tr.env.accounting()["G"]
    _script_drain(tr)
    t = tr.finalize()
    d = t["drain_mission"]
    assert d["available"] and d["step_idx"] == K and tr.env.step_idx == s0 + K  # the environment really stepped (beyond the episode end)
    assert d["generated_mbit"] >= 0.0 and tr.env.accounting()["G"] >= g0        # sensor arrivals continue (never decrease)
    assert d["E_prop_j"] > 0.0                                                  # propulsion/baseline power is spent
    assert d["E_comp_j"] == 0.0                                                 # no computation energy: no local update
    sem = t["drain_semantics"]
    assert sem == DRAIN_SEMANTICS
    assert sem["mission_state_evolution"]["sensor_arrivals"] and sem["mission_state_evolution"]["sensor_data_uplink"]
    assert not any(sem["training_and_learning"].values())
    assert sem["communication_progress"]["transfers_progress"] and not sem["communication_progress"]["new_federated_updates"]


def test_drain_energy_is_reported_separately_and_not_in_episode_or_evaluation_energy(smoke_cfg):
    tr = _trained(smoke_cfg)
    logged = dict(tr.curve[-1])
    _script_drain(tr)
    t = tr.finalize()
    d = t["drain_mission"]
    assert d["included_in_reported_training_energy"] is False and d["included_in_evaluation_energy"] is False
    assert tr.curve[-1] == logged                                              # the logged episode record is untouched by the drain
    assert t["drain_semantics"]["energy"]["drain_energy_recorded_separately"] is True


def test_evaluation_is_independent_of_the_drain_and_starts_after_it(smoke_cfg):
    cfg = smoke_cfg
    a = _trained(cfg)
    ref = evaluate(cfg, a.global_policies(), 1)
    _script_drain(a)
    t = a.finalize()
    after = evaluate(cfg, a.global_policies(), 1)
    assert [r["delivered_mbit"] for r in ref] == [r["delivered_mbit"] for r in after] if isinstance(ref, list) else ref == after
    st = t["stages"]
    assert st["6_evaluation_start_s"] == st["5_drain_end_s"] == a.global_time and st["3_drain_start_s"] == st["1_training_cutoff_s"]
    src = (ROOT / "src" / "fed_matd3_iout" / "experiment.py").read_text()
    assert src.index("tr.finalize()") < src.index("evaluate(cfg, tr.global_policies())")   # finalize (drain) strictly before evaluation


def test_statistics_stay_balanced_after_a_drain_and_unresolved_updates_are_reported(smoke_cfg):
    tr = _trained(smoke_cfg)
    _script_drain(tr)
    t = tr.finalize()
    assert t["statistics_accounting"]["balanced"]
    assert t["unresolved_at_drain_expiry"] == t["pending_at_evaluation_start"]["unresolved_total"]
    assert t["evaluation_started_with_unresolved"] == bool(t["unresolved_at_drain_expiry"] > 0)


def test_documentation_states_the_implemented_behaviour():
    txt = (ROOT / "docs" / "PROTOCOL_2.2_DECISION_LOG.md").read_text()
    assert "D8 — Drain Semantics and Continued Mission Dynamics" in txt
    for k in ("sensor arrivals", "AUV movement", "task allocation", "sensor-data uplink", "gradient", "evaluation starts"):
        assert re.search(k, txt, re.I), k
