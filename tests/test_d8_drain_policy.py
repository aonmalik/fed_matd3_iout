"""D8 (proposed, provisional): bounded communication drain after the training cutoff.  Stages: cutoff -> census -> drain start ->
bounded drain -> drain end -> evaluation start.  No learning during the drain; unresolved updates are reported, never forced."""
import numpy as np
import pytest

from fed_fixtures import Harness, link
from fed_matd3_iout.federated.drain_bound import server_side_drain_bound_s
from test_final_drain import _fin, _prime


def test_config_drain_horizon_covers_the_derived_server_side_bound_and_is_marked_provisional(cfg):
    b = server_side_drain_bound_s(cfg)
    assert b == pytest.approx(400.0 + 110.0 / 1500.0 + 120.0 + float(cfg["env"]["dt_s"]))
    assert float(cfg["federated"]["final_drain_s"]) >= b                        # the configured horizon is not below the derived bound
    src = open(__import__("pathlib").Path(__file__).resolve().parents[1] / "configs" / "federated_learning.yaml").read()
    assert "PROVISIONAL" in src and "sensitivity" in src                         # honest labelling in the config itself


def test_zero_pending_updates(cfg):
    h = Harness(cfg, 2, "async"); h.hold()
    tr, t = _fin(h, 100.0)
    assert t["pending_at_cutoff"]["unresolved_total"] == 0 and t["drain_steps"] == 0
    assert t["drain_terminated_by"] == "nothing_pending_at_cutoff" and t["evaluation_started_with_unresolved"] is False


def test_one_pending_update_resolved_within_the_horizon(cfg):
    h = Harness(cfg, 2, "async", agg_buffer_size=1); h.hold(False)
    _prime(h, 0); h.env.links[0] = link(rate=h.rate_for_steps(2), prop=4.0)
    h.run_until(lambda: h.proto.n_transmitted == 1)
    tr, t = _fin(h, 200.0)
    assert t["pending_at_cutoff"]["unresolved_total"] == 1 and t["resolved_during_drain"] == 1
    assert t["unresolved_at_drain_expiry"] == 0 and t["drain_terminated_by"] == "all_resolved"
    assert t["evaluation_started_with_unresolved"] is False and 0 < t["drain_s"] < 200.0


def test_multiple_pending_updates_resolve_independently(cfg):
    h = Harness(cfg, 3, "async", agg_buffer_size=1); h.hold(False)
    for i in range(3):
        _prime(h, i, 0.1 + 0.01 * i); h.env.links[i] = link(rate=h.rate_for_steps(2), prop=3.0 + i)
    h.run_until(lambda: h.proto.n_transmitted == 3)
    tr, t = _fin(h, 300.0)
    assert t["pending_at_cutoff"]["unresolved_total"] == 3 and t["unresolved_at_drain_expiry"] == 0
    assert t["resolved_during_drain"] == 3 and t["statistics_accounting"]["balanced"]


def test_unresolved_updates_at_expiry_are_reported_not_discarded(cfg):
    h = Harness(cfg, 2, "async", agg_buffer_size=3); h.hold(False)           # buffer can never fill with one client
    _prime(h, 0); h.env.links[0] = link(rate=h.rate_for_steps(2)); h.run(8)
    tr, t = _fin(h, 10.0)
    assert t["drain_terminated_by"] == "budget_exhausted" and t["unresolved_at_drain_expiry"] == 1
    assert t["evaluation_started_with_unresolved"] is True
    assert t["statistics_accounting"]["unresolved"] == 1 and t["statistics_accounting"]["balanced"]   # still held, not lost


def test_zero_length_drain_equals_cutoff_evaluation(cfg):
    h = Harness(cfg, 2, "async", agg_buffer_size=3); h.hold(False)
    _prime(h, 0); h.env.links[0] = link(rate=h.rate_for_steps(2)); h.run(8)
    v = h.server.version
    tr, t = _fin(h, 0.0)
    assert t["drain_steps"] == 0 and t["drain_terminated_by"] == "drain_not_requested" and h.server.version == v
    s = t["stages"]
    assert s["1_training_cutoff_s"] == s["5_drain_end_s"] == s["6_evaluation_start_s"] and s["3_drain_start_s"] is None


def test_no_training_activity_during_the_drain(smoke_cfg):
    """Real trainer: gradient steps, replay contents, local steps and created updates are frozen while the drain runs."""
    import torch
    from fed_matd3_iout.training.trainer import Trainer
    smoke_cfg["env"]["episode_steps"] = 60
    tr = Trainer(smoke_cfg, torch.device("cpu"), train_seed=0)
    tr.train(2)
    snap = lambda: ([c.total_steps for c in tr.clients], [len(c.replay) if hasattr(c.replay, "__len__") else c.replay.n for c in tr.clients],
                    tr.proto.n_created, [c.update_counter for c in tr.clients])
    before = snap()
    term = tr.finalize(drain_s=30.0)
    assert snap() == before                                         # nothing was trained, stored or created
    assert term["stages"]["6_evaluation_start_s"] >= term["stages"]["1_training_cutoff_s"]


def test_evaluation_start_is_deterministic(smoke_cfg):
    import torch
    from fed_matd3_iout.training.trainer import Trainer
    smoke_cfg["env"]["episode_steps"] = 60
    out = []
    for _ in range(2):
        tr = Trainer(smoke_cfg, torch.device("cpu"), train_seed=0)
        tr.train(2)
        t = tr.finalize(drain_s=30.0)
        out.append((t["stages"], t["resolved_during_drain"], t["unresolved_at_drain_expiry"], t["drain_terminated_by"]))
    assert out[0] == out[1]
