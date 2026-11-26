"""D10: fixed-workload completion metric (separate from backlog_clearance_time_s)."""
import numpy as np
import pytest

from fed_matd3_iout.evaluation.fixed_workload import (FixedWorkloadTracker, evaluate_fixed_workload, fixed_workload_cfg,
                                                     rollout_fixed_workload)
from fed_matd3_iout.evaluation.metrics import BacklogClearanceTracker

T_OUT = 100.0


def _run(W, steps, lost=None, gen=None):
    tr = FixedWorkloadTracker(W)
    for k, d in enumerate(steps):
        tr.update(float(k + 1), d, (lost or {}).get(k, 0.0), (gen or {}).get(k))
    return tr.result(T_OUT)


def test_immediate_completion_at_first_step():
    r = _run(10.0, [10.0])
    assert r["fixed_workload_completion_time_s"] == 1.0 and r["fixed_workload_censored"] == 0.0 and r["fixed_workload_reason"] == "completed"


def test_normal_completion_time_is_the_step_the_last_unit_is_delivered():
    r = _run(10.0, [2.0, 5.0, 9.0, 10.0, 10.0])
    assert r["fixed_workload_completion_time_s"] == 4.0 and r["fixed_workload_censored"] == 0.0


def test_partial_completion_is_censored_at_timeout_with_delivered_fraction():
    r = _run(10.0, [2.0, 5.0, 7.5])
    assert r["fixed_workload_completion_time_s"] == T_OUT and r["fixed_workload_censored"] == 1.0
    assert r["fixed_workload_delivered_fraction"] == pytest.approx(0.75) and r["fixed_workload_reason"] == "timeout"


def test_task_failure_means_censored_failed_units_even_if_the_rest_is_delivered():
    r = _run(10.0, [2.0, 9.0, 9.0], lost={1: 1.0})
    assert r["fixed_workload_censored"] == 1.0 and r["fixed_workload_reason"] == "failed_units"
    assert r["fixed_workload_completion_time_s"] == T_OUT                       # kept (not dropped) and valued at the timeout


def test_zero_task_edge_case_completes_at_time_zero_and_is_flagged():
    r = _run(0.0, [0.0])
    assert r["fixed_workload_completion_time_s"] == 0.0 and r["fixed_workload_censored"] == 0.0 and r["fixed_workload_reason"] == "zero_workload"
    with pytest.raises(ValueError):
        FixedWorkloadTracker(float("nan"))


def test_timeout_without_any_delivery():
    r = _run(10.0, [0.0, 0.0])
    assert r["fixed_workload_censored"] == 1.0 and r["fixed_workload_delivered_fraction"] == 0.0


def test_new_generation_is_flagged_as_a_violated_workload():
    r = _run(10.0, [5.0, 10.0], gen={0: 10.0, 1: 14.0})
    assert r["fixed_workload_violated"] == 1.0


def test_transmitted_or_collected_is_not_completion_only_delivery_counts():
    """The tracker is driven by DELIVERED mbit; feeding collected/onboard amounts instead would be a caller error and is not accepted
    implicitly: below the workload nothing completes."""
    assert _run(10.0, [0.0, 0.0, 9.999])["fixed_workload_censored"] == 1.0


def test_timing_is_deterministic_under_controlled_inputs():
    a = _run(10.0, [1.0, 3.0, 6.0, 10.0])
    b = _run(10.0, [1.0, 3.0, 6.0, 10.0])
    assert a == b and a["fixed_workload_completion_time_s"] == 4.0


def test_backlog_clearance_is_unchanged_and_independent():
    b = BacklogClearanceTracker()
    b.update(5.0, 100.0, 9.0)
    assert b.result(1000.0) == (5.0, False)                                       # first-passage semantics intact


def _zero_policy(M):
    return [lambda o: np.zeros(2) for _ in range(M)]


def test_environment_workload_is_identical_across_policies_and_has_no_arrivals(smoke_cfg):
    smoke_cfg["env"]["episode_steps"] = 80
    cfg = smoke_cfg
    M = int(cfg["env"]["n_auv"]) if "n_auv" in cfg["env"] else 3
    rows = {}
    for name, pol in {"zero": _zero_policy(M), "forward": [lambda o: np.ones(2) for _ in range(M)]}.items():
        rows[name] = rollout_fixed_workload(cfg, pol, 0)
    assert rows["zero"]["fixed_workload_mbit"] == rows["forward"]["fixed_workload_mbit"] > 0       # matched workload
    assert rows["zero"]["fixed_workload_violated"] == 0.0 and rows["forward"]["fixed_workload_violated"] == 0.0
    assert fixed_workload_cfg(cfg)["traffic"]["scale"] == 0.0 and cfg["traffic"]["scale"] != 0.0     # original cfg untouched
    assert 0.0 <= rows["zero"]["fixed_workload_delivered_fraction"] <= 1.0


def test_evaluate_fixed_workload_is_deterministic(smoke_cfg):
    smoke_cfg["env"]["episode_steps"] = 60
    M = int(smoke_cfg["env"]["n_auv"]) if "n_auv" in smoke_cfg["env"] else 3
    a = evaluate_fixed_workload(smoke_cfg, _zero_policy(M), 2)
    b = evaluate_fixed_workload(smoke_cfg, _zero_policy(M), 2)
    assert a == b
