"""Protocol 2.2: the former `completion_time_s` is a first-passage BACKLOG-CLEARANCE time, not a fixed-workload completion time."""
import numpy as np
import pytest

from fed_matd3_iout.evaluation.evaluator import rollout
from fed_matd3_iout.evaluation.metrics import BACKLOG_CLEARANCE_FRACTION, BacklogClearanceTracker

T = 100.0


def test_threshold_is_ten_percent_of_generated_so_far():
    assert BACKLOG_CLEARANCE_FRACTION == 0.10
    k = BacklogClearanceTracker()
    k.update(1.0, 100.0, 10.0)                                   # exactly 10%: cleared (<=)
    assert k.result(T) == (1.0, False)


def test_continuing_generation_moves_the_denominator():
    k = BacklogClearanceTracker()
    k.update(1.0, 100.0, 50.0)                                   # 50% -> not cleared
    k.update(2.0, 600.0, 50.0)                                   # generation continued: 50/600 < 10% -> cleared now
    assert k.result(T) == (2.0, False)


def test_zero_generated_workload_is_censored_not_cleared():
    k = BacklogClearanceTracker()
    for t in range(1, 50):
        k.update(float(t), 0.0, 0.0)
    assert k.result(T) == (T, True)


def test_clearance_followed_by_new_generation_stays_cleared_at_first_passage():
    k = BacklogClearanceTracker()
    k.update(5.0, 100.0, 5.0)
    k.update(6.0, 200.0, 190.0)                                  # a large new backlog appears afterwards
    assert k.result(T) == (5.0, False)


def test_never_cleared_is_censored_at_episode_length():
    k = BacklogClearanceTracker()
    for t in range(1, 100):
        k.update(float(t), 100.0 + t, 90.0)
    assert k.result(T) == (T, True)


def test_rollout_reports_new_name_and_identical_legacy_alias(smoke_cfg):
    smoke_cfg["env"]["episode_steps"] = 20
    pol = [lambda o: np.zeros(2) for _ in range(smoke_cfg["env"]["n_auvs"])]
    m = rollout(smoke_cfg, pol, 0)
    assert m["completion_time_s"] == m["backlog_clearance_time_s"]
    assert m["completion_censored"] == m["backlog_clearance_censored"] in (0.0, 1.0)
    assert 0 < m["backlog_clearance_time_s"] <= 20 * smoke_cfg["env"]["dt_s"] + 1e-9


def test_censored_value_equals_episode_length(smoke_cfg):
    smoke_cfg["env"]["episode_steps"] = 5
    pol = [lambda o: np.zeros(2) for _ in range(smoke_cfg["env"]["n_auvs"])]
    m = rollout(smoke_cfg, pol, 0)
    if m["backlog_clearance_censored"] == 1.0:
        assert m["backlog_clearance_time_s"] == pytest.approx(5 * smoke_cfg["env"]["dt_s"])
