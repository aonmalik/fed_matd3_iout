import numpy as np
import pytest
from scipy.stats import poisson
from fed_matd3_iout.allocation.service_arrival_predictor import (certainty_equivalent_backlog, expected_backlog_exact,
                                                                  travel_time_s)
from fed_matd3_iout.allocation.service_capacity import capacity_constant, capacity_integral, collectable_units


def brute(b, m, bmax, n=400000):
    k = np.arange(n); p = poisson.pmf(k, m)
    return (np.minimum(bmax, b + k) * p).sum(), (np.maximum(b + k - bmax, 0) * p).sum()


@pytest.mark.parametrize("b,m,bmax", [(100, 100, 500), (4900, 3600, 5000), (10, 60, 5000), (480, 50, 500), (0, 5, 20)])
def test_exact_matches_brute_force(b, m, bmax):
    eb, el = expected_backlog_exact(b, m, 1.0, bmax)
    rb, rl = brute(b, m, bmax)
    assert float(eb) == pytest.approx(rb, rel=1e-8, abs=1e-6)
    assert float(el) == pytest.approx(rl, rel=1e-6, abs=1e-6)


def test_overflow_conservation_identity():
    eb, el = expected_backlog_exact(300, 80, 1.0, 400)
    assert float(el) == pytest.approx(300 + 80 - float(eb), abs=1e-9)


def test_zero_arrival_and_zero_travel():
    eb, el = expected_backlog_exact(123.0, 0.0, 10.0, 500)
    assert float(eb) == 123.0 and float(el) == 0.0
    assert float(travel_time_s(0.0, 1.7)) == 0.0
    eb, el = expected_backlog_exact(123.0, 9.0, 0.0, 500)
    assert float(eb) == 123.0 and float(el) == 0.0


def test_buffer_already_full():
    eb, el = expected_backlog_exact(500.0, 30.0, 1.0, 500)
    assert float(eb) == pytest.approx(500.0) and float(el) == pytest.approx(30.0)


def test_certainty_equivalent_is_upper_bound_not_exact():
    for b, m in [(480, 50), (100, 100), (495, 5)]:
        eb, _ = expected_backlog_exact(b, m, 1.0, 500)
        ce = float(certainty_equivalent_backlog(b, m, 1.0, 500))
        assert ce >= float(eb) - 1e-9                           # Jensen: min(.,.) concave
    # borderline cap (b + lambda T = B_max): CE says 500 but the exact expectation is strictly lower (E[(20-N)^+] ~ 2.5)
    gap = float(certainty_equivalent_backlog(480, 20, 1.0, 500)) - float(expected_backlog_exact(480, 20, 1.0, 500)[0])
    assert gap > 1.0


def test_truncation_accuracy_bound():
    # force the tail truncation branch (buffer room far larger than the Poisson support) and check error <= tol*B_max
    b, m, bmax = 0.0, 50.0, 10 ** 7
    for tol in (1e-3, 1e-6, 1e-9):
        eb, _ = expected_backlog_exact(b, m, 1.0, bmax, tol)
        assert abs(float(eb) - 50.0) <= tol * bmax + 1e-6
    eb, _ = expected_backlog_exact(b, m, 1.0, bmax, 1e-12)
    assert float(eb) == pytest.approx(50.0, abs=1e-5)


def test_service_capacity_constant_and_integral():
    assert float(capacity_constant(0.05, 60)) == pytest.approx(3.0)
    assert capacity_integral(lambda t: 0.05 + 0 * t, 0.0, 60.0) == pytest.approx(3.0)
    assert capacity_integral(lambda t: 0.01 * t, 0.0, 10.0) == pytest.approx(0.5, rel=1e-6)     # integral of linear rate
    assert float(collectable_units(100, 0.04, 0.0004)) == pytest.approx(100.0)                  # min(B_arr, C/q)
    assert float(collectable_units(1000, 0.04, 0.0004)) == pytest.approx(100.0)
    assert float(collectable_units(1000, 0.04, 0.0004, onboard_free_mbit=0.02)) == pytest.approx(50.0)
