import numpy as np
import pytest
from fed_matd3_iout.environment.sensor_queue import SensorQueue
from fed_matd3_iout.semantic.mission_criticality import build_categories


@pytest.fixture
def q(cfg):
    return SensorQueue(0, 5000, 0.0004, 0.6, build_categories(cfg["semantic"]))


def test_poisson_arrival_mean(cfg):
    rng = np.random.default_rng(1)
    lam, dt = 8.0, 1.0
    a = rng.poisson(lam * dt, 20000)
    assert abs(a.mean() - lam * dt) < 0.1          # A ~ Poisson(lambda dt)


def test_buffer_capacity_and_overflow(q):
    rng = np.random.default_rng(0)
    q.initialise(0.0, 4900, rng)
    gen, loss = q.arrive(1.0, 300, rng)
    assert loss == pytest.approx(200.0) and q.occ == pytest.approx(5000.0)   # Q1-Q3
    assert q.residual_check() == pytest.approx(0, abs=1e-9)


def test_arrival_before_service_order(q):
    """B(t+1) = min(B_max, B+A) - D  : a full buffer loses arrivals BEFORE service frees space."""
    rng = np.random.default_rng(0)
    q.initialise(0.0, 5000, rng)
    q.arrive(1.0, 100, rng)                      # all 100 lost although service follows in the same step
    assert q.lost == pytest.approx(100.0)
    got = q.serve(50)
    assert sum(b.units for b in got) == pytest.approx(50.0) and q.occ == pytest.approx(4950.0)


def test_no_negative_and_no_double_collection(q):
    rng = np.random.default_rng(2)
    q.initialise(0.0, 1000, rng)
    first = sum(b.units for b in q.serve(600))
    second = sum(b.units for b in q.serve(600))
    third = sum(b.units for b in q.serve(600))
    assert first == pytest.approx(600) and second == pytest.approx(400) and third == 0.0
    assert q.occ >= 0.0 and q.collected == pytest.approx(1000)


def test_fifo_order(q):
    rng = np.random.default_rng(3)
    q.initialise(0.0, 100, rng)
    q.arrive(5.0, 50, rng)
    got = q.serve(120)
    ts = [b.t_gen for b in got]
    assert ts == sorted(ts)


def test_partial_batch_split_conserves_units(q):
    rng = np.random.default_rng(4)
    q.initialise(0.0, 100, rng)
    before = sum(b.units for b in q.q_batches)
    got = q.serve(33.3)
    after = sum(b.units for b in q.q_batches)
    assert before == pytest.approx(after + sum(b.units for b in got))
