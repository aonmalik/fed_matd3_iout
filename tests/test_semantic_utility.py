import math
import numpy as np
import pytest
from fed_matd3_iout.semantic.mission_criticality import build_categories, sigma_of, mean_semantic_value_density
from fed_matd3_iout.semantic.packet import Batch
from fed_matd3_iout.semantic.semantic_utility import UtilityLedger, delivered_utility
from fed_matd3_iout.semantic.value_of_information import voi


def mk(units=10, sigma=0.8, mu=0.05, c=2.5, cat="critical", pid=1):
    return Batch(pid, 0, 0.0, units, sigma, cat, mu, c, t_collect=1.0)


def test_importance_normalised(cfg):
    for c in build_categories(cfg["semantic"]):
        for om in cfg.semantic.hotspot_weights:
            assert 0.0 <= sigma_of(om, c) <= 1.0


def test_category_probabilities_validated(cfg):
    cfg.semantic.categories.critical["prob"] = 0.9
    with pytest.raises(ValueError):
        build_categories(cfg["semantic"])


def test_voi_decay():
    assert float(voi(0.7, 0.05, 0.0)) == pytest.approx(0.7)
    assert float(voi(0.7, 0.05, 10.0)) == pytest.approx(0.7 * math.exp(-0.5))
    ages = np.linspace(0, 100, 20)
    assert np.all(np.diff(voi(0.7, 0.05, ages)) < 0)


def test_mission_criticality_applied_once():
    b = mk(units=1, sigma=0.8, mu=0.0, c=2.5)
    assert delivered_utility(b, 0.0) == pytest.approx(2.5 * 0.8)              # c * sigma, not c^2 * sigma


def test_successful_failed_and_delayed_delivery():
    b = mk()
    assert delivered_utility(b, 5.0, True) == pytest.approx(10 * 2.5 * 0.8 * math.exp(-0.25))
    assert delivered_utility(b, 5.0, False) == 0.0
    assert delivered_utility(b, 50.0) < delivered_utility(b, 5.0)


def test_no_duplicate_utility():
    L = UtilityLedger()
    b = mk()
    L.record(b, 3.0, auv=0)
    with pytest.raises(RuntimeError):
        L.record(b, 3.0, auv=0)                                               # the very same chunk cannot count twice
    assert L.U == pytest.approx(delivered_utility(b, 3.0))


def test_ledger_aggregates():
    L = UtilityLedger()
    L.record(mk(pid=1), 2.0)
    L.record(mk(pid=2, units=5, cat="routine", c=1.0, mu=0.01, sigma=0.3), 4.0)
    assert L.units == 15 and 0 < L.voi_retention < 1 and L.mean_age == pytest.approx((10 * 2 + 5 * 4) / 15)


def test_density_aggregate_approximation(cfg):
    cats = build_categories(cfg["semantic"])
    assert mean_semantic_value_density(1.0, cats) > mean_semantic_value_density(0.3, cats)
