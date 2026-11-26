"""D9 (provisional): compute energy in JOULES or ``not_measured`` -- never a silent zero, never an invented conversion."""
import math

import pytest

from fed_matd3_iout.utils.compute_energy import (ComputeMeter, NOT_MEASURED, counter_delta_joules, integrate_power_joules)


def test_missing_instrumentation_is_not_measured_and_not_zero():
    m = ComputeMeter(counters=[])
    with m:
        sum(range(1000))
    r = m.report()
    assert r["status"] == "not_measured" and r["energy_j"] is None and r["energy_j"] != 0.0
    assert r["compute_time_wall_s"] > 0 and r["n_update_intervals"] == 1 and r["instrument"] is None
    assert "not zero" in r["note"]


def test_valid_measurement_trapezoid_in_joules():
    r = integrate_power_joules([0, 1, 2, 3], [10, 10, 20, 20], "W")
    assert r["status"] == "measured" and r["energy_j"] == pytest.approx(10 + 15 + 20) and r["covered_s"] == 3 and r["missing_s"] == 0


@pytest.mark.parametrize("unit,factor", [("W", 1.0), ("mW", 1e-3), ("kW", 1e3)])
def test_power_unit_conversion(unit, factor):
    assert integrate_power_joules([0, 2], [5, 5], unit)["energy_j"] == pytest.approx(10 * factor)


@pytest.mark.parametrize("unit,factor", [("J", 1.0), ("mJ", 1e-3), ("uJ", 1e-6), ("kJ", 1e3), ("Wh", 3600.0)])
def test_energy_counter_unit_conversion(unit, factor):
    assert counter_delta_joules(100, 600, unit)["energy_j"] == pytest.approx(500 * factor)


def test_unknown_units_raise():
    with pytest.raises(ValueError):
        integrate_power_joules([0, 1], [1, 1], "hp")
    with pytest.raises(ValueError):
        counter_delta_joules(0, 1, "cal")


def test_missing_interval_is_reported_and_downgrades_status():
    r = integrate_power_joules([0, 1, 2, 10, 11], [10, 10, 10, 10, 10], "W", max_gap_s=2.0)      # gap 2 -> 10 is missing
    assert r["status"] == "partial" and r["missing_s"] == pytest.approx(8.0) and r["covered_s"] == pytest.approx(3.0)
    assert r["energy_j"] == pytest.approx(30.0)                                                    # only covered time; not extrapolated


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, "x", None])
def test_invalid_readings_are_never_integrated(bad):
    r = integrate_power_joules([0, 1, 2], [10, bad, 10], "W")
    assert r["n_invalid"] >= 1 and r["status"] == "not_measured" and r["energy_j"] is None   # both intervals touch the bad sample
    r = integrate_power_joules([0, 1, 2, 3], [10, 10, bad, 10], "W")
    assert r["status"] == "partial" and r["energy_j"] == pytest.approx(10.0) and r["missing_s"] == pytest.approx(2.0)


def test_counter_wrap_and_invalid_counter():
    assert counter_delta_joules(900, 100, "uJ", wrap=1000)["energy_j"] == pytest.approx(200e-6)
    assert counter_delta_joules(900, 100, "uJ")["status"] == "not_measured"                  # decrease without wrap range: not guessed
    assert counter_delta_joules(float("nan"), 5, "uJ")["energy_j"] is None


def test_too_few_samples_is_not_measured():
    assert integrate_power_joules([0.0], [5.0])["energy_j"] is None and integrate_power_joules([], [])["status"] == "not_measured"


def test_meter_with_a_readable_counter_measures_joules(tmp_path):
    f = tmp_path / "energy_uj"
    f.write_text("1000")
    m = ComputeMeter(counters=[str(f)])
    with m:
        f.write_text("1501000")                                                              # +1.5 J
    r = m.report()
    assert r["status"] == "measured" and r["energy_j"] == pytest.approx(1.5) and r["instrument"]


def test_meter_with_an_unreadable_counter_falls_back_to_not_measured(tmp_path):
    m = ComputeMeter(counters=[str(tmp_path / "missing")])
    with m:
        pass
    assert m.report()["energy_j"] is None


def test_mappo_and_matd3_trainers_report_time_and_unmeasured_energy(smoke_cfg):
    import torch
    from fed_matd3_iout.baselines.mappo import MAPPOTrainer
    from fed_matd3_iout.training.trainer import Trainer
    smoke_cfg["env"]["episode_steps"] = 40
    smoke_cfg["matd3"]["ppo"]["rollout_steps"] = 20
    tm = MAPPOTrainer(smoke_cfg, torch.device("cpu"), None, 0, None)
    tm.train(1)
    r = tm.compute_report()
    assert r["n_update_intervals"] >= 1 and r["compute_time_wall_s"] > 0
    if not ComputeMeter().paths:                                                              # no instrument in this environment
        assert r["status"] == "not_measured" and r["energy_j"] is None
    t2 = Trainer(smoke_cfg, torch.device("cpu"), train_seed=0)
    t2.train(1)
    assert "compute_time_wall_s" in t2.compute_report()
