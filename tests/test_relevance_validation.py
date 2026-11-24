"""Final validation item 3: mathematical validation of the batch ledger, weight stability and maturity sensitivity."""
import math

import pytest

from fed_matd3_iout.semantic.attribution import BatchBook
from fed_matd3_iout.validation import relevance_checks as R


def test_auv1_batch_is_credited_to_auv1_not_to_the_auv_active_at_delivery():
    ex = R.auv1_example()
    assert ex["AUV1_credited_with_B1_only"] and ex["AUV0_received_none_of_B1"]
    b1 = ex["batch_B1"]
    expect = 10 * 2.0 * 0.8 * math.exp(-b1["mu"] * (250.0 - 95.0))      # decay measured from generation (t_collect - 5 s)
    assert ex["claim_AUV1"]["U_mature"] == pytest.approx(expect) and ex["rho_sem_AUV1"] == pytest.approx(math.exp(-1.55))
    assert ex["claim_AUV0"]["U_mature"] == pytest.approx(ex["batch_B0"]["U_delivered"])
    assert set(ex["legacy_window_ratio_AUV1"].values()) == {0.0}         # the old windowed ratio saw 0 / undefined for AUV 1


def test_real_environment_mass_balance_utility_units_per_auv(cfg):
    r = R.real_env_mass_balance(cfg, scenarios=(0,))
    s = r["scenarios"][0]
    assert r["passed"] and s["U_abs_err"] < 1e-9 and s["per_auv_abs_err"] < 1e-9
    assert s["delivered_units_env"] == pytest.approx(s["delivered_units_book"]) and s["delivered_units_env"] <= s["collected_units_env"]


def test_mass_balance_with_late_delivery_and_censoring():
    bk = BatchBook(1, 0.05)
    b = R._b(1, 0.0, 10, 0.5, 2.0)
    bk.on_collect(0, b)
    h, rest = b.split(4.0)
    bk.on_deliver(0, h, R._u(h, 50.0))
    bk.claim(0, 400.0)                                                   # claimed by age with 6 units still undelivered
    bk.on_deliver(0, rest, R._u(rest, 450.0))                            # late
    assert bk.utility_accounted(0) == pytest.approx(R._u(h, 50.0) + R._u(rest, 450.0))
    assert bk.late_units[0] == pytest.approx(6.0)


def test_weight_stability(cfg):
    w = R.weight_stability(cfg.federated)
    assert w["weights_valid_everywhere"] and w["bounded"]
    assert w["empirical_max_abs_dw_dM_nocap"] <= w["analytic_bound_abs_dw_dM_nocap"] + 1e-9
    assert w["max_relative_weight_change_from_relevance_alone"] == pytest.approx(cfg.federated.eps_m ** (-cfg.federated.eta))


def test_maturity_sensitivity_is_descriptive_and_stable(smoke_cfg):
    r = R.maturity_sensitivity(smoke_cfg, eps_list=(0.02, 0.05, 0.2), scenarios=(0,), claim_every_steps=5)
    rows = {x["eps"]: x for x in r["rows"]}
    assert set(rows) == {0.02, 0.05, 0.2} and r["reference_eps"] == 0.05
    for x in r["rows"]:
        assert 0.0 <= x["pooled_rho_sem"] <= 1.0 or math.isnan(x["pooled_rho_sem"])    # bounded for every threshold
    # larger eps => shorter horizon => at least as much value judged mature (monotone), never more value than was collected
    assert rows[0.2]["mature_value_V"] >= rows[0.05]["mature_value_V"] >= rows[0.02]["mature_value_V"]
