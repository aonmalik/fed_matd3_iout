"""Protocol 2.2: relevance distribution diagnostics (descriptive only; parameters are never tuned)."""
import numpy as np
import pytest

from fed_matd3_iout.validation.aggregation_diagnostics import diagnose_rounds, heterogeneity_sensitivity, relevance_distributions

F = dict(eps_m=0.1, eta=1.0, alpha=1.0, kappa=0.3)


def rec(clients, M, sem=None, svc=None, w=None, w0=None):
    k = len(clients)
    return {"clients": clients, "updates": [1] * k, "relevance": M, "rho_sem": sem if sem is not None else [None] * k,
            "rho_svc": svc if svc is not None else [None] * k, "n": [1.0] * k, "stale_factor": [1.0] * k,
            "w": w or [1.0 / k] * k, "w_norel": w0 or [1.0 / k] * k, "staleness": [0] * k, "norms": [1.0] * k,
            "cf_rel_tv": 0.0, "cf_rel_effect": 0.0, "clip_scale": [1.0] * k}


def test_percent_undefined_and_distributions():
    h = [rec([0, 1], [0.0, 0.5], sem=[None, 0.5], svc=[None, None]), rec([0, 1], [0.4, 0.6], sem=[0.4, 0.6], svc=[0.4, 0.6])]
    d = relevance_distributions(h, F)
    assert d["percent_undefined"]["rho_sem"] == pytest.approx(25.0)
    assert d["percent_undefined"]["rho_svc"] == pytest.approx(50.0)
    assert d["percent_undefined"]["both_so_M_is_0"] == pytest.approx(25.0)
    assert d["rho_sem_distribution"]["n"] == 3 and d["combined_M_distribution"]["n"] == 4


def test_tv_from_uniform_before_and_after_relevance():
    h = [rec([0, 1], [0.0, 1.0], w=[0.25, 0.75], w0=[0.5, 0.5])]
    d = relevance_distributions(h, F)["weights"]
    assert d["tv_from_uniform_after_relevance_applied"]["mean"] == pytest.approx(0.25)
    assert d["tv_from_uniform_before_relevance_counterfactual"]["mean"] == pytest.approx(0.0)


def test_inert_relevance_is_reported_as_not_distinguishing_clients():
    h = [rec([0, 1, 2], [0.0, 0.0, 0.0], sem=[None] * 3, svc=[None] * 3) for _ in range(4)]
    assert "inert" in relevance_distributions(h, F)["does_relevance_distinguish_clients"]


def test_single_update_rounds_have_nothing_to_compare():
    h = [rec([0], [0.5], sem=[0.5], svc=[0.5])]
    assert "no multi-update round" in relevance_distributions(h, F)["does_relevance_distinguish_clients"]


def test_old_history_without_components_is_flagged_unavailable():
    h = [rec([0, 1], [0.1, 0.2])]
    for r in h:
        del r["rho_sem"], r["rho_svc"]
    d = relevance_distributions(h, F)
    assert d["components_recorded"] is False and "unavailable" in d["rho_sem_distribution"]


def test_heterogeneity_what_if_is_monotone_and_zero_without_spread():
    rows = heterogeneity_sensitivity(F)
    tv = [r["tv_from_uniform"] for r in rows]
    assert tv[0] == pytest.approx(0.0) and all(b >= a - 1e-12 for a, b in zip(tv, tv[1:]))
    assert tv[-1] > tv[1] > 0


def test_diagnose_rounds_includes_the_new_sections():
    h = [rec([0, 1], [0.2, 0.4], sem=[0.2, 0.4], svc=[0.2, 0.4], w=[0.45, 0.55], w0=[0.5, 0.5])]
    d = diagnose_rounds(h, F)
    assert "relevance_distributions" in d and "heterogeneity_sensitivity_what_if" in d
