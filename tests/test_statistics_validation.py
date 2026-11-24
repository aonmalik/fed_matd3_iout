"""Final validation item 6: statistical evaluation (no fabricated significance, correct tests, corrections, effect sizes)."""
import math
import warnings

import numpy as np
import pytest

from fed_matd3_iout.evaluation.report_generator import comparisons, statistical_design_report
from fed_matd3_iout.evaluation.statistical_tests import (cohens_d_paired, hedges_g_paired, holm_bonferroni, mean_sd_ci, min_wilcoxon_p,
                                                        paired_test, seeds_needed_for_holm)


def mk(base, noise=0.0, seeds=range(4), rs=0):
    rng = np.random.default_rng(rs)
    return {s: [{"U_total": base + s + noise * rng.standard_normal() + i, "U_norm": 1.0, "delivered_mbit": base} for i in range(2)]
            for s in seeds}


def test_constant_differences_do_not_produce_significance_or_warnings():
    a = np.array([10., 11, 12, 13])
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                   # the scipy precision-loss warning must be gone
        r = paired_test(a, a - 2.0)
    assert r["degenerate"] and math.isnan(r["t_p"]) and math.isnan(r["cohens_d"]) and r["mean_diff"] == pytest.approx(2.0)
    assert r["wilcoxon_p"] == pytest.approx(0.125)                       # exact two-sided minimum for n=4: 2/2^4, NOT 0


def test_identical_observations_give_no_effect_and_no_pvalue():
    r = paired_test(np.ones(5), np.ones(5))
    assert r["degenerate"] and math.isnan(r["t_p"]) and math.isnan(r["wilcoxon_p"]) and r["cohens_d"] == 0.0


def test_nearly_identical_noise_is_treated_as_constant():
    a = np.array([1., 2, 3, 4, 5])
    r = paired_test(a + 1.0 + 1e-13 * np.arange(5), a)
    assert r["degenerate"] and math.isnan(r["t_p"])


def test_known_values_effect_size_ci_and_tests():
    a, b = np.array([5., 6, 7, 8, 9]), np.array([4., 5.2, 6, 7.1, 8])
    r = paired_test(a, b)
    d = a - b
    assert r["cohens_d"] == pytest.approx(d.mean() / d.std(ddof=1))
    assert r["hedges_g"] == pytest.approx(r["cohens_d"] * (1 - 3 / (4 * 4 - 1)))      # small-sample correction, n = 5
    assert r["diff_ci_lo"] < d.mean() < r["diff_ci_hi"] and r["t_p"] < 1e-3
    assert r["wilcoxon_p"] == pytest.approx(0.0625)                      # all 5 differences positive: exact p = 2/32
    assert not r["degenerate"]


def test_n1_has_no_spread_or_interval():
    s = mean_sd_ci([3.0])
    assert s["mean"] == 3.0 and math.isnan(s["sd"]) and math.isnan(s["ci_lo"]) and math.isnan(s["ci_hi"])


def test_holm_is_correct_and_monotone():
    adj = holm_bonferroni([0.01, 0.04, 0.03, float("nan")])
    assert adj[:3] == pytest.approx([0.03, 0.06, 0.06]) and math.isnan(adj[3])
    assert all(x >= p for x, p in zip(adj[:3], [0.01, 0.04, 0.03]))


def test_seed_budget_limits_nonparametric_testing_are_computed_not_asserted():
    assert min_wilcoxon_p(5) == pytest.approx(0.0625) and min_wilcoxon_p(5) > 0.05    # 5 seeds: Wilcoxon can never reach 0.05
    assert min_wilcoxon_p(6) <= 0.05
    assert seeds_needed_for_holm(10, 0.05) == 9 and seeds_needed_for_holm(1, 0.05) == 6
    d = statistical_design_report(5, 10)
    assert not d["wilcoxon_can_reach_alpha_uncorrected"] and not d["wilcoxon_can_reach_alpha_after_holm"]
    assert d["seeds_needed_for_holm_wilcoxon"] == 9


def test_comparisons_match_seeds_by_key_and_never_drop_silently():
    res = {"ref": mk(10.0, 0.3, rs=1), "a": mk(8.0, 0.3, rs=2), "b": mk(9.0, 0.3, seeds=[1, 2, 3, 4], rs=3)}   # b has a different seed set
    out = comparisons(res, "ref", ("U_total",))
    st = {r["method"]: r for r in out}
    assert st["b"]["status"] == "skipped_unmatched_seeds" and st["b"]["method_seeds"] == [1, 2, 3, 4]
    assert st["a"]["status"] == "ok" and "t_p_holm" in st["a"] and "wilcoxon_p_holm" in st["a"] and st["a"]["n_comparisons_in_family"] == 1


def test_holm_family_is_per_metric_and_per_test():
    res = {"ref": mk(10.0, 0.5, rs=1), "a": mk(9.0, 0.5, rs=2), "b": mk(9.5, 0.5, rs=3), "c": mk(7.0, 0.5, rs=4)}
    out = [r for r in comparisons(res, "ref", ("U_total",)) if r["status"] == "ok"]
    assert len(out) == 3 and all(r["n_comparisons_in_family"] == 3 for r in out)
    raw = sorted(r["t_p"] for r in out)
    assert min(r["t_p_holm"] for r in out) == pytest.approx(min(1.0, 3 * raw[0]))


def test_no_significance_is_reported_for_identical_methods():
    res = {"ref": mk(10.0, 0.5), "same": mk(10.0, 0.5)}
    r = [x for x in comparisons(res, "ref", ("U_total",))][0]
    assert r["degenerate"] and math.isnan(r["t_p"]) and math.isnan(r["t_p_holm"]) and r["mean_diff"] == 0.0
