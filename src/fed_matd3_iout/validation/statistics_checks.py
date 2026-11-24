"""Executed checks of the statistics code (implementation correctness, NOT evidence about any method)."""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..evaluation.report_generator import comparisons, statistical_design_report
from ..evaluation.statistical_tests import holm_bonferroni, min_wilcoxon_p, paired_test, seeds_needed_for_holm


def run_all(n_seeds: int, n_comparisons: int, alpha: float = 0.05) -> dict:
    rs = np.random.default_rng(12345)
    a = rs.normal(10, 1, 10); b = a - 0.5 + rs.normal(0, 0.3, 10)
    r = paired_test(a, b)
    ch = {}
    ch["paired_t_matches_scipy"] = abs(r["t_p"] - stats.ttest_rel(a, b).pvalue) < 1e-12
    ch["paired_wilcoxon_matches_scipy"] = abs(r["wilcoxon_p"] - stats.wilcoxon(a, b).pvalue) < 1e-12
    ch["ci_of_mean_difference_contains_estimate_and_is_ordered"] = r["diff_ci_lo"] < r["mean_diff"] < r["diff_ci_hi"]
    ch["hedges_g_smaller_in_magnitude_than_cohens_d"] = abs(r["hedges_g"]) < abs(r["cohens_d"])
    c = paired_test(a, a - 1.0)                                       # constant difference
    ch["constant_difference_gives_nan_p_not_zero"] = bool(c["degenerate"] and np.isnan(c["t_p"]))
    p = [0.001, 0.02, 0.04, 0.5, float("nan")]
    h = holm_bonferroni(p)
    ch["holm_not_smaller_than_raw_and_capped_at_one"] = all(hh >= pp - 1e-15 and hh <= 1 for hh, pp in zip(h[:4], p[:4]))
    ch["holm_matches_manual_computation"] = np.allclose(h[:4], [0.004, 0.06, 0.08, 0.5])
    ch["holm_preserves_nan"] = bool(np.isnan(h[4]))
    ch["wilcoxon_min_p_is_exact_two_to_one_minus_n"] = abs(min_wilcoxon_p(10) - 2.0 ** -9) < 1e-15
    x = np.arange(1.0, 11.0)
    ch["exact_wilcoxon_attains_the_stated_minimum"] = abs(stats.wilcoxon(x, np.zeros(10)).pvalue - min_wilcoxon_p(10)) < 1e-15
    res = {"ref": {s: [{"U_total": 10 + s + 0.1 * k} for k in range(3)] for s in range(4)},
           "m1": {s: [{"U_total": 9 + s + 0.07 * k * (s + 1)} for k in range(3)] for s in range(4)},
           "m2": {s: [{"U_total": 9 + s} for k in range(3)] for s in (3, 2, 1, 0)},          # same seeds, different insertion order
           "m3": {s: [{"U_total": 9.0} for k in range(3)] for s in range(3)}}                # unmatched seed set
    cmp = {x["method"]: x for x in comparisons(res, "ref", ("U_total",))}
    ch["unmatched_seed_sets_are_skipped_not_mispaired"] = cmp["m3"]["status"] == "skipped_unmatched_seeds"
    ch["seeds_matched_by_key_not_by_position"] = cmp["m2"]["status"] in ("ok", "degenerate") and abs(cmp["m2"]["mean_diff"] - 1.1) < 1e-9
    d = statistical_design_report(n_seeds, n_comparisons, alpha)
    ch["design_report_consistent_with_min_p"] = (d["wilcoxon_can_reach_alpha_after_holm"] == (n_comparisons * min_wilcoxon_p(n_seeds) <= alpha)
                                                  and (n_seeds >= seeds_needed_for_holm(n_comparisons, alpha)) == d["wilcoxon_can_reach_alpha_after_holm"])
    return {"passed": all(ch.values()), "checks": {k: bool(v) for k, v in ch.items()}, "design": d,
            "design_adequacy": {
                "kind": "attainability_only",
                "holm_corrected_wilcoxon_attainable": bool(d["wilcoxon_can_reach_alpha_after_holm"]),
                "statement": ("Attainable means the exact test CAN return p <= alpha/k at this seed count if every paired difference has the same "
                              "sign. It does NOT mean it will, and it says nothing about statistical power for any realistic effect size. "
                              "Power depends on the unknown between-seed variance and effect sizes and is not estimated here. "
                              "No result is implied or guaranteed by the seed count.")}}
