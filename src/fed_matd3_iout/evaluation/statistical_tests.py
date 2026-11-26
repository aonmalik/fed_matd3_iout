"""Mean / SD / 95% CI (t-based), paired tests, effect sizes, multiple-comparison correction.

Unit of replication for method comparison = training seed (each seed's score = mean over the SAME matched evaluation
scenarios); seeds are independent training runs (distinct environment streams and network initialisations).  Pairing is by seed
(common random numbers for the environment stream).  Mission-to-mission variability is averaged inside each seed score.

Rules enforced here (final validation, item 6):
  * No statistic is invented for degenerate data.  If the paired differences are (numerically) constant, the t statistic is
    undefined: ``t_p`` is NaN, ``degenerate`` is True and the reason is stored.  (SciPy would return t = inf, p = 0.)
  * n < 2 gives NaN SD / CI (a zero-width interval from one observation would be a fabrication).
  * Effect size: paired Cohen's d_z and its small-sample corrected Hedges' g; NaN when the SD of the differences is 0.
  * CI of the mean paired difference (t-based) is reported next to the p-values.
  * Multiple comparisons: Holm-Bonferroni across the competing methods, separately for the t-test and the Wilcoxon p-values.
  * Wilcoxon signed-rank has a smallest attainable two-sided p of 2^(1-n_nonzero): with 5 seeds it can NEVER be < 0.05
    (``min_wilcoxon_p``); ``seeds_needed_for_holm`` gives the smallest n for which a Holm-corrected Wilcoxon test can reject.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

_REL_TOL = 1e-9


def mean_sd_ci(x, conf: float = 0.95) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    m = float(x.mean())
    if n == 1:                                           # one observation carries no information about spread
        return {"n": 1, "mean": m, "sd": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    sd = float(x.std(ddof=1))
    h = float(stats.t.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n))
    return {"n": n, "mean": m, "sd": sd, "ci_lo": m - h, "ci_hi": m + h}


def _is_constant(d: np.ndarray) -> bool:
    """True if the differences are identical up to floating-point noise (the t statistic is then undefined)."""
    scale = max(1.0, float(np.abs(d).max())) if len(d) else 1.0
    return len(d) < 2 or float(np.ptp(d)) <= _REL_TOL * scale


def cohens_d_paired(a, b) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) < 2:
        return float("nan")
    if _is_constant(d):                                  # SD of the differences is 0: d_z is undefined unless the mean is 0 too
        return 0.0 if abs(float(d.mean())) <= _REL_TOL else float("nan")
    return float(d.mean() / d.std(ddof=1))


def hedges_g_paired(a, b) -> float:
    d = cohens_d_paired(a, b)
    n = len(np.asarray(a))
    return float(d * (1.0 - 3.0 / (4.0 * (n - 1) - 1.0))) if n > 2 and not math.isnan(d) else float("nan")


def min_wilcoxon_p(n_nonzero: int) -> float:
    """Smallest attainable two-sided exact p-value of the Wilcoxon signed-rank test with n non-zero differences."""
    return min(1.0, 2.0 ** (1 - int(n_nonzero))) if n_nonzero > 0 else 1.0


def seeds_needed_for_holm(k: int, alpha: float = 0.05) -> int:
    """Smallest n such that the Holm-corrected smallest p (= k * min_wilcoxon_p) can reach alpha for k comparisons."""
    n = 1
    while k * min_wilcoxon_p(n) > alpha and n < 64:
        n += 1
    return n


def paired_test(a, b, conf: float = 0.95) -> dict:
    """Paired t-test and Wilcoxon signed-rank on matched samples (a - b), with effect size and CI of the mean difference."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired test requires matched samples"
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    d = a - b
    n = len(d)
    out = {"n": n, "mean_diff": float(d.mean()) if n else float("nan"),
           "sd_diff": float(d.std(ddof=1)) if n > 1 else float("nan"),
           "cohens_d": cohens_d_paired(a, b) if n else float("nan"), "hedges_g": hedges_g_paired(a, b) if n else float("nan"),
           "diff_ci_lo": float("nan"), "diff_ci_hi": float("nan"), "t_p": float("nan"), "wilcoxon_p": float("nan"),
           "wilcoxon_n_nonzero": int(np.count_nonzero(d)), "wilcoxon_min_attainable_p": min_wilcoxon_p(int(np.count_nonzero(d))),
           "degenerate": False, "degenerate_reason": ""}
    if n < 2:
        out.update(degenerate=True, degenerate_reason="fewer than 2 matched pairs")
        return out
    if _is_constant(d):                                  # identical differences: t undefined; do NOT report p = 0
        out.update(degenerate=True, degenerate_reason="paired differences are (numerically) constant: t-test undefined")
    else:
        h = float(stats.t.ppf(0.5 + conf / 2, n - 1) * out["sd_diff"] / np.sqrt(n))
        out["diff_ci_lo"], out["diff_ci_hi"] = out["mean_diff"] - h, out["mean_diff"] + h
        out["t_p"] = float(stats.ttest_rel(a, b).pvalue)
    if np.any(d != 0):
        try:                                             # exact for small n without ties/zeros, normal approximation otherwise
            out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
        except ValueError:
            out["wilcoxon_p"] = float("nan")
    return out


def holm_bonferroni(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    m = int(np.sum(~np.isnan(p)))
    adj = np.full_like(p, np.nan)
    run = 0.0
    for rank, idx in enumerate(order[:m]):
        run = max(run, (m - rank) * p[idx])
        adj[idx] = min(1.0, run)
    return adj.tolist()
