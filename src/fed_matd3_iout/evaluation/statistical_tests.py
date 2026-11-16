"""Mean / SD / 95% CI (t-based), paired tests, effect sizes, multiple-comparison correction.

Unit of replication for method comparison = training seed (each seed's score = mean over the SAME matched
evaluation scenarios).  Mission-to-mission variability is therefore averaged inside each seed score and reported
separately (``within_seed_sd``); training variability is the across-seed SD.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def mean_sd_ci(x, conf: float = 0.95) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    m = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    h = float(stats.t.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    return {"n": n, "mean": m, "sd": sd, "ci_lo": m - h, "ci_hi": m + h}


def cohens_d_paired(a, b) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    return float(d.mean() / sd) if sd > 0 else (0.0 if d.mean() == 0 else float("inf"))


def paired_test(a, b) -> dict:
    """Paired t-test and Wilcoxon signed-rank on matched samples (a - b)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired test requires matched samples"
    out = {"n": len(a), "mean_diff": float((a - b).mean()), "cohens_d": cohens_d_paired(a, b)}
    if len(a) >= 2 and np.any(a != b):
        out["t_p"] = float(stats.ttest_rel(a, b).pvalue)
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
        except ValueError:
            out["wilcoxon_p"] = float("nan")
    else:
        out["t_p"] = out["wilcoxon_p"] = float("nan")
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
