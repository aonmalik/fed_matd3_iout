"""Aggregate per-seed results into tables (Markdown/CSV) and compare methods with paired tests."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .comparability import ATTR, comparability, deployment
from .statistical_tests import holm_bonferroni, mean_sd_ci, paired_test

KEY_METRICS = ["U_total", "U_norm", "delivered_mbit", "overflow_mbit", "collection_ratio", "delivery_ratio",
               "critical_pdr", "E_total_j", "U_per_E", "mean_age_s", "unsafe_events", "workload_imbalance",
               "backlog_clearance_time_s", "backlog_clearance_censored"]
# D10: separate fixed-workload measurement (only present in results produced with experiment.fixed_workload_eval=true)
FIXED_WORKLOAD_METRICS = ["fixed_workload_completion_time_s", "fixed_workload_censored", "fixed_workload_delivered_fraction"]


def load_results(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def seed_scores(per_seed: dict, metric: str) -> np.ndarray:
    """per_seed: {seed: [scenario metrics dicts]} -> array over seeds of scenario-mean."""
    out = []
    for _, rows in sorted(per_seed.items()):
        v = np.array([r[metric] for r in rows], dtype=float)
        out.append(float(np.nanmean(v)) if np.any(~np.isnan(v)) else float("nan"))
    return np.array(out)


def table(results: dict, metrics=KEY_METRICS) -> list[dict]:
    rows = []
    has_fw = all(FIXED_WORKLOAD_METRICS[0] in next(iter(ps.values()))[0] for ps in results.values() if ps and next(iter(ps.values())))
    if metrics is KEY_METRICS and has_fw:
        metrics = KEY_METRICS + FIXED_WORKLOAD_METRICS
    for method, per_seed in results.items():
        row = {"method": method, "deployment": deployment(method) if method in ATTR else "n/a"}
        for m in metrics:
            s = mean_sd_ci(seed_scores(per_seed, m))
            row[m] = f"{s['mean']:.4g} ± {s['sd']:.3g} [{s['ci_lo']:.4g}, {s['ci_hi']:.4g}]"
        rows.append(row)
    return rows


def comparisons(results: dict, reference: str, metrics=("U_total", "U_norm", "delivered_mbit")) -> list[dict]:
    """Paired comparison of every method against ``reference`` on matched TRAINING SEEDS.

    * Seeds are matched by key (not by position); a method whose seed set differs from the reference is reported as
      ``status='skipped_unmatched_seeds'`` instead of being silently dropped or mis-paired.
    * Holm-Bonferroni is applied across the competing methods, separately per metric and separately for the t-test and the
      Wilcoxon p-values.  Degenerate comparisons (constant differences) carry NaN p-values, never p = 0.
    """
    out = []                                   # NOTE: rows of methods outside the registry (ablation tags) carry no comparability field
    ref_keys = sorted(results[reference])
    for m in metrics:
        ref = seed_scores(results[reference], m)
        rows = []
        for method, per_seed in results.items():
            if method == reference:
                continue
            if sorted(per_seed) != ref_keys:
                rows.append({"method": method, "metric": m, "reference": reference, "status": "skipped_unmatched_seeds",
                             "reference_seeds": ref_keys, "method_seeds": sorted(per_seed)})
                continue
            r = paired_test(ref, seed_scores(per_seed, m))
            r.update(method=method, metric=m, reference=reference, status="degenerate" if r["degenerate"] else "ok")
            if method in ATTR and reference in ATTR:
                r["comparability"] = comparability(reference, method)
            rows.append(r)
        ok = [r for r in rows if r["status"] != "skipped_unmatched_seeds"]
        for key in ("t_p", "wilcoxon_p"):
            for r, a in zip(ok, holm_bonferroni([r[key] for r in ok])):
                r[key + "_holm"] = a
        for r in ok:
            r["n_comparisons_in_family"] = len(ok)
        out.extend(rows)
    return out


def statistical_design_report(n_seeds: int, n_methods_vs_reference: int, alpha: float = 0.05) -> dict:
    """What the seed count can and cannot support (computed, not asserted)."""
    from .statistical_tests import min_wilcoxon_p, seeds_needed_for_holm
    return {"n_seeds": n_seeds, "n_comparisons_per_metric": n_methods_vs_reference, "alpha": alpha,
            "wilcoxon_min_attainable_p": min_wilcoxon_p(n_seeds),
            "wilcoxon_can_reach_alpha_uncorrected": min_wilcoxon_p(n_seeds) <= alpha,
            "wilcoxon_can_reach_alpha_after_holm": n_methods_vs_reference * min_wilcoxon_p(n_seeds) <= alpha,
            "seeds_needed_for_holm_wilcoxon": seeds_needed_for_holm(n_methods_vs_reference, alpha),
            "seeds_needed_for_uncorrected_wilcoxon": seeds_needed_for_holm(1, alpha)}


def write_markdown(rows: list[dict], path: str | Path) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    Path(path).write_text("\n".join(lines) + "\n")


def write_csv(rows: list[dict], path: str | Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
