"""Aggregate per-seed results into tables (Markdown/CSV) and compare methods with paired tests."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .statistical_tests import holm_bonferroni, mean_sd_ci, paired_test

KEY_METRICS = ["U_total", "U_norm", "delivered_mbit", "overflow_mbit", "collection_ratio", "delivery_ratio",
               "critical_pdr", "E_total_j", "U_per_E", "mean_age_s", "unsafe_events", "workload_imbalance"]


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
    for method, per_seed in results.items():
        row = {"method": method}
        for m in metrics:
            s = mean_sd_ci(seed_scores(per_seed, m))
            row[m] = f"{s['mean']:.4g} ± {s['sd']:.3g} [{s['ci_lo']:.4g}, {s['ci_hi']:.4g}]"
        rows.append(row)
    return rows


def comparisons(results: dict, reference: str, metrics=("U_total", "U_norm", "delivered_mbit")) -> list[dict]:
    out = []
    for m in metrics:
        ref = seed_scores(results[reference], m)
        names, ps = [], []
        for method, per_seed in results.items():
            if method == reference or len(per_seed) != len(results[reference]):
                continue
            r = paired_test(ref, seed_scores(per_seed, m))
            r.update(method=method, metric=m, reference=reference)
            out.append(r)
            ps.append(r["t_p"])
        adj = holm_bonferroni([o["t_p"] for o in out if o["metric"] == m])
        for o, a in zip([o for o in out if o["metric"] == m], adj):
            o["t_p_holm"] = a
    return out


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
