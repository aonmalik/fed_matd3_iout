"""Diagnostic reporting for the (unchanged) semantic aggregation rule  Z_i = n_i^a * exp(-k D_i) * [eps + (1-eps) M_i]^eta.

Everything here is DESCRIPTIVE: it reads the per-round records written by ``USVServer.aggregate_now`` and never alters them.
Question answered: does the relevance factor have enough dynamic range to change the aggregation measurably?

Per aggregation round R (|R| = number of accepted updates):
  * participation: |R|;  single-update rounds (|R| = 1) have weight exactly 1, so staleness and relevance are INERT there.
  * relevance factor g_i = eps + (1-eps) M_i  in [eps, 1];  within-round spread max g / min g.
  * contribution to the final weight, exact log-variance decomposition over the clients of a round (|R| >= 2):
        log Z_i = a log n_i  +  (-k D_i)  +  eta log g_i ,   share_c = Cov(c, log Z) / Var(log Z)   (the shares sum to 1)
    (cap renormalisation is applied afterwards by the server and ignored here; it only reduces dispersion).
  * counterfactual effect (recorded by the server): TV(w, w_without_relevance) and the relative L2 change of the applied
    parameter update caused by relevance alone.
  * variation across rounds: per-client mean / std / CV of the final weight and of M, and per-round weight entropy.
"""
from __future__ import annotations

import numpy as np


def _q(x, qs=(0.05, 0.5, 0.95)):
    x = np.asarray(x, float)
    return [float(v) for v in np.quantile(x, qs)] if len(x) else None


def _share(comp, total):
    v = np.var(total)
    if v < 1e-18:
        return None
    return float(np.cov(comp, total, bias=True)[0, 1] / v)


def diagnose_rounds(history: list[dict], fcfg) -> dict:
    eps, eta, alpha, kappa = (float(fcfg[k]) for k in ("eps_m", "eta", "alpha", "kappa"))
    rounds = len(history)
    if rounds == 0:
        return {"n_rounds": 0, "note": "no aggregation round recorded"}
    sizes = np.array([len(r["clients"]) for r in history])
    multi = [r for r in history if len(r["clients"]) >= 2]
    per_client: dict[int, dict[str, list]] = {}
    for r in history:
        for c, m, w, st in zip(r["clients"], r["relevance"], r["w"], r["stale_factor"]):
            d = per_client.setdefault(int(c), {"M": [], "w": [], "stale": []})
            d["M"].append(float(m)); d["w"].append(float(w)); d["stale"].append(float(st))
    pc = {}
    for c, d in sorted(per_client.items()):
        w = np.array(d["w"]); m = np.array(d["M"])
        pc[c] = {"rounds_participated": len(w), "relevance_mean": float(m.mean()), "relevance_min": float(m.min()),
                 "relevance_max": float(m.max()), "weight_mean": float(w.mean()), "weight_std": float(w.std()),
                 "weight_cv": float(w.std() / w.mean()) if w.mean() > 0 else None}
    shares = {"sample_count": [], "staleness": [], "relevance": []}
    spread, tv, eff, ent = [], [], [], []
    for r in multi:
        n = np.asarray(r["n"], float); st = np.asarray(r["stale_factor"], float); m = np.clip(np.asarray(r["relevance"], float), 0, 1)
        g = eps + (1 - eps) * m
        comps = {"sample_count": alpha * np.log(np.maximum(n, 1e-12)), "staleness": np.log(np.maximum(st, 1e-12)), "relevance": eta * np.log(g)}
        tot = sum(comps.values())
        for k, v in comps.items():
            s = _share(v, tot)
            if s is not None:
                shares[k].append(s)
        spread.append(float(g.max() / g.min()))
        if "cf_rel_tv" in r:
            tv.append(float(r["cf_rel_tv"])); eff.append(float(r["cf_rel_effect"]))
    for r in history:
        w = np.asarray(r["w"], float); w = w[w > 0]
        ent.append(float(-(w * np.log(w)).sum()))
    all_m = np.array([x for r in history for x in r["relevance"]], float)
    mm = np.array([x for r in multi for x in r["relevance"]], float)
    out = {
        "n_rounds": rounds,
        "participants_per_round": {"mean": float(sizes.mean()), "histogram": {int(k): int((sizes == k).sum()) for k in np.unique(sizes)}},
        "single_update_rounds": int((sizes == 1).sum()),
        "single_update_round_fraction": float((sizes == 1).mean()),
        "multi_update_rounds": len(multi),
        "relevance_M": {"all_updates_min_mean_max": [float(all_m.min()), float(all_m.mean()), float(all_m.max())],
                        "multi_rounds_quantiles_5_50_95": _q(mm)},
        "relevance_factor_g": {"floor_eps_m": eps, "possible_range": [eps, 1.0],
                               "observed_range": [float(eps + (1 - eps) * all_m.min()), float(eps + (1 - eps) * all_m.max())],
                               "within_round_spread_max_over_min": {"quantiles_5_50_95": _q(spread), "max": float(max(spread)) if spread else None}},
        "weight_variance_shares_multi_rounds": {k: ({"mean": float(np.mean(v)), "n": len(v)} if v else None) for k, v in shares.items()},
        "counterfactual_relevance_effect": {
            "weight_total_variation": {"mean": float(np.mean(tv)), "quantiles_5_50_95": _q(tv), "max": float(max(tv))} if tv else None,
            "relative_update_change_L2": {"mean": float(np.mean(eff)), "quantiles_5_50_95": _q(eff), "max": float(max(eff))} if eff else None,
            "note": "relevance changes the applied update by this relative L2 amount versus the same round with relevance factor == 1"},
        "weight_entropy_per_round": {"mean": float(np.mean(ent)), "std": float(np.std(ent))},
        "per_client": pc,
        "inert_rounds_note": "single-update rounds have weight 1.0: sample count, staleness and relevance cannot matter there",
    }
    out["relevance_distributions"] = relevance_distributions(history, fcfg)
    out["heterogeneity_sensitivity_what_if"] = heterogeneity_sensitivity(fcfg)
    out["verdict"] = verdict(out)
    return out


def _dist(x):
    x = np.asarray([v for v in x if v is not None], float)
    if not len(x):
        return None
    return {"n": int(len(x)), "min": float(x.min()), "mean": float(x.mean()), "max": float(x.max()), "std": float(x.std()),
            "quantiles_5_50_95": _q(x)}


def relevance_distributions(history: list[dict], fcfg) -> dict:
    """Protocol 2.2 additions (descriptive; nothing here feeds back into the update rule or tunes a parameter).

    * distributions of rho_sem, rho_svc and combined M over ACCEPTED updates, and the share of updates where each component is
      undefined (None);
    * between-client variation: within-round standard deviation / range of M (rounds with >= 2 updates);
    * aggregation weights before relevance (counterfactual ``w_norel`` recorded by the server) and after (applied ``w``);
    * total-variation distance of the applied weights from UNIFORM aggregation over the round's participants
      (TV_uniform = 0.5 sum |w_i - 1/|R||), and of the no-relevance weights from uniform, so the part of the departure from
      uniform that sample counts/staleness cause can be separated from the part relevance causes.
    Records written by an older protocol lack ``rho_sem``/``rho_svc``; those fields are then reported as unavailable.
    """
    if not history:
        return {"n_rounds": 0}
    sem = [x for r in history for x in r.get("rho_sem", [])]
    svc = [x for r in history for x in r.get("rho_svc", [])]
    have = all("rho_sem" in r for r in history)
    M = [float(x) for r in history for x in r["relevance"]]
    multi = [r for r in history if len(r["clients"]) >= 2]
    within_std = [float(np.std(r["relevance"])) for r in multi]
    within_rng = [float(np.ptp(r["relevance"])) for r in multi]
    tv_after, tv_before = [], []
    for r in multi:
        k = len(r["w"]); u = np.full(k, 1.0 / k)
        tv_after.append(float(0.5 * np.abs(np.asarray(r["w"]) - u).sum()))
        if "w_norel" in r:
            tv_before.append(float(0.5 * np.abs(np.asarray(r["w_norel"]) - u).sum()))
    undefined_both = [(a is None and b is None) for r in history for a, b in zip(r.get("rho_sem", []), r.get("rho_svc", []))]
    return {
        "n_rounds": len(history), "n_updates": len(M), "components_recorded": bool(have),
        "rho_sem_distribution": _dist(sem) if have else "unavailable (history from an older protocol)",
        "rho_svc_distribution": _dist(svc) if have else "unavailable (history from an older protocol)",
        "combined_M_distribution": _dist(M),
        "percent_undefined": ({"rho_sem": 100.0 * float(np.mean([x is None for x in sem])) if sem else None,
                               "rho_svc": 100.0 * float(np.mean([x is None for x in svc])) if svc else None,
                               "both_so_M_is_0": 100.0 * float(np.mean(undefined_both)) if undefined_both else None} if have else None),
        "between_client_variation_in_multi_rounds": {"n_rounds": len(multi), "within_round_std_of_M": _dist(within_std),
                                                     "within_round_range_of_M": _dist(within_rng)},
        "weights": {"tv_from_uniform_after_relevance_applied": _dist(tv_after),
                    "tv_from_uniform_before_relevance_counterfactual": _dist(tv_before) if tv_before else None,
                    "note": "uniform = equal weight over the round's participants; departure before relevance is due to n_i and staleness only"},
        "does_relevance_distinguish_clients": _reading(within_std, tv_after, tv_before, M),
    }


def _reading(within_std, tv_after, tv_before, M) -> str:
    if not within_std:
        return "no multi-update round: relevance cannot distinguish clients (nothing to compare within a round)"
    m = float(np.mean(M)); s = float(np.median(within_std))
    extra = float(np.median(tv_after)) - (float(np.median(tv_before)) if tv_before else 0.0)
    if m < 0.05:
        return (f"M is ~0 for almost all updates (mean {m:.3f}): the factor sits at the floor eps_M, so relevance is inert; "
                "it does not distinguish clients under these conditions")
    return (f"median within-round std of M = {s:.3f}; relevance adds about {extra:+.3f} to the median TV distance from uniform. "
            "Read against the scale of M, not as evidence of benefit.")


def heterogeneity_sensitivity(fcfg, spreads=(0.0, 0.05, 0.1, 0.2, 0.4), k=3, centre=0.5, n_i=None) -> list[dict]:
    """WHAT-IF on the IMPLEMENTED formula (not a measurement): K clients with equal n_i and no staleness, M values spread
    symmetrically around ``centre`` with half-range ``s``.  Reports the TV distance from uniform caused by relevance alone.
    Shows how large a between-client spread of M the current (eps_M, eta) needs before weights move; parameters are NOT tuned."""
    from ..federated.aggregation import normalise_with_cap, raw_weights
    eps, eta = float(fcfg["eps_m"]), float(fcfg["eta"])
    n = np.ones(k) if n_i is None else np.asarray(n_i, float)
    out = []
    for s in spreads:
        m = np.linspace(centre - s, centre + s, k) if k > 1 else np.array([centre])
        m = np.clip(m, 0.0, 1.0)
        w = normalise_with_cap(raw_weights(n, np.ones(k), m, 1.0, eta, eps), 1.0)
        u = np.full(k, 1.0 / k)
        out.append({"half_range_of_M": float(s), "M": [float(x) for x in m], "weights": [float(x) for x in w],
                    "tv_from_uniform": float(0.5 * np.abs(w - u).sum())})
    return out


def verdict(d: dict) -> dict:
    """Plain, threshold-explicit reading of the numbers (the thresholds are for READING the diagnostic, not tuned on results)."""
    f_single, tvs = d["single_update_round_fraction"], d["counterfactual_relevance_effect"]["weight_total_variation"]
    eff = d["counterfactual_relevance_effect"]["relative_update_change_L2"]
    share = d["weight_variance_shares_multi_rounds"]["relevance"]
    rows = {
        "fraction_of_rounds_where_relevance_can_act": 1.0 - f_single,
        "median_weight_tv_in_multi_rounds": None if tvs is None else tvs["quantiles_5_50_95"][1],
        "median_relative_update_change": None if eff is None else eff["quantiles_5_50_95"][1],
        "relevance_share_of_weight_log_variance": None if share is None else share["mean"],
    }
    tv_med = rows["median_weight_tv_in_multi_rounds"]
    if tv_med is None:
        read = "NO multi-update round observed: relevance had no opportunity to act"
    elif tv_med < 0.02:
        read = "relevance moves the weights by <2% (TV) in a typical multi-update round: contribution is NOT expected to be measurable"
    elif tv_med < 0.10:
        read = "relevance moves the weights by 2-10% (TV): small; detectable only with many seeds and a paired design"
    else:
        read = "relevance moves the weights by >=10% (TV) in a typical multi-update round: a measurable mechanism, effect on reward not implied"
    return {**rows, "reading": read, "thresholds_tv": [0.02, 0.10],
            "caveat": "descriptive only; says nothing about whether the shift improves performance"}
