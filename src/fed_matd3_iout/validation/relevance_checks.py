"""Mathematical validation of the batch-attribution ledger (BatchBook) and of the relevance -> weight map.

 * ``auv1_example``            : hand-checkable numbers: a batch collected by AUV 1 and delivered later is credited to AUV 1.
 * ``real_env_mass_balance``   : in the real environment, claimed + open + late utility/units/value equal the global ledger.
 * ``weight_stability``        : Lipschitz-type sensitivity of the aggregation weights to the relevance score.
 * ``maturity_sensitivity``    : descriptive sweep of the maturity threshold eps (NO performance metric is involved).
"""
from __future__ import annotations

import math

import numpy as np

from ..environment.multi_auv_iout_env import MultiAUVIoUTEnv
from ..federated.aggregation import normalise_with_cap, raw_weights
from ..federated.relevance import relevance, rho_sem
from ..semantic.attribution import BatchBook
from ..semantic.packet import Batch
from ..baselines.reference_controller import goal_seeking_policies

MU = 0.01


def _b(pid, t_collect, units, sigma, c, mu=MU, auv=0):
    return Batch(pid, 0, t_collect - 5.0, units, sigma, "x", mu, c, "onboard", t_collect, auv)


def _u(b, t_del):                                   # same decay law as the ledger: age measured from GENERATION
    return b.units * b.c * b.sigma * math.exp(-b.mu * (t_del - b.t_gen))


def auv1_example(eps=0.05) -> dict:
    """AUV 1 collects batch B1 at t=100 s and delivers it at t=250 s.  AUV 0 collects/delivers its own batch B0 around t=240-260 s,
    i.e. AUV 0 is the AUV 'active' at delivery time.  Returns every number so that the result can be checked by hand."""
    bk = BatchBook(2, eps)
    b1 = _b(11, 100.0, units=10, sigma=0.8, c=2.0, auv=1)           # V = 10*2*0.8 = 16
    b0 = _b(7, 235.0, units=6, sigma=0.5, c=1.0, auv=0)             # V = 6*1*0.5 = 3
    bk.on_collect(1, b1)
    bk.on_collect(0, b0)
    u1 = _u(b1, 250.0)
    u0 = _u(b0, 255.0)
    bk.on_deliver(1, b1, u1)                                         # delivered at t=250 s (delayed by 150 s)
    bk.on_deliver(0, b0, u0)                                         # AUV 0 delivers at t=255 s
    now = 600.0
    c0, c1 = bk.claim(0, now), bk.claim(1, now)
    old_style = {                                                    # legacy window formula: U_del/V_coll inside the SAME time window
        "AUV1 window [0,200): V_coll=16, U_del=0 ->": 0.0,
        "AUV1 window [200,400): V_coll=0 (nothing collected), U_del=%.4f -> den=0 -> 0 (delivery ignored)" % u1: 0.0}
    return {"batch_B1": {"collected_by": 1, "t_collect": 100.0, "t_deliver": 250.0, "units": 10, "sigma": 0.8, "c": 2.0, "mu": MU,
                         "V": 16.0, "U_delivered": u1},
            "batch_B0": {"collected_by": 0, "t_collect": 235.0, "t_deliver": 255.0, "V": 3.0, "U_delivered": u0},
            "claim_AUV1": c1, "claim_AUV0": c0,
            "rho_sem_AUV1": rho_sem(c1), "rho_sem_AUV0": rho_sem(c0),
            "AUV0_received_none_of_B1": abs(c0["U_mature"] - u0) < 1e-12 and abs(c0["V_mature"] - 3.0) < 1e-12,
            "AUV1_credited_with_B1_only": abs(c1["U_mature"] - u1) < 1e-12 and abs(c1["V_mature"] - 16.0) < 1e-12,
            "closed_form_rho_AUV1": u1 / 16.0, "legacy_window_ratio_AUV1": old_style}


# ------------------------------------------------------------------------------------------------ real environment
class FanoutBook:
    """Feeds the SAME collect/deliver events to several BatchBooks (one per maturity eps) so they can be compared on one rollout."""

    def __init__(self, n_auv, eps_list):
        self.eps_list = list(eps_list)
        self.books = {e: BatchBook(n_auv, e) for e in self.eps_list}
        self.claims = {e: [] for e in self.eps_list}                  # (auv, now, claim dict)

    def on_collect(self, auv, b):
        for bk in self.books.values():
            bk.on_collect(auv, b)

    def on_deliver(self, auv, chunk, u):
        for bk in self.books.values():
            bk.on_deliver(auv, chunk, u)
        return "fan"

    def claim_all(self, auv, now):
        for e, bk in self.books.items():
            self.claims[e].append((auv, now, bk.claim(auv, now)))

    def reset_episode(self):
        for bk in self.books.values():
            bk.reset_episode()


def rollout_with_fanout(cfg, scenario, eps_list, claim_every_steps=100):
    env = MultiAUVIoUTEnv(cfg)
    obs, _ = env.reset(int(cfg["seeds"]["eval_scenario_base"]) + scenario)
    fan = FanoutBook(env.M, eps_list)
    env.book = fan
    pol = goal_seeking_policies(env.M)
    for t in range(env.T):
        obs, r, term, trunc, info = env.step(np.stack([pol[i](obs[i]) for i in range(env.M)]))
        if (t + 1) % claim_every_steps == 0:                         # emulate periodic update creation (claim mature batches)
            for a in range(env.M):
                fan.claim_all(a, env.step_idx * env.dt)
        if trunc:
            break
    for a in range(env.M):                                           # final claim at the end of the episode
        fan.claim_all(a, env.step_idx * env.dt)
    return env, fan


def real_env_mass_balance(cfg, scenarios=(0, 1, 2), eps=0.05, claim_every_steps=100) -> dict:
    out = []
    for k in scenarios:
        env, fan = rollout_with_fanout(cfg, k, [eps], claim_every_steps)
        bk = fan.books[eps]
        rows = []
        for a in range(env.M):
            pend = bk.pending(a)
            rows.append({"auv": a,
                         "U_ledger": env.ledger.by_auv.get(a, 0.0), "U_book": bk.utility_accounted(a),
                         "units_del_book": bk.claimed_units_del[a] + pend["units_del"] + bk.late_units[a],
                         "V_book": bk.claimed_V[a] + pend["V"]})
        coll_units = sum(s.queue.collected for s in env.sensors)
        tot_U_ledger, tot_U_book = env.ledger.U, sum(r["U_book"] for r in rows)
        bounds_ok = all(c["U_mature"] <= c["V_mature"] + 1e-9 and c["U_mature"] >= -1e-12 for _, _, c in fan.claims[eps])
        out.append({"scenario": k, "U_ledger": tot_U_ledger, "U_book": tot_U_book,
                    "U_abs_err": abs(tot_U_ledger - tot_U_book),
                    "per_auv_abs_err": max(abs(r["U_ledger"] - r["U_book"]) for r in rows),
                    "delivered_units_env": env.delivered_units, "delivered_units_book": sum(r["units_del_book"] for r in rows),
                    "collected_units_env": coll_units,
                    "bounds_ok": bool(bounds_ok), "n_claims": len(fan.claims[eps]),
                    "late_units": float(sum(bk.late_units))})
    return {"eps": eps, "scenarios": out,
            "passed": all(o["U_abs_err"] < 1e-9 and o["per_auv_abs_err"] < 1e-9
                          and abs(o["delivered_units_env"] - o["delivered_units_book"]) < 1e-6 and o["bounds_ok"] for o in out)}


# ------------------------------------------------------------------------------------------------ weight stability
def weight_stability(fcfg, n_updates=4, grid=21) -> dict:
    """Sensitivity of the normalised weights to ONE client's relevance (others fixed) on a deterministic grid of (n_i, staleness, M).

    Analytic bound without the cap:  w_i = Z_i/sum Z,  Z_i = A_i f(M_i),  f(M) = [eps_M + (1-eps_M) M]^eta  =>
        |dw_i/dM_i| = w_i (1-w_i) f'/f  <=  (1/4) eta (1-eps_M)/eps_M   (since w(1-w) <= 1/4 and f'/f <= eta (1-eps_M)/eps_M).
    The relevance factor itself lies in [eps_M^eta, 1], so relevance alone can change a relative weight by at most eps_M^(-eta).
    The cap (iterative renormalisation) is reported empirically only."""
    eps_m, eta, alpha, kappa = (float(fcfg[k]) for k in ("eps_m", "eta", "alpha", "kappa"))
    ns = [100.0, 400.0, 1600.0, 400.0][:n_updates]
    sts = [1.0, math.exp(-kappa), math.exp(-2 * kappa), 1.0][:n_updates]
    base_rel = [0.2, 0.5, 0.8, 0.5][:n_updates]
    ms = np.linspace(0.0, 1.0, grid)

    def sweep(cap):
        worst, valid = 0.0, True
        for i in range(n_updates):
            ws = []
            for m in ms:
                rel = list(base_rel); rel[i] = float(m)
                w = normalise_with_cap(raw_weights(ns, sts, rel, alpha, eta, eps_m), cap)
                valid &= bool(abs(w.sum() - 1.0) < 1e-9 and (w >= -1e-12).all())
                ws.append(w)
            worst = max(worst, float((np.abs(np.diff(np.array(ws), axis=0)) / (ms[1] - ms[0])).max()))
        return worst, valid
    w_nocap, v1 = sweep(1.0)
    w_cap, v2 = sweep(float(fcfg["max_client_weight"]))
    bound = 0.25 * eta * (1.0 - eps_m) / eps_m
    return {"eps_m": eps_m, "eta": eta, "max_client_weight": float(fcfg["max_client_weight"]),
            "relevance_factor_range": [eps_m ** eta, 1.0], "max_relative_weight_change_from_relevance_alone": eps_m ** (-eta),
            "analytic_bound_abs_dw_dM_nocap": bound, "empirical_max_abs_dw_dM_nocap": w_nocap,
            "empirical_max_abs_dw_dM_with_cap": w_cap, "weights_valid_everywhere": bool(v1 and v2),
            "bounded": bool(w_nocap <= bound + 1e-9)}


# ------------------------------------------------------------------------------------------------ maturity sensitivity
def maturity_sensitivity(cfg, eps_list=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5), scenarios=(0, 1, 2), claim_every_steps=100) -> dict:
    """Descriptive effect of the maturity threshold eps on rho_sem for ONE fixed scripted policy and fixed scenarios.
    No training performance is used, so eps cannot be selected by 'which value looks best'; the default (0.05) was fixed a priori."""
    per = {e: {"U": 0.0, "V": 0.0, "n_mature": 0, "n_immature_seen": 0, "censored_V": 0.0, "units_del": 0.0, "units_coll": 0.0,
               "late_units": 0.0, "claim_rho": [], "by_auv": {}} for e in eps_list}
    keyed: dict[float, dict[tuple, float]] = {e: {} for e in eps_list}
    total_V = 0.0
    for k in scenarios:
        env, fan = rollout_with_fanout(cfg, k, eps_list, claim_every_steps)
        for e in eps_list:
            bk = fan.books[e]
            bk.reset_episode()                                     # whatever is still unclaimed at the end is censored
            per[e]["censored_V"] += bk.censored_V
            per[e]["late_units"] += float(sum(bk.late_units))
            for a, now, c in fan.claims[e]:
                per[e]["U"] += c["U_mature"]; per[e]["V"] += c["V_mature"]; per[e]["n_mature"] += int(c["n_mature_batches"])
                per[e]["units_del"] += c["units_del_mature"]; per[e]["units_coll"] += c["units_coll_mature"]
                u, v = per[e]["by_auv"].get((k, a), (0.0, 0.0))
                per[e]["by_auv"][(k, a)] = (u + c["U_mature"], v + c["V_mature"])
            # per-(scenario, AUV, claim-index) relevance for rank comparisons
            for idx, (a, now, c) in enumerate(fan.claims[e]):
                r = rho_sem(c)
                keyed[e][(k, a, idx // env.M)] = float("nan") if r is None else r
                if r is not None:
                    per[e]["claim_rho"].append(r)
        total_V += sum(bk_claim["V_mature"] for _, _, bk_claim in fan.claims[eps_list[0]]) + fan.books[eps_list[0]].censored_V
    from scipy.stats import spearmanr
    ref = 0.05 if 0.05 in eps_list else eps_list[0]
    out = []
    for e in eps_list:
        p = per[e]
        keys = [q for q in keyed[e] if not (math.isnan(keyed[e][q]) or math.isnan(keyed[ref].get(q, float("nan"))))]
        rho_rank = (float(spearmanr([keyed[e][q] for q in keys], [keyed[ref][q] for q in keys]).correlation)
                    if len(keys) >= 3 else float("nan"))
        denom = p["V"] + p["censored_V"]
        out.append({"eps": e, "horizon_factor_ln_1_over_eps": math.log(1 / e),
                    "pooled_rho_sem": (p["U"] / p["V"]) if p["V"] > 0 else float("nan"),
                    "mean_claim_rho_sem": float(np.mean(p["claim_rho"])) if p["claim_rho"] else float("nan"),
                    "n_defined_claims": len(p["claim_rho"]), "mature_batches": p["n_mature"],
                    "mature_value_V": p["V"], "censored_value_V": p["censored_V"],
                    "censored_share_of_value": p["censored_V"] / denom if denom > 0 else float("nan"),
                    "late_units_not_reattributed": p["late_units"], "late_units_share_of_mature_units": p["late_units"] / max(p["units_coll"], 1.0),
                    "across_auv_pooled_rho_sem": {"min": min((u / v for u, v in p["by_auv"].values() if v > 0), default=float("nan")),
                                                  "max": max((u / v for u, v in p["by_auv"].values() if v > 0), default=float("nan"))},
                    "spearman_vs_reference_eps": rho_rank})
    return {"policy": "goal_seeking_reference_controller (scripted, no learning)", "scenarios": list(scenarios),
            "claim_every_steps": claim_every_steps, "reference_eps": ref, "rows": out,
            "note": "Descriptive only. Smaller eps = longer wait before a batch is judged = fewer, later, but better-resolved claims."}


# ------------------------------------------------------------------------------------------------ dynamic range under reference policies
def dynamic_range_reference_policy(cfg, scenarios=(0, 1), window_steps=100, policy="goal_seeking") -> dict:
    """Ceiling/reference measurement of the relevance dynamic range WITHOUT any learning: roll the real environment with a fixed,
    non-learning policy, cut the AUVs' statistics into windows exactly as the federated protocol does (``pop_window_stats``) and
    evaluate M_i (s3a) per (scenario, window, AUV).  Each window is then treated as one aggregation round with all AUVs
    participating, equal sample counts and no staleness, so the only source of weight difference is relevance.
    Policies: 'goal_seeking' (competent heuristic from baselines.reference_controller) or 'zero' (do nothing).  Descriptive only."""
    import numpy as np
    from ..baselines.reference_controller import goal_seeking_policies
    from ..environment.multi_auv_iout_env import MultiAUVIoUTEnv
    from ..federated.aggregation import normalise_with_cap, raw_weights
    from ..federated.relevance import relevance, rho_sem, rho_svc
    f = cfg.federated
    env = MultiAUVIoUTEnv(cfg)
    pols = goal_seeking_policies(env.M) if policy == "goal_seeking" else [(lambda o: np.zeros(2)) for _ in range(env.M)]
    Ms, rows, tvs, spreads, defined = [], [], [], [], {"sem": 0, "svc": 0, "n": 0}
    for k in scenarios:
        obs, _ = env.reset(int(cfg["seeds"]["eval_scenario_base"]) + int(k))
        for t in range(env.T):
            obs, *_ = env.step(np.stack([pols[i](obs[i]) for i in range(env.M)]))
            if (t + 1) % window_steps:
                continue
            st = [env.pop_window_stats(i) for i in range(env.M)]
            m = np.array([relevance(s, "s3a", float(f["beta_sem"])) for s in st])
            for s in st:
                defined["n"] += 1; defined["sem"] += rho_sem(s) is not None; defined["svc"] += rho_svc(s) is not None
            Ms.append(m)
            ones = np.ones(env.M)
            w = normalise_with_cap(raw_weights(ones, ones, m, 1.0, float(f["eta"]), float(f["eps_m"])), float(f["max_client_weight"]))
            w0 = np.full(env.M, 1.0 / env.M)
            tvs.append(float(0.5 * np.abs(w - w0).sum()))
            g = float(f["eps_m"]) + (1 - float(f["eps_m"])) * m
            spreads.append(float(g.max() / g.min()))
            rows.append({"scenario": k, "t": t + 1, "M": m.round(4).tolist()})
    M = np.array(Ms)
    q = lambda x: [float(v) for v in np.quantile(x, (0.05, 0.5, 0.95))]
    return {"policy": policy, "window_steps": window_steps, "scenarios": list(scenarios), "n_windows": len(Ms),
            "M_quantiles_5_50_95": q(M), "M_min_max": [float(M.min()), float(M.max())],
            "fraction_M_zero": float((M == 0).mean()),
            "within_window_spread_of_M_max_minus_min": q(M.max(1) - M.min(1)),
            "factor_g_spread_max_over_min": q(spreads), "equal_n_no_staleness_weight_TV_vs_uniform": q(tvs),
            "fraction_components_defined": {"rho_sem": defined["sem"] / defined["n"], "rho_svc": defined["svc"] / defined["n"]},
            "note": "reference-policy measurement, not a learning result; TV is vs uniform weights because n and staleness are equal here",
            "windows": rows[:12]}
