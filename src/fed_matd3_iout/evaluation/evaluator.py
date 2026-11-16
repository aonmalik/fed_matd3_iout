"""Deterministic evaluation on matched scenarios (same layout / arrivals / fading / comm seeds across methods).

Evaluation scenario k uses stream offset eval_scenario_base + k for every stream; training episodes use
offsets < 10^6*... disjoint ranges.  No learning and no federation happens during evaluation.
"""
from __future__ import annotations

import numpy as np

from ..environment.multi_auv_iout_env import MultiAUVIoUTEnv
from .metrics import episode_metrics


def rollout(cfg, policies, scenario: int, env: MultiAUVIoUTEnv | None = None) -> dict:
    env = env or MultiAUVIoUTEnv(cfg)
    obs, _ = env.reset(int(cfg["seeds"]["eval_scenario_base"]) + int(scenario))
    G_seen_clear = None
    for t in range(env.T):
        act = np.stack([policies[i](obs[i]) for i in range(env.M)])
        obs, r, term, trunc, info = env.step(act)
        acc = env.accounting()
        if G_seen_clear is None and acc["G"] > 0 and (acc["residual"] + acc["onboard"]) <= 0.10 * acc["G"]:
            G_seen_clear = (t + 1) * env.dt
        if trunc:
            break
    env.completion_time_s = G_seen_clear if G_seen_clear is not None else env.T * env.dt
    m = episode_metrics(env)
    m["completion_censored"] = float(G_seen_clear is None)
    return m


def evaluate(cfg, policies, n_scenarios: int | None = None) -> list[dict]:
    n = int(n_scenarios or cfg["experiment"]["eval_scenarios"])
    env = MultiAUVIoUTEnv(cfg)
    return [rollout(cfg, policies, k, env) for k in range(n)]


def summarise(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = float(np.nanmean(v)) if np.any(~np.isnan(v)) else float("nan")
    return out
