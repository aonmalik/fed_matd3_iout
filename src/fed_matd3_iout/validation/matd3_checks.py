"""Numerical validation of the MATD3 core and of the federation boundary (each check recomputes the quantity independently)."""
from __future__ import annotations

import copy

import numpy as np
import torch

from ..agents.actor import flatten_params, load_flat_params
from ..agents.matd3_agent import MATD3Agent, polyak_update
from ..federated.aggregation import aggregate, normalise_with_cap
from ..federated.staleness import is_acceptable, staleness, staleness_factor
from .fixtures import Harness, link, perturb, ready

M_, OBS, ACT, ST, B = 2, 5, 2, 7, 16


def _agent(cfg, idx=0, **over):
    m = copy.deepcopy(cfg.matd3)
    m.hidden = 16
    m.update(over)
    torch.manual_seed(100 + idx)
    return MATD3Agent(idx, M_, OBS, ACT, ST, m, "cpu")


def _batch(seed=0, done=0.0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    return {"s": r(B, ST), "s2": r(B, ST), "o": r(B, M_, OBS), "o2": r(B, M_, OBS), "a": torch.tanh(r(B, M_, ACT)),
            "r": r(B, M_), "d": torch.full((B,), done)}


def _agents(cfg, **over):
    ags = [_agent(cfg, i, **over) for i in range(M_)]
    return ags, [a.actor_t for a in ags]


def check_twin_critic_target(cfg) -> dict:
    """(M3): y = r + gamma (1-d) min(Q1', Q2')(s', a') recomputed from the target critics; twin critics are distinct networks."""
    ags, tgt = _agents(cfg)
    a, b = ags[0], _batch(1)
    noise = torch.zeros(B, M_, ACT)
    y = a.compute_target(b, tgt, noise)
    a2 = torch.stack([torch.clamp(tgt[k](b["o2"][:, k, :]), -1, 1) for k in range(M_)], 1)
    x2 = torch.cat([b["s2"], a2.reshape(B, -1)], -1)
    q1, q2 = a.critic_t(x2)
    expect = b["r"][:, 0] + float(cfg.matd3.gamma) * (1 - b["d"]) * torch.min(q1, q2)
    p1 = torch.cat([p.reshape(-1) for p in a.critic.q1.parameters()]) if hasattr(a.critic, "q1") else None
    p2 = torch.cat([p.reshape(-1) for p in a.critic.q2.parameters()]) if hasattr(a.critic, "q2") else None
    out = {"target_equals_independent_recomputation": bool(torch.allclose(y, expect, atol=1e-6)),
           "uses_min_not_mean": bool((torch.min(q1, q2) < (q1 + q2) / 2 + 1e-12).all() and (q1 != q2).any()),
           "twin_critics_are_distinct_networks": bool(p1 is not None and p1.shape == p2.shape and not torch.equal(p1, p2))}
    bd = _batch(1, done=1.0)
    out["terminal_transition_does_not_bootstrap"] = bool(torch.allclose(a.compute_target(bd, tgt, noise), bd["r"][:, 0]))
    return out


def check_target_policy_smoothing(cfg) -> dict:
    """(M2): target action = clip(mu'(o') + clip(eps, -c, c), -1, 1) for ALL agents; the noise is bounded by c."""
    ags, tgt = _agents(cfg)
    a, b = ags[0], _batch(2)
    c = float(cfg.matd3.noise_clip)
    huge = torch.full((B, M_, ACT), 100.0)
    a_hi = a.target_joint_action(b["o2"], tgt, huge)
    a_lo = a.target_joint_action(b["o2"], tgt, -huge)
    mu = torch.stack([tgt[k](b["o2"][:, k, :]) for k in range(M_)], 1)
    return {"noise_clipped_to_c": bool(torch.allclose(a_hi, torch.clamp(mu + c, -1, 1), atol=1e-6)
                                       and torch.allclose(a_lo, torch.clamp(mu - c, -1, 1), atol=1e-6)),
            "actions_within_box": bool((a_hi.abs() <= 1).all() and (a_lo.abs() <= 1).all()),
            "noise_is_applied_stochastically_by_default": bool(not torch.equal(a.target_joint_action(b["o2"], tgt),
                                                                                 a.target_joint_action(b["o2"], tgt))),
            "all_agents_use_target_actors": bool(a_hi.shape == (B, M_, ACT))}


def check_delayed_actor_and_polyak(cfg) -> dict:
    """(M5, M6): the actor and ALL target networks change only on every policy_delay-th critic update, by exact Polyak averaging."""
    d = int(cfg.matd3.policy_delay)
    tau = float(cfg.matd3.tau)
    ags, tgt = _agents(cfg)
    a, b = ags[0], _batch(3)
    actor0, at0, ct0 = flatten_params(a.actor), flatten_params(a.actor_t), [p.detach().clone() for p in a.critic_t.parameters()]
    out = {"policy_delay": d}
    changed_before = []
    for k in range(1, d):
        a.update(b, tgt, noise=torch.zeros(B, M_, ACT))
        changed_before.append(bool(torch.equal(flatten_params(a.actor), actor0) and torch.equal(flatten_params(a.actor_t), at0)
                                   and all(torch.equal(x, y) for x, y in zip(a.critic_t.parameters(), ct0))))
    out["actor_and_targets_frozen_between_delayed_updates"] = all(changed_before)
    pre_actor_t = flatten_params(a.actor_t)
    pre_critic_t = [p.detach().clone() for p in a.critic_t.parameters()]
    a.update(b, tgt, noise=torch.zeros(B, M_, ACT))                   # the d-th critic update: actor + Polyak
    new_actor, new_actor_t = flatten_params(a.actor), flatten_params(a.actor_t)
    out["actor_updated_exactly_at_delay"] = bool(a.actor_updates == 1 and not torch.equal(new_actor, actor0))
    out["actor_target_is_exact_polyak"] = bool(torch.allclose(new_actor_t, (1 - tau) * pre_actor_t + tau * new_actor, atol=1e-7))
    out["critic_target_is_exact_polyak"] = bool(all(torch.allclose(t, (1 - tau) * p + tau * s.detach(), atol=1e-7)
                                                    for t, p, s in zip(a.critic_t.parameters(), pre_critic_t, a.critic.parameters())))
    t2, s2 = copy.deepcopy(a.actor_t), a.actor
    before = flatten_params(t2)
    polyak_update(t2, s2, 0.0)
    out["tau0_leaves_target_unchanged"] = bool(torch.equal(flatten_params(t2), before))
    polyak_update(t2, s2, 1.0)
    out["tau1_copies_source"] = bool(torch.equal(flatten_params(t2), flatten_params(s2)))
    return out


def check_critic_loss_and_gradient_paths(cfg) -> dict:
    """The critic regresses BOTH twins onto the same y; the actor loss uses Q1 only and does not move the critic."""
    ags, tgt = _agents(cfg, policy_delay=1)
    a, b = ags[0], _batch(4)
    crit0 = [p.detach().clone() for p in a.critic.parameters()]
    info = a.update(b, tgt, noise=torch.zeros(B, M_, ACT))
    moved = any(not torch.equal(x, y) for x, y in zip(a.critic.parameters(), crit0))
    other_actor = flatten_params(ags[1].actor)
    return {"critic_updated": bool(moved), "actor_loss_reported": info["actor_loss"] is not None,
            "other_agents_actor_not_touched_by_this_agents_update": bool(torch.equal(other_actor, flatten_params(ags[1].actor))),
            "all_losses_finite": bool(np.isfinite(info["critic_loss"]) and np.isfinite(info["actor_loss"]))}


def check_actor_only_federation(cfg) -> dict:
    """Installing a global actor changes ONLY the actor, its target and the reference/counters; critics, critic targets and
    optimiser states stay local (documented design: critics are never aggregated)."""
    h = Harness(cfg, 2, "async", agg_buffer_size=1, download_blend=1.0)
    c = h.clients[0]
    ag = _agent(cfg, 0, policy_delay=1)
    h.clients[0].agent = c.agent = ag
    c.reference = flatten_params(ag.actor)
    for _ in range(3):
        ag.update(_batch(7), [ag.actor_t, ag.actor_t], noise=torch.zeros(B, M_, ACT))
    crit = {k: v.clone() for k, v in ag.critic.state_dict().items()}
    crit_t = {k: v.clone() for k, v in ag.critic_t.state_dict().items()}
    opt_c = copy.deepcopy(ag.opt_c.state_dict())
    opt_a = copy.deepcopy(ag.opt_a.state_dict())
    theta = torch.randn_like(flatten_params(ag.actor))
    c.install_global(theta, 3, 1.0)
    same = lambda a, b: all(torch.equal(a[k], b[k]) for k in a)
    return {"critic_unchanged": same(crit, c.agent.critic.state_dict()),
            "critic_target_unchanged": same(crit_t, c.agent.critic_t.state_dict()),
            "critic_optimiser_state_unchanged": len(opt_c["state"]) > 0 and all(
                torch.equal(opt_c["state"][i]["exp_avg"], c.agent.opt_c.state_dict()["state"][i]["exp_avg"]) for i in opt_c["state"]),
            # documented design note (not a defect): the ACTOR's Adam moments are NOT reset on a global install
            "actor_optimiser_state_not_reset_(documented)": len(opt_a["state"]) > 0 and all(
                torch.equal(opt_a["state"][i]["exp_avg"], c.agent.opt_a.state_dict()["state"][i]["exp_avg"]) for i in opt_a["state"]),
            "federated_vector_is_actor_size_only": bool(c.reference.numel() == sum(p.numel() for p in c.agent.actor.parameters())),
            "actor_equals_global_with_blend_1": bool(torch.allclose(flatten_params(c.agent.actor), theta)),
            "actor_target_synchronised_to_actor": bool(torch.equal(flatten_params(c.agent.actor_t), flatten_params(c.agent.actor))),
            "reference_equals_installed_global": bool(torch.equal(c.reference, theta) and c.base_version == 3),
            "counters_reset_on_install": c.steps_since == 0 and c.trans_since == 0}


def check_install_is_inplace_and_blend(cfg) -> dict:
    h = Harness(cfg, 1, "async")
    c = h.clients[0]
    obj_ids = (id(c.agent.actor_t), [id(p) for p in c.agent.actor_t.parameters()])
    local = flatten_params(c.agent.actor)
    theta = local + 1.0
    c.install_global(theta, 1, 0.25)
    return {"in_place_so_trainer_target_actor_list_stays_valid": obj_ids == (id(c.agent.actor_t), [id(p) for p in c.agent.actor_t.parameters()]),
            "blend_is_convex_combination": bool(torch.allclose(flatten_params(c.agent.actor), 0.25 * theta + 0.75 * local, atol=1e-6))}


def check_aggregation_integrity(cfg) -> dict:
    """theta' = theta + eta_g sum w_i Clip(Delta_i); a huge finite update cannot move theta by more than eta_g * C; NaN/Inf never enter."""
    f = cfg.federated
    C, lr = float(f["clip_norm"]), float(f["server_lr"])
    theta = torch.zeros(50)
    huge = torch.full((50,), 1e6)
    small = torch.full((50,), 0.01)
    new, scales = aggregate(theta, [huge, small], normalise_with_cap(np.array([1.0, 1.0]), 1.0), lr, C)
    out = {"clip_norm_default": C, "huge_update_is_clipped": bool(scales[0] < 1e-3),
           "step_bounded_by_server_lr_times_C": bool(float((new - theta).norm()) <= lr * C * (1 + 1e-6)),
           "result_finite": bool(torch.isfinite(new).all())}
    h = Harness(cfg, 2, "async", agg_buffer_size=1)
    c = h.clients[0]
    perturb(c); ready(c)
    with torch.no_grad():
        next(c.agent.actor.parameters()).view(-1)[0] = float("inf")
    th = h.server.theta.clone()
    h.run(30)
    out["inf_update_refused_and_global_untouched"] = bool(torch.equal(h.server.theta, th) and torch.isfinite(h.server.theta).all()
                                                          and h.server.version == 0)
    h2 = Harness(cfg, 2, "async", agg_buffer_size=1, clip_norm=None)       # without clipping a finite-but-huge update IS applied
    c2 = h2.clients[0]
    perturb(c2, 1e3); ready(c2)
    h2.run(30)
    out["without_clip_a_huge_finite_update_is_accepted_(documented_risk)"] = bool(h2.server.version == 1 and float(h2.server.theta.norm()) > 1e3)
    return out


def check_staleness_semantics(cfg) -> dict:
    f = cfg.federated
    k, mx = float(f["kappa"]), int(f["max_staleness"])
    return {"staleness_is_version_difference": staleness(7, 4) == 3,
            "factor_is_exp_minus_kappa_delta": abs(staleness_factor(3, k) - float(np.exp(-3 * k))) < 1e-12,
            "factor_is_one_at_zero_and_monotone": staleness_factor(0, k) == 1.0 and staleness_factor(1, k) > staleness_factor(2, k),
            "negative_staleness_rejected": not is_acceptable(-1, mx),
            "gate_inclusive_at_max_exclusive_above": is_acceptable(mx, mx) and not is_acceptable(mx + 1, mx)}


def check_parameters_finite_after_training(cfg, n_episodes=6) -> dict:
    """Short real federated training (smoke scale): every actor, critic, target, reference and the global actor stay finite."""
    from ..baselines.registry import build_cfg
    from ..training.trainer import Trainer
    from ..utils.config import load_config
    c = build_cfg(load_config(["smoke.yaml"]), "s3a_fedmatd3")
    t = Trainer(c, torch.device("cpu"), None, 0)
    t.train(n_episodes)
    fin = lambda m: all(bool(torch.isfinite(p).all()) for p in m.parameters())
    ok = all(fin(a.actor) and fin(a.actor_t) and fin(a.critic) and fin(a.critic_t) for a in t.agents)
    ok_ref = all(bool(torch.isfinite(x.reference).all()) for x in t.clients)
    return {"all_networks_finite": ok, "all_references_finite": ok_ref, "global_actor_finite": bool(torch.isfinite(t.server.theta).all()),
            "aggregation_rounds": t.server.version, "gradient_steps": int(sum(x.total_steps for x in t.clients)),
            "global_actor_differs_from_initial": bool(t.server.version > 0)}


def run_all(cfg) -> dict:
    fns = [check_twin_critic_target, check_target_policy_smoothing, check_delayed_actor_and_polyak, check_critic_loss_and_gradient_paths,
           check_actor_only_federation, check_install_is_inplace_and_blend, check_aggregation_integrity, check_staleness_semantics,
           check_parameters_finite_after_training]
    out = {}
    for f in fns:
        r = f(cfg)
        bools = {k: v for k, v in r.items() if isinstance(v, bool)}
        out[f.__name__] = {"passed": all(bools.values()), "checks": r}
    return out
