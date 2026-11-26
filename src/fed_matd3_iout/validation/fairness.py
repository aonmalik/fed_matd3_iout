"""Baseline-fairness audit: what is identical across methods (verified numerically) and what cannot be made equivalent (declared)."""
from __future__ import annotations

import copy

import numpy as np
import torch

from ..baselines.registry import METHODS, build_cfg
from ..environment.multi_auv_iout_env import MultiAUVIoUTEnv
from ..evaluation.comparability import deployment
from ..training.trainer import TRAIN_OFFSET_STRIDE

# keys of the merged config that must be IDENTICAL for every method unless the method's own override list names them
PROTECTED_SECTIONS = ("env", "traffic", "energy", "acoustic", "semantic", "reward", "seeds", "experiment", "usv")

NON_EQUIVALENT = [
    {"id": "N1", "methods": ["greedy_matd3", "hungarian_matd3"], "aspect": "task allocation",
     "detail": "Different allocator by design (Paper-2 greedy / Hungarian on the priority matrix). All other methods use the proposed STA allocator, so only learning differs for them.",
     "can_equalise": False},
    {"id": "N2", "methods": ["mappo"], "aspect": "gradient-update budget",
     "detail": "Same environment-step budget, but MAPPO does epochs x 4 minibatch steps per 500-step rollout (~16 minibatch updates per 1000 env steps) versus one gradient step per env step per agent for the MATD3 family. On-policy data cannot be replayed, so the update budgets cannot be equalised without changing the algorithm.",
     "can_equalise": False},
    {"id": "N3", "methods": ["mappo"], "aspect": "policy class / exploration",
     "detail": "Shared tanh-squashed Gaussian actor (Protocol 2.2: latent Gaussian, executed action tanh(u), Jacobian-corrected PPO likelihood) with learned log-std and a latent-entropy bonus; MATD3 family uses per-AUV deterministic actors with additive Gaussian noise on a decay schedule. Evaluation is deterministic (mean action) for all.",
     "can_equalise": False},
    {"id": "N4", "methods": ["mappo"], "aspect": "parameter sharing",
     "detail": "MAPPO shares ONE actor across AUVs for training and evaluation (value input adds an AUV one-hot).",
     "can_equalise": False},
    {"id": "N5", "methods": ["fedavg_matd3", "stale_fedmatd3", "semantic_fedmatd3", "service_fedmatd3", "s3a_fedmatd3"],
     "aspect": "deployed policy",
     "detail": "Federated methods are scored with the single GLOBAL actor deployed to every AUV (headline `global`); non-federated MATD3-family methods are scored with each AUV's own actor. Local-actor scores of federated methods are also stored (`local`).",
     "can_equalise": False},
    {"id": "N6", "methods": ["fedavg_matd3", "stale_fedmatd3", "semantic_fedmatd3", "service_fedmatd3", "s3a_fedmatd3"],
     "aspect": "initialisation",
     "detail": "Federated methods start from a common initial actor (agent 0's initialisation copied to all AUVs, as in FL); non-federated methods use independent per-AUV initialisations from the same seed stream.",
     "can_equalise": True, "note": "Equalising would make independent initialisation the only difference; kept as is (standard FL practice) and documented."},
    {"id": "N7", "methods": ["fedavg_matd3", "stale_fedmatd3", "semantic_fedmatd3", "service_fedmatd3", "s3a_fedmatd3"],
     "aspect": "model-traffic cost",
     "detail": "Federated methods spend acoustic airtime and communication energy on model transfers (part of the proposed trade-off); non-federated methods spend none. This is intended and is reported in the comm/energy metrics.",
     "can_equalise": False},
    {"id": "N8", "methods": ["centralized_matd3", "maddpg", "mappo", "greedy_matd3", "hungarian_matd3"], "aspect": "information at training time",
     "detail": "Centralised critics read the simulator global state (CTDE) and, for MATD3/MADDPG, the other agents' target actors and the shared joint replay store, i.e. training is NOT communication-constrained for these methods and is also not constrained for the federated methods' critics (only their ACTORS are communicated). Execution uses local observations only for every method.",
     "can_equalise": False},
    {"id": "N9", "methods": ["independent_td3"], "aspect": "critic information",
     "detail": "Independent TD3 critic sees only (o_i, a_i): intentionally less information (baseline definition).", "can_equalise": False},
    {"id": "N10", "methods": ["maddpg"], "aspect": "TD3 stabilisers",
     "detail": "MADDPG disables twin critics, delayed actor updates and target smoothing by definition.", "can_equalise": False},
    {"id": "N11", "methods": ["mappo"], "aspect": "training computation-energy accounting",
     "detail": "MODELLED vs MEASURED: the MATD3 family charges the ASSUMED constant e_grad_j per local gradient step (training E_comp_j, a model parameter, not a measurement); MAPPO charges nothing, so its training E_comp_j is NOT 0 'energy' but UNACCOUNTED. D9 (provisional): compute energy is reported in joules only from a real measurement (utils/compute_energy.py); no instrument exists in the build environment, so every method reports compute energy as `not_measured` next to its measured compute TIME (training_compute in evaluation.json). Cross-method TRAINING total-energy comparisons are therefore INCOMPLETE and no energy-efficiency claim may rest on them. Mission energy (propulsion + communication) and EVALUATION energy (no learning in any method) are separate and unaffected.",
     "can_equalise": False},
]


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _env_signature(cfg, offset, steps=15):
    env = MultiAUVIoUTEnv(cfg)
    env.reset(offset)
    pos = np.array([s.pos for s in env.sensors])
    z = np.zeros((env.M, env.act_dim))
    for _ in range(steps):
        env.step(z)
    return pos, float(sum(s.queue.generated for s in env.sensors)), env.obs_dim, env.act_dim, env.state_dim


def run_audit(base_cfg, n_probe_scenarios=3) -> dict:
    base = copy.deepcopy(base_cfg)
    flat_base = _flat(base)
    rows, checks = [], {}
    sigs = {}
    for name, m in METHODS.items():
        cfg = build_cfg(base, name)
        flat = _flat(cfg)
        changed = sorted(k for k in set(flat) | set(flat_base) if flat.get(k) != flat_base.get(k))
        declared = set(m["overrides"])
        undeclared = [k for k in changed if k not in declared]
        protected = [k for k in changed if k.split(".")[0] in PROTECTED_SECTIONS]
        total_steps = int(cfg.experiment.train_episodes) * int(cfg.env.episode_steps)
        if m["trainer"] == "mappo":
            ppo = cfg.matd3.ppo
            grads = total_steps // int(ppo.rollout_steps) * int(ppo.epochs) * 4
        else:
            grads = max(0, total_steps - int(cfg.matd3.warmup)) // int(cfg.matd3.update_every)
        rows.append({"method": name, "trainer": m["trainer"], "changed_keys": changed, "undeclared_changes": undeclared,
                     "changed_protected_sections": protected, "federated": bool(cfg.federated.enabled),
                     "aggregation": cfg.federated.aggregation if cfg.federated.enabled else None,
                     "allocation": cfg.allocation.method, "critic_mode": cfg.matd3.critic_mode, "twin": bool(cfg.matd3.twin),
                     "env_step_budget": total_steps, "gradient_updates_per_agent_or_minibatch": grads,
                     "eval_scenarios": int(cfg.experiment.eval_scenarios), "eval_policy": "global actor" if cfg.federated.enabled else "own actors",
                     "deployment": deployment(name)})
        sigs[name] = [_env_signature(cfg, int(cfg.seeds.eval_scenario_base) + k) for k in range(n_probe_scenarios)]
    ref = next(iter(sigs.values()))
    same_env = all(all(np.array_equal(a[0], b[0]) and a[1:] == b[1:] for a, b in zip(ref, s)) for s in sigs.values())
    checks["only_declared_keys_differ_per_method"] = all(not r["undeclared_changes"] for r in rows)
    checks["no_method_changes_environment_traffic_energy_channel_semantic_reward_seed_or_experiment_sections"] = all(
        not r["changed_protected_sections"] for r in rows)
    checks["identical_scenario_layout_arrivals_obs_act_state_dims_for_all_methods"] = bool(same_env)
    checks["same_env_step_budget_for_all_methods"] = len({r["env_step_budget"] for r in rows}) == 1
    checks["same_eval_scenarios_for_all_methods"] = len({r["eval_scenarios"] for r in rows}) == 1
    checks["observation_and_action_spaces_identical"] = len({(s[0][2], s[0][3]) for s in sigs.values()}) == 1
    n_train = int(base.experiment.train_episodes)
    checks["train_and_eval_scenario_seeds_disjoint"] = ((int(base.experiment.n_train_seeds) - 1) * TRAIN_OFFSET_STRIDE + n_train
                                                         <= int(base.seeds.eval_scenario_base)) and n_train <= TRAIN_OFFSET_STRIDE
    checks["training_seeds_use_distinct_scenario_offsets"] = n_train <= TRAIN_OFFSET_STRIDE     # seed s uses [s*stride, s*stride + n_train)
    mat = [r for r in rows if r["trainer"] == "matd3"]
    checks["matd3_family_same_gradient_budget_except_none"] = len({r["gradient_updates_per_agent_or_minibatch"] for r in mat}) == 1
    from ..evaluation.comparability import ATTR, FEDERATED, comparability
    checks["every_method_has_comparability_attributes"] = set(ATTR) == set(METHODS)
    checks["comparability_attributes_match_built_configs"] = all(
        (r["method"] in FEDERATED) == r["federated"] and (r["trainer"] == "mappo") == (ATTR[r["method"]][1] == "mappo")
        and (r["allocation"] == "greedy") == (ATTR[r["method"]][2] == "greedy") and (r["critic_mode"] == "independent") == (ATTR[r["method"]][3] == "independent")
        and r["twin"] == ATTR[r["method"]][4] for r in rows)
    names = list(METHODS)
    matrix = {a: {b: comparability(a, b)["class"] for b in names if b != a} for a in names}
    n_like = sum(1 for a in names for b in names if a < b and matrix[a][b].startswith("like_for_like"))
    return {"rows": rows, "checks": checks, "non_equivalent": NON_EQUIVALENT, "comparability_matrix": matrix,
            "n_pairs_like_for_like": n_like, "n_pairs_total": len(names) * (len(names) - 1) // 2,
            "statement": "The 11 methods are NOT equivalent: numeric equivalence holds only for observations, action constraints, environment, "
                         "step budget, evaluation scenarios and seeds. Deployment, trainer, allocator and information differences are declared.",
            "gradient_budget_ratio_matd3_to_mappo_minibatches":
                next(r["gradient_updates_per_agent_or_minibatch"] for r in mat) / next(
                    r["gradient_updates_per_agent_or_minibatch"] for r in rows if r["trainer"] == "mappo")}
