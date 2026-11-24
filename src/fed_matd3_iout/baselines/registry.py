"""Method registry: every method is (config overrides on the SAME base config) + a trainer kind.

Fairness rules enforced here:
  * identical environment, traffic, evaluation missions and experience budget for all methods;
  * non-allocation baselines use the proposed STA allocator so that only the LEARNING component differs;
  * weighting-only federated baselines (stale / semantic / service) keep the proposed clipping and weight cap,
    so only the weighting differs; FedAvg-MATD3 is the conventional configuration (no clipping/cap/relevance/staleness).
"""
from __future__ import annotations

import copy

from ..utils.config import set_by_path, validate_config

NOFED = {"federated.enabled": False}

METHODS: dict[str, dict] = {
    "greedy_matd3": {"overrides": {**NOFED, "allocation.method": "greedy"}, "trainer": "matd3",
                     "desc": "Greedy (Paper 2 priority, sequential) allocation + MATD3 mobility, no federation.",
                     "note": "Allocation differs; learning identical to centralized_matd3."},
    "hungarian_matd3": {"overrides": {**NOFED, "allocation.method": "hungarian_priority"}, "trainer": "matd3",
                        "desc": "Hungarian assignment on the Paper 2 priority matrix + MATD3 mobility, no federation.",
                        "note": "Allocation uses priority without service-arrival prediction/semantics."},
    "centralized_matd3": {"overrides": {**NOFED}, "trainer": "matd3",
                          "desc": "Standard CTDE MATD3: per-AUV actors, centralised twin critics, NO model communication (no federation).",
                          "note": "Upper reference w.r.t. communication: actors never wait for the USV."},
    "independent_td3": {"overrides": {**NOFED, "matd3.critic_mode": "independent"}, "trainer": "matd3",
                        "desc": "Independent TD3: critic Q(o_i, a_i), no other-agent information.",
                        "note": "Not CTDE; non-stationarity is not addressed."},
    "maddpg": {"overrides": {**NOFED, "matd3.twin": False, "matd3.policy_delay": 1, "matd3.target_noise": 0.0},
               "trainer": "matd3", "desc": "MADDPG: centralised single critic, no TD3 stabilisers.", "note": ""},
    "mappo": {"overrides": {**NOFED}, "trainer": "mappo",
              "desc": "MAPPO: on-policy PPO with shared actor and centralised value.",
              "note": "On-policy; not directly comparable in sample efficiency (same env-step budget)."},
    "fedavg_matd3": {"overrides": {"federated.aggregation": "fedavg", "federated.clip_norm": None,
                                   "federated.max_client_weight": 1.0}, "trainer": "matd3",
                     "desc": "Conventional FedAvg weights (n_i) on the same asynchronous protocol.",
                     "note": "n_i-weighted; under async arrival it is FedAvg-style only, not synchronous FedAvg."},
    "stale_fedmatd3": {"overrides": {"federated.aggregation": "stale"}, "trainer": "matd3",
                       "desc": "n_i * exp(-kappa*Delta).", "note": ""},
    "semantic_fedmatd3": {"overrides": {"federated.aggregation": "semantic"}, "trainer": "matd3",
                          "desc": "Semantic-only relevance (+staleness).", "note": ""},
    "service_fedmatd3": {"overrides": {"federated.aggregation": "service"}, "trainer": "matd3",
                         "desc": "Service-only relevance (+staleness).", "note": ""},
    "s3a_fedmatd3": {"overrides": {"federated.aggregation": "s3a"}, "trainer": "matd3",
                     "desc": "Proposed: semantic-service-staleness-aware asynchronous FedMATD3.", "note": "Proposed."},
}

ABLATIONS: dict[str, dict] = {
    "full": {},
    "no_semantic_utility": {"semantic.use_in_decisions": False},
    "no_service_arrival_prediction": {"allocation.backlog_model": "current"},
    "no_service_aware_aggregation": {"federated.aggregation": "semantic"},
    "no_semantic_aware_aggregation": {"federated.aggregation": "service"},
    "no_staleness_correction": {"federated.kappa": 0.0, "federated.max_staleness": 1000000},
    "no_async_communication": {"federated.mode": "sync"},
    "no_update_clipping": {"federated.clip_norm": None},
    "no_assignment_continuity": {"allocation.weights.switch": 0.0},
    "certainty_equivalent_backlog": {"allocation.backlog_model": "ce"},
    "centralized_actor_training": {"federated.enabled": False},
}


def build_cfg(base_cfg, method: str | None = None, overrides: dict | None = None):
    cfg = copy.deepcopy(base_cfg)
    ov = {}
    if method:
        ov.update(METHODS[method]["overrides"])
    ov.update(overrides or {})
    for k, v in ov.items():
        set_by_path(cfg, k, v)
    validate_config(cfg)
    return cfg


def make_trainer(cfg, device, method: str | None = None, run_dir=None, seed: int = 0, logger=None):
    kind = METHODS[method]["trainer"] if method else "matd3"
    if kind == "mappo":
        from .mappo import MAPPOTrainer
        return MAPPOTrainer(cfg, device, run_dir, seed, logger)
    from ..training.trainer import Trainer
    return Trainer(cfg, device, run_dir, seed, logger)
