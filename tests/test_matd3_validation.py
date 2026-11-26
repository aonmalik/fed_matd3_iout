"""Final validation item 4: MATD3 core, actor-only federation, aggregation integrity -- plus mutations proving the checks can fail."""
import pytest
import torch

import fed_matd3_iout.agents.matd3_agent as MA
from fed_matd3_iout.federated import aggregation as AG
from fed_matd3_iout.federated.federated_client import FederatedClient
from fed_matd3_iout.validation import matd3_checks as X


def ok(r):
    return all(v for v in r.values() if isinstance(v, bool))


@pytest.mark.parametrize("fn", [X.check_twin_critic_target, X.check_target_policy_smoothing, X.check_delayed_actor_and_polyak,
                                X.check_critic_loss_and_gradient_paths, X.check_actor_only_federation,
                                X.check_install_is_inplace_and_blend, X.check_aggregation_integrity, X.check_staleness_semantics])
def test_check_passes(cfg, fn):
    r = fn(cfg)
    assert ok(r), {k: v for k, v in r.items() if v is False}


def test_parameters_finite_after_federated_training(cfg):
    r = X.check_parameters_finite_after_training(cfg, n_episodes=4)
    assert ok(r) and r["aggregation_rounds"] > 0 and r["gradient_steps"] > 0


# ---------------------------------------------------------------- mutations (each re-introduces a classic MATD3 / federation bug)
def test_mutation_mean_instead_of_min_detected(cfg, monkeypatch):
    def bad(self, batch, target_actors, noise=None):
        a2 = self.target_joint_action(batch["o2"], target_actors, noise)
        q1, q2 = self.critic_t(self.critic_input(batch["s2"], batch["o2"][:, self.i, :], a2))
        return batch["r"][:, self.i] + float(self.cfg["gamma"]) * (1 - batch["d"]) * torch.max(q1, q2)
    monkeypatch.setattr(MA.MATD3Agent, "compute_target", torch.no_grad()(bad))
    assert not ok(X.check_twin_critic_target(cfg))


def test_mutation_unclipped_target_noise_detected(cfg, monkeypatch):
    def bad(self, o2, target_actors, noise=None):
        acts = [torch.clamp(target_actors[k](o2[:, k, :]) + (noise[:, k, :] if noise is not None else 0), -1.0, 1.0)
                for k in range(self.M)]                                  # noise NOT clipped to +-c
        return torch.stack(acts, 1)
    monkeypatch.setattr(MA.MATD3Agent, "target_joint_action", torch.no_grad()(bad))
    assert not ok(X.check_target_policy_smoothing(cfg))


def test_mutation_no_policy_delay_detected(cfg, monkeypatch):
    cfg2 = cfg
    monkeypatch.setitem(cfg2.matd3, "policy_delay", 2)
    monkeypatch.setattr(MA, "polyak_update", lambda t, s, tau: None)       # targets never move
    assert not ok(X.check_delayed_actor_and_polyak(cfg2))


def test_mutation_federation_touches_critic_detected(cfg, monkeypatch):
    orig = FederatedClient.install_global

    def bad(self, theta, version, blend):
        orig(self, theta, version, blend)
        with torch.no_grad():
            for p in self.agent.critic.parameters():                      # critics wrongly overwritten
                p.add_(0.5)
    monkeypatch.setattr(FederatedClient, "install_global", bad)
    assert not ok(X.check_actor_only_federation(cfg))


def test_mutation_no_clipping_detected(cfg, monkeypatch):
    monkeypatch.setattr(AG, "l2_clip", lambda d, c: (d, 1.0))
    assert not ok(X.check_aggregation_integrity(cfg))
