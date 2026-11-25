"""Protocol 2.2: MAPPO tanh-squashed policy -- the PPO likelihood must describe the EXECUTED action."""
import math

import numpy as np
import pytest
import torch

from fed_matd3_iout.baselines.mappo import GaussianActor, MAPPOTrainer, tanh_log_det


def _actor(init_log_std=-0.5, seed=0):
    torch.manual_seed(seed)
    return GaussianActor(7, 2, 16, init_log_std)


def test_actions_always_inside_closed_unit_interval_even_for_large_noise():
    a = _actor(init_log_std=3.0)                       # std ~ 20: the old Gaussian would have been clipped most of the time
    act, u, lp = a.sample(torch.randn(4096, 7))
    assert act.abs().max() <= 1.0 and torch.isfinite(act).all()


def test_stored_and_recomputed_log_prob_agree():
    a, o = _actor(), torch.randn(256, 7)
    act, u, lp = a.sample(o)
    assert torch.allclose(lp, a.log_prob_latent(o, u), atol=1e-6)


def test_log_prob_matches_change_of_variables_numerically():
    """Density of a = tanh(u) integrates to one: sum p(a) da over a fine grid ~ 1 for a 1-D case (independent check)."""
    a = GaussianActor(3, 1, 8, -0.3)
    o = torch.zeros(1, 3)
    grid = torch.linspace(-0.999999, 0.999999, 400001)
    u = torch.atanh(grid).unsqueeze(-1)
    lp = (a.dist(o.expand(len(grid), -1)).log_prob(u) - tanh_log_det(u)).sum(-1)
    integral = torch.trapz(lp.exp(), grid)
    assert integral.item() == pytest.approx(1.0, abs=2e-3)


def test_jacobian_is_finite_and_exact_in_the_saturated_regime():
    u = torch.tensor([-30.0, -10.0, 0.0, 10.0, 30.0], requires_grad=True)
    ld = tanh_log_det(u)
    assert torch.isfinite(ld).all()
    ld.sum().backward()
    assert torch.isfinite(u.grad).all()
    ref = torch.log1p(-torch.tanh(torch.tensor([0.5, 1.0])) ** 2)
    assert torch.allclose(tanh_log_det(torch.tensor([0.5, 1.0])), ref, atol=1e-6)


def test_ratios_are_one_before_any_parameter_change_and_gradients_finite():
    a, o = _actor(), torch.randn(128, 7)
    act, u, lp_old = a.sample(o)
    lp_new = a.log_prob_latent(o, u)
    ratio = (lp_new - lp_old).exp()
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5)
    lp_new.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters())


def test_policy_gradient_update_moves_probability_toward_rewarded_action(smoke_cfg):
    """Controlled one-step problem: reward = -(a_0 - 0.6)^2.  The shared actor + value update must learn a_0 -> ~0.6."""
    cfg = smoke_cfg
    cfg["matd3"]["ppo"]["epochs"] = 4
    tr = MAPPOTrainer(cfg, torch.device("cpu"), train_seed=0)
    M, sd, od = tr.env.M, tr.env.state_dim, tr.env.obs_dim
    g = torch.Generator().manual_seed(1)
    obs = torch.zeros(1, od)

    def mean_a0():
        with torch.no_grad():
            return float(tr.actor.mean_action(obs)[0, 0])

    start = mean_a0()
    for it in range(120):
        T = 256
        O = torch.randn(T, M, od, generator=g) * 0.0                 # context-free bandit
        S = torch.zeros(T, sd)
        buf = {k: [] for k in ("s", "s2", "o", "a", "u", "lp", "r", "last")}
        with torch.no_grad():
            act, u, lp = tr.actor.sample(O.reshape(-1, od))
        act, u, lp = act.reshape(T, M, -1), u.reshape(T, M, -1), lp.reshape(T, M)
        r = -(act[..., 0] - 0.6) ** 2
        buf["s"], buf["s2"], buf["o"] = list(S.numpy()), list(S.numpy()), list(O.numpy())
        buf["a"], buf["u"], buf["lp"] = list(act.numpy()), list(u.numpy()), list(lp.numpy())
        buf["r"], buf["last"] = list(r.numpy()), [1.0] * T
        info = tr._update(buf)
        assert math.isfinite(info["policy_loss"]) and math.isfinite(info["value_loss"])
    end = mean_a0()
    assert abs(end - 0.6) < abs(start - 0.6) and abs(end - 0.6) < 0.25, (start, end)


def test_real_rollout_sends_bounded_finite_actions_to_the_environment(smoke_cfg, monkeypatch):
    """The rollout passes the tanh output straight to the environment (no clip step); all executed actions are in [-1, 1]."""
    cfg = smoke_cfg
    cfg["experiment"]["train_episodes"] = 1
    cfg["env"]["episode_steps"] = 12
    cfg["matd3"]["ppo"]["rollout_steps"] = 10_000                    # no update in this test
    tr = MAPPOTrainer(cfg, torch.device("cpu"), train_seed=0)
    seen = []
    real = tr.env.step
    monkeypatch.setattr(tr.env, "step", lambda a: (seen.append(np.array(a)) or real(a)))
    tr.train(1)
    assert seen and all(np.abs(a).max() <= 1.0 for a in seen)
    assert all(np.isfinite(a).all() for a in seen)


# ---------------------------------------------------------------- entropy regulariser (matd3.ppo.entropy)
def test_squashed_entropy_estimator_matches_numerical_entropy_of_the_density():
    """1-D: -integral p(a) log p(a) da on a fine grid  vs  mean of the reparameterised estimator."""
    a = GaussianActor(3, 1, 8, -0.3)
    o = torch.zeros(1, 3)
    grid = torch.linspace(-0.999999, 0.999999, 400001)
    u = torch.atanh(grid).unsqueeze(-1)
    lp = (a.dist(o.expand(len(grid), -1)).log_prob(u) - tanh_log_det(u)).sum(-1)
    num = torch.trapz(-lp.exp() * lp, grid).item()
    torch.manual_seed(0)
    est = a.squashed_entropy(o.expand(200_000, -1)).mean().item()
    assert est == pytest.approx(num, abs=0.02)


def test_squashed_entropy_never_exceeds_latent_entropy_and_differs_from_it():
    a = _actor(init_log_std=0.5)
    o = torch.randn(4096, 7)
    torch.manual_seed(1)
    sq = a.squashed_entropy(o).mean()
    lat = a.dist(o).entropy().sum(-1).mean()
    assert sq < lat                                                       # E[log(1 - tanh^2)] < 0


def test_squashed_entropy_gradients_are_finite_and_reach_mean_and_log_std():
    a, o = _actor(init_log_std=2.0), torch.randn(256, 7)                  # wide policy: strong saturation
    a.squashed_entropy(o).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters())
    assert a.log_std.grad.abs().sum() > 0 and a.mu[0].weight.grad.abs().sum() > 0


def test_entropy_option_is_validated_and_both_modes_update(smoke_cfg):
    cfg = smoke_cfg
    cfg["matd3"]["ppo"]["entropy"] = "bogus"
    with pytest.raises(ValueError, match="entropy"):
        MAPPOTrainer(cfg, torch.device("cpu"), train_seed=0)
    for mode in ("squashed", "latent"):
        cfg["matd3"]["ppo"]["entropy"] = mode
        tr = MAPPOTrainer(cfg, torch.device("cpu"), train_seed=0)
        assert tr.entropy_mode == mode
        M, sd, od, T = tr.env.M, tr.env.state_dim, tr.env.obs_dim, 64
        O = torch.randn(T, M, od)
        with torch.no_grad():
            act, u, lp = tr.actor.sample(O.reshape(-1, od))
        buf = {"s": list(np.zeros((T, sd), dtype=np.float32)), "s2": list(np.zeros((T, sd), dtype=np.float32)), "o": list(O.numpy()),
               "a": list(act.reshape(T, M, -1).numpy()), "u": list(u.reshape(T, M, -1).numpy()), "lp": list(lp.reshape(T, M).numpy()),
               "r": list(np.ones((T, M), dtype=np.float32)), "last": [0.0] * (T - 1) + [1.0]}
        info = tr._update(buf)
        assert math.isfinite(info["policy_loss"]) and math.isfinite(info["value_loss"])
