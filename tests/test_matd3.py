import copy
import numpy as np
import pytest
import torch
from fed_matd3_iout.agents.actor import Actor, flatten_params, load_flat_params
from fed_matd3_iout.agents.matd3_agent import MATD3Agent

M, OD, AD, SD, B = 3, 5, 2, 9, 16


def mk(cfg, **kw):
    for k, v in kw.items():
        cfg.matd3[k] = v
    cfg.matd3.hidden = 16
    torch.manual_seed(0)
    return [MATD3Agent(i, M, OD, AD, SD, cfg.matd3, "cpu") for i in range(M)]


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(s=torch.randn(B, SD, generator=g), s2=torch.randn(B, SD, generator=g), o=torch.randn(B, M, OD, generator=g),
                o2=torch.randn(B, M, OD, generator=g), a=torch.rand(B, M, AD, generator=g) * 2 - 1,
                r=torch.randn(B, M, generator=g), d=torch.zeros(B))


class Const(torch.nn.Module):
    def __init__(self, v): super().__init__(); self.v = v
    def forward(self, x): return torch.full((x.shape[0],) + tuple(np.atleast_1d(self.v).shape), 0.0) + torch.as_tensor(self.v, dtype=torch.float32)


class ConstQ(torch.nn.Module):
    def __init__(self, q1, q2): super().__init__(); self.q1, self.q2 = q1, q2
    def forward(self, x): return torch.full((x.shape[0],), self.q1), torch.full((x.shape[0],), self.q2)


def test_actor_output_bounds(cfg):
    a = Actor(OD, AD, 16)
    out = a(torch.randn(1000, OD) * 100)
    assert out.shape == (1000, AD) and out.abs().max() <= 1.0


def test_critic_dimensions(cfg):
    ag = mk(cfg)[0]
    x = ag.critic_input(batch()["s"], batch()["o"][:, 0], batch()["a"])
    assert x.shape == (B, SD + M * AD)                                       # Q(s, a_1..a_M)
    q1, q2 = ag.critic(x)
    assert q1.shape == q2.shape == (B,)
    ind = mk(cfg, critic_mode="independent")[0]
    assert ind.critic_input(batch()["s"], batch()["o"][:, 0], batch()["a"]).shape == (B, OD + AD)


def test_joint_action_uses_all_target_actors(cfg):
    ag = mk(cfg, target_noise=0.0)[0]
    consts = [0.1, -0.4, 0.7]
    acts = [Const(np.array([c, -c], dtype=np.float32)) for c in consts]
    a2 = ag.target_joint_action(batch()["o2"], acts)
    assert a2.shape == (B, M, AD)
    for k, c in enumerate(consts):
        assert torch.allclose(a2[:, k, 0], torch.full((B,), c)) and torch.allclose(a2[:, k, 1], torch.full((B,), -c))


def test_target_policy_smoothing_clip_and_bounds(cfg):
    ag = mk(cfg, target_noise=0.2, noise_clip=0.5)[0]
    acts = [Const(np.array([0.9, -0.9], dtype=np.float32)) for _ in range(M)]
    noise = torch.full((B, M, AD), 10.0)                                      # huge noise must be clipped to +-0.5
    a2 = ag.target_joint_action(batch()["o2"], acts, noise)
    assert torch.allclose(a2[:, :, 0], torch.ones(B, M))                      # clip(0.9 + 0.5, -1, 1) = 1
    assert torch.allclose(a2[:, :, 1], torch.full((B, M), -0.4))              # -0.9 + 0.5
    assert a2.min() >= -1 and a2.max() <= 1


def test_twin_target_minimum_and_terminal(cfg):
    ag = mk(cfg, gamma=0.9, target_noise=0.0)[0]
    ag.critic_t = ConstQ(5.0, 3.0)
    b = batch(); b["d"] = torch.tensor([0.0] * 8 + [1.0] * 8)
    y = ag.compute_target(b, [a.actor_t for a in mk(cfg)])
    r = b["r"][:, 0]
    assert torch.allclose(y[:8], r[:8] + 0.9 * 3.0)                           # min(5,3)
    assert torch.allclose(y[8:], r[8:])                                       # terminal: no bootstrap
    ag.critic_t = ConstQ(2.0, 7.0)
    assert torch.allclose(ag.compute_target(b, [a.actor_t for a in mk(cfg)])[:8], r[:8] + 0.9 * 2.0)


def test_single_critic_mode_uses_q1(cfg):
    ag = mk(cfg, twin=False, gamma=0.5)[0]
    assert ag.critic.q2 is None
    ag.critic_t = ConstQ(4.0, 100.0)
    b = batch()
    y = ag.compute_target(b, [a.actor_t for a in mk(cfg, twin=False)])
    assert torch.allclose(y, b["r"][:, 0] + 0.5 * 4.0)
