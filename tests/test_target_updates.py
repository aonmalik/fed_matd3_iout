import copy
import pytest
import torch
from fed_matd3_iout.agents.actor import flatten_params
from fed_matd3_iout.agents.matd3_agent import MATD3Agent, polyak_update
from test_matd3 import M, OD, AD, SD, batch, mk


def test_polyak_averaging_exact():
    a, b = torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)
    t = copy.deepcopy(a)
    expect = [0.25 * sp.detach() + 0.75 * tp.detach() for sp, tp in zip(b.parameters(), t.parameters())]
    polyak_update(t, b, 0.25)
    for e, p in zip(expect, t.parameters()):
        assert torch.allclose(e, p)


def test_delayed_actor_update_and_target_ordering(cfg):
    agents = mk(cfg, policy_delay=2, tau=0.5)
    ag = agents[0]
    tg = [a.actor_t for a in agents]
    a0, at0, ct0 = flatten_params(ag.actor), flatten_params(ag.actor_t), torch.cat([p.detach().flatten() for p in ag.critic_t.parameters()])
    c0 = torch.cat([p.detach().flatten() for p in ag.critic.parameters()])
    ag.update(batch(1), tg)                                                       # 1st critic update: NO actor / target update
    assert torch.equal(flatten_params(ag.actor), a0) and torch.equal(flatten_params(ag.actor_t), at0)
    assert torch.equal(torch.cat([p.detach().flatten() for p in ag.critic_t.parameters()]), ct0)
    assert not torch.equal(torch.cat([p.detach().flatten() for p in ag.critic.parameters()]), c0)   # critic did move
    assert ag.actor_updates == 0
    ag.update(batch(2), tg)                                                       # 2nd: actor + Polyak
    assert not torch.equal(flatten_params(ag.actor), a0) and ag.actor_updates == 1
    assert not torch.equal(flatten_params(ag.actor_t), at0)
    assert ag.critic_updates == 2


def test_critic_loss_decreases_on_fixed_batch(cfg):
    agents = mk(cfg, policy_delay=2)
    ag, tg, b = agents[0], [a.actor_t for a in agents], batch(3)
    first = ag.update(b, tg)["critic_loss"]
    for _ in range(80):
        last = ag.update(b, tg)["critic_loss"]
    assert last < first


def test_actor_gradient_flows_only_through_own_action(cfg):
    agents = mk(cfg, policy_delay=1)
    ag, tg = agents[1], [a.actor_t for a in agents]
    peers_before = [flatten_params(a.actor) for a in agents]
    ag.update(batch(4), tg)
    after = [flatten_params(a.actor) for a in agents]
    assert not torch.equal(after[1], peers_before[1])                              # own actor changed
    assert torch.equal(after[0], peers_before[0]) and torch.equal(after[2], peers_before[2])


def test_maddpg_configuration_is_distinct(cfg):
    from fed_matd3_iout.baselines.registry import build_cfg
    c = build_cfg(cfg, "maddpg")
    assert c.matd3.twin is False and c.matd3.policy_delay == 1 and c.matd3.target_noise == 0.0
    ind = build_cfg(cfg, "independent_td3")
    assert ind.matd3.critic_mode == "independent"
