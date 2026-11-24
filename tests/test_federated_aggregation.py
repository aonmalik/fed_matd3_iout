import numpy as np
import pytest
import torch
from fed_matd3_iout.federated.aggregation import aggregate, l2_clip, normalise_with_cap, raw_weights
from fed_matd3_iout.federated.federated_client import FederatedClient
from fed_matd3_iout.federated.relevance import relevance, rho_sem, rho_svc
from fed_matd3_iout.federated.update_packet import UpdatePacket, UpdateStatus
from fed_matd3_iout.federated.usv_server import USVServer
from fed_matd3_iout.agents.actor import flatten_params
from fed_matd3_iout.agents.matd3_agent import MATD3Agent
from fed_matd3_iout.agents.replay_buffer import ClientReplay, JointStore

P = 12


def pkt(cid, base, delta, n=100.0, stats=None, uid=1):
    d = torch.as_tensor(delta, dtype=torch.float32)
    p = UpdatePacket(cid, uid, base, d, d.clone(), n, 10, 10, float(d.norm()), stats or {})
    p.status = UpdateStatus.IN_FLIGHT          # the server only ever receives updates whose last bit has left the AUV
    return p


def srv(cfg, **kw):
    for k, v in kw.items(): cfg.federated[k] = v
    return USVServer(torch.zeros(P), cfg.federated, 3, True)


def test_weight_formula_and_normalisation():
    z = raw_weights([100, 200], [1.0, np.exp(-0.3)], [1.0, 0.5], alpha=1.0, eta=2.0, eps_m=0.1)
    exp0 = 100 * 1 * (0.1 + 0.9 * 1.0) ** 2
    exp1 = 200 * np.exp(-0.3) * (0.1 + 0.9 * 0.5) ** 2
    assert z == pytest.approx([exp0, exp1])
    w = normalise_with_cap(z, 1.0)
    assert w.sum() == pytest.approx(1.0) and w == pytest.approx(z / z.sum())


def test_zero_relevance_still_positive_weight():
    assert raw_weights([10], [1.0], [0.0], 1.0, 1.0, 0.1)[0] > 0                # eps_M prevents zero weight


def test_weight_cap_and_renormalisation():
    w = normalise_with_cap(np.array([100.0, 1.0, 1.0]), cap=0.6)
    assert w.sum() == pytest.approx(1.0) and w.max() <= 0.6 + 1e-9 and w[1] == pytest.approx(w[2])
    assert normalise_with_cap(np.array([1.0, 1.0]), cap=0.1).tolist() == [0.5, 0.5]   # infeasible cap -> uniform


def test_l2_clip():
    d = torch.tensor([3.0, 4.0])
    c, s = l2_clip(d, 1.0)
    assert float(c.norm()) == pytest.approx(1.0) and s == pytest.approx(0.2)
    assert torch.equal(l2_clip(d, 10.0)[0], d) and torch.equal(l2_clip(d, None)[0], d)
    assert torch.equal(l2_clip(torch.zeros(3), 1.0)[0], torch.zeros(3))


def test_aggregate_formula_and_empty_and_mismatch():
    th = torch.ones(4)
    out, sc = aggregate(th, [torch.tensor([1., 0, 0, 0]), torch.tensor([0., 2, 0, 0])], np.array([0.25, 0.75]), 0.5, None)
    assert torch.allclose(out, th + 0.5 * torch.tensor([0.25, 1.5, 0, 0]))
    assert torch.equal(aggregate(th, [], np.array([]), 1.0, None)[0], th)           # empty set -> unchanged
    with pytest.raises(ValueError):
        aggregate(th, [torch.ones(3)], np.array([1.0]), 1.0, None)


def test_no_aggregation_before_receipt(cfg):
    s = srv(cfg, agg_buffer_size=1)
    assert s.maybe_aggregate(100.0) is None and s.version == 0 and s.n_received == 0
    s.receive(pkt(0, 0, np.ones(P)), 5.0)
    rec = s.maybe_aggregate(5.0)
    assert rec is not None and s.version == 1
    assert float(torch.ones(P).norm()) < cfg.federated.clip_norm            # not clipped, single client => w = 1
    assert torch.allclose(s.theta, torch.ones(P))                           # theta^{r+1} = theta^r + eta_g * 1 * Delta


def test_failed_updates_never_enter_server(cfg):
    s = srv(cfg, agg_buffer_size=1)
    assert s.buffer == []                                                         # a failed upload never calls receive()
    assert s.maybe_aggregate(1e6) is None and s.version == 0


def test_stale_update_rejected(cfg):
    s = srv(cfg, agg_buffer_size=1, max_staleness=2)
    s.version = 10
    assert not s.receive(pkt(0, 7, np.ones(P)), 1.0) and s.n_rejected == 1 and s.buffer == []
    assert s.receive(pkt(0, 8, np.ones(P), uid=2), 1.0)                                  # staleness 2 accepted


def test_shape_mismatch_rejected(cfg):
    s = srv(cfg)
    assert not s.receive(pkt(0, 0, np.ones(P + 1)), 1.0)
    assert s.events[-1]["reason"] == "shape_mismatch"


def test_stale_at_aggregation_is_dropped(cfg):
    s = srv(cfg, agg_buffer_size=2, max_staleness=1, agg_timeout_s=1e9)
    s.receive(pkt(0, 0, np.ones(P)), 1.0)
    s.version = 5                                                                 # server advanced while buffered
    s.receive(pkt(1, 5, np.ones(P)), 2.0)
    rec = s.maybe_aggregate(3.0)
    assert rec["clients"] == [1] and rec["rejected"] == [(0, 5)]


def test_semantic_service_weights_are_logged_and_not_fedavg(cfg):
    s = srv(cfg, aggregation="s3a", agg_buffer_size=2, clip_norm=None, max_client_weight=1.0)
    good = dict(U_mature=90.0, V_mature=100.0, coll_mbit=9.0, owned_loss_mbit=1.0)
    bad = dict(U_mature=10.0, V_mature=100.0, coll_mbit=1.0, owned_loss_mbit=9.0)
    s.receive(pkt(0, 0, np.ones(P), 100, good), 1.0); s.receive(pkt(1, 0, -np.ones(P), 100, bad), 1.0)
    rec = s.maybe_aggregate(1.0)
    assert rec["w"][0] > rec["w"][1] and sum(rec["w"]) == pytest.approx(1.0)       # same n_i, different relevance
    assert rec["relevance"][0] > rec["relevance"][1] and "Z" in rec and "stale_factor" in rec
    f = srv(cfg, aggregation="fedavg", agg_buffer_size=2, clip_norm=None, max_client_weight=1.0)
    f.receive(pkt(0, 0, np.ones(P), 100, good), 1.0); f.receive(pkt(1, 0, -np.ones(P), 100, bad), 1.0)
    assert f.maybe_aggregate(1.0)["w"] == pytest.approx([0.5, 0.5])                # FedAvg ignores relevance


def test_aggregation_deterministic(cfg):
    outs = []
    for _ in range(2):
        s = srv(cfg, aggregation="s3a", agg_buffer_size=2)
        r = np.random.default_rng(0)
        s.receive(pkt(0, 0, r.normal(size=P), 80, dict(U_mature=5., V_mature=10., coll_mbit=1., owned_loss_mbit=1.)), 1.0)
        s.receive(pkt(1, 0, r.normal(size=P), 120, dict(U_mature=2., V_mature=10., coll_mbit=2., owned_loss_mbit=1.)), 1.0)
        s.maybe_aggregate(1.0); outs.append(s.theta.clone())
    assert torch.equal(outs[0], outs[1])


def test_clipping_applied_in_server(cfg):
    s = srv(cfg, aggregation="fedavg", agg_buffer_size=1, clip_norm=1.0, max_client_weight=1.0)
    s.receive(pkt(0, 0, np.full(P, 10.0)), 1.0); rec = s.maybe_aggregate(1.0)
    assert float(s.theta.norm()) == pytest.approx(1.0, rel=1e-5) and rec["clip_scale"][0] < 1


def test_relevance_measures():
    st = dict(U_mature=50, V_mature=100, coll_mbit=3, owned_loss_mbit=1, units_del_mature=40, units_coll_mature=80)
    assert rho_sem(st) == pytest.approx(0.5) and rho_svc(st) == pytest.approx(0.75)
    assert relevance(st, "s3a", 0.5) == pytest.approx(0.625) and relevance(st, "fedavg", 0.5) == 1.0
    assert rho_sem(st, use_semantics=False) == pytest.approx(0.5)
    with pytest.raises(ValueError):                                                 # no clipping: inconsistent ledger is an error
        rho_sem(dict(U_mature=500, V_mature=100))
    assert rho_sem({}) is None and rho_svc({}) is None                              # no evidence => undefined, not zero
    assert 0 <= relevance(st, "s3a", 0.3) <= 1


def test_client_update_construction_and_versions(cfg):
    cfg.matd3.hidden = 8
    ag = MATD3Agent(0, 2, 4, 2, 6, cfg.matd3, "cpu")
    st = JointStore(10, 6, 2, 4, 2)
    c = FederatedClient(0, ag, ClientReplay(st, 10, np.random.default_rng(0)), cfg.federated, 16, 1.0)
    base = flatten_params(ag.actor)
    with torch.no_grad():
        for p in ag.actor.parameters(): p.add_(0.1)
    c.steps_since, c.trans_since = 100, 600
    u = c.make_update(3.0, {"U_mature": 1.0})
    assert torch.allclose(u.delta, flatten_params(ag.actor) - base, atol=1e-6)        # Delta = theta_local - theta_base
    assert u.base_version == 0 and u.n_samples == 100 * 16 and u.update_id == 1 and u.norm == pytest.approx(float(u.delta.norm()))
    assert u.size_bytes == u.delta.numel() * cfg.federated.model_bytes_per_param
    c.install_global(base + 1.0, 4, 1.0)
    assert c.base_version == 4 and torch.allclose(flatten_params(ag.actor), base + 1.0) and torch.allclose(flatten_params(ag.actor_t), base + 1.0)
    assert c.steps_since == 0 and torch.allclose(c.reference, base + 1.0)
