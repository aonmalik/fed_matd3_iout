"""Protocol 2.2: explicit finite-horizon termination -- training cutoff, bounded communication drain, evaluation start."""
import numpy as np
import pytest

from fed_fixtures import Harness, link, perturb
from fed_matd3_iout.federated.update_packet import UpdateStatus
from fed_matd3_iout.training.trainer import Trainer

S = dict(coll_mbit=5.0, owned_loss_mbit=2.0, U_mature=3.0, V_mature=4.0)


class _StubEnv:
    """Environment surface used by Trainer.finalize: keeps the harness' scripted links, steps the protocol clock only."""

    def __init__(self, h):
        self.h, self.M, self.dt = h.env, h.n, 1.0
        self.links = h.env.links

    def __getattr__(self, k):
        return getattr(self.h, k)

    def _obs_all(self):
        return np.zeros((self.M, 4))

    def step(self, act):
        return self._obs_all(), None, False, False, {}


class _Stub(Trainer):
    def __init__(self, h, drain_s):
        self.proto, self.env, self.server = h.proto, _StubEnv(h), h.server
        self.cfg = {"federated": {"final_drain_s": drain_s}}
        self.global_time = float(h.t)
        self.log = type("L", (), {"log": staticmethod(lambda *a, **k: None)})()
        self.agents = h.clients
        self.termination = None

    def local_policies(self):
        return [lambda o: np.zeros(2) for _ in range(self.env.M)]


def _fin(h, drain_s):
    tr = _Stub(h, drain_s)
    return tr, tr.finalize()


def _prime(h, i=0, d=0.1):
    c = h.clients[i]
    perturb(c, d)
    c.steps_since = c.trans_since = 5
    h.env.inject(i, **S)
    return c


def test_census_counts_updates_in_every_lifecycle_state(cfg):
    h = Harness(cfg, 4, "async", agg_buffer_size=3)
    h.hold(False)
    for i in range(4):
        _prime(h, i, 0.1 + 0.01 * i)
    h.env.links[0] = link(rate=h.rate_for_steps(50))               # c0: slow -> UPLOADING
    h.env.links[1] = link(rate=h.rate_for_steps(2), prop=30.0)     # c1: long propagation -> IN_FLIGHT
    h.env.links[2] = link(rate=h.rate_for_steps(2))                # c2: arrives -> BUFFERED (buffer needs 3)
    h.env.links[3] = link(up=False)                                # c3: nothing created
    h.run(8)
    r = h.proto.pending_report()
    assert r["uploading"] == 1 and r["in_flight"] == 1 and r["buffered"] == 1 and r["unresolved_total"] == 3


def test_cutoff_only_policy_runs_no_drain_and_reports_pending(cfg):
    h = Harness(cfg, 2, "async", agg_buffer_size=3)
    h.hold(False)
    _prime(h, 0)
    h.env.links[0] = link(rate=h.rate_for_steps(50))
    h.run(3)
    t0, v0 = h.t, h.server.version
    tr, term = _fin(h, 0.0)
    assert term["drain_steps"] == 0 and term["drain_s"] == 0.0
    assert term["pending_at_cutoff"]["uploading"] == 1 and term["pending_at_evaluation_start"]["uploading"] == 1
    assert term["t_training_cutoff_s"] == term["t_evaluation_start_s"] == float(t0)
    assert h.server.version == v0 and term["aggregated_during_drain"] == 0


def test_drain_resolves_an_in_flight_update_and_creates_no_new_update(cfg):
    h = Harness(cfg, 2, "async", agg_buffer_size=1)
    h.hold(False)
    c = _prime(h, 0)
    h.env.links[0] = link(rate=h.rate_for_steps(2), prop=6.0)
    h.run_until(lambda: h.proto.n_transmitted == 1)                # last bit left; still propagating
    assert h.proto.pending_report()["in_flight"] == 1
    created = h.proto.n_created
    c.steps_since = c.trans_since = 10_000                          # a NEW update would be ready if drain allowed it
    tr, term = _fin(h, 120.0)
    assert h.proto.n_created == created                             # no new update during the drain
    assert term["aggregated_during_drain"] == 1
    assert term["pending_at_evaluation_start"]["unresolved_total"] == 0
    assert term["drain_s"] < 120.0 and not term["drain_exhausted_budget"]
    assert h.proto.accept_new is True                               # drain restores the flag
    assert term["statistics_accounting"]["balanced"]


def test_drain_is_bounded_and_does_not_force_unresolved_updates(cfg):
    h = Harness(cfg, 2, "async", agg_buffer_size=3)                 # buffer will never fill with one client
    h.hold(False)
    _prime(h, 0)
    h.env.links[0] = link(rate=h.rate_for_steps(2))
    h.run(8)
    assert h.proto.pending_report()["buffered"] == 1
    tr, term = _fin(h, 25.0)
    assert term["drain_s"] == 25.0 and term["drain_exhausted_budget"]
    assert term["pending_at_evaluation_start"]["buffered"] == 1     # reported, not forced to expire/aggregate
    assert term["statistics_accounting"]["balanced"] and term["statistics_accounting"]["unresolved"] == 1


def test_drain_lets_the_timeout_path_settle_statistics_exactly_once(cfg):
    h = Harness(cfg, 2, "async", agg_buffer_size=3, max_buffer_age_s=15.0, agg_timeout_s=5.0)
    h.hold(False)
    _prime(h, 0)
    h.env.links[0] = link(rate=h.rate_for_steps(2))
    h.run(6)
    n_before = sum(h.proto.n_status.values())
    tr, term = _fin(h, 200.0)
    acc = term["statistics_accounting"]
    assert acc["balanced"] and sum(h.proto.n_status.values()) >= n_before
    assert sum(acc["stats_disposition"].values()) == sum(h.proto.n_status.values())   # one disposition per settled update


def test_negative_drain_is_rejected_by_config_validation(cfg):
    from fed_matd3_iout.utils.config import ConfigError, validate_config
    cfg["federated"]["final_drain_s"] = -1.0
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_real_trainer_finalize_reports_cutoff_drain_and_evaluation_start(smoke_cfg):
    import torch
    smoke_cfg["env"]["episode_steps"] = 60
    tr = Trainer(smoke_cfg, torch.device("cpu"), train_seed=0)
    tr.train(2)
    term = tr.finalize(drain_s=40.0)                               # real environment stepped past its episode horizon
    assert term["t_evaluation_start_s"] >= term["t_training_cutoff_s"] and term["drain_s"] <= 40.0
    assert term["statistics_accounting"]["balanced"]
    assert set(term["pending_at_cutoff"]) >= {"uploading", "in_flight", "buffered", "unresolved_total"}
    assert tr.proto.accept_new is True
