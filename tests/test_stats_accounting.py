"""Regression tests 9-12: window-statistics accounting and episode isolation (audit item 3)."""
import numpy as np
import pytest

from fed_fixtures import build, link, perturb, run
from fed_matd3_iout.environment.multi_auv_iout_env import MultiAUVIoUTEnv
from fed_matd3_iout.federated.update_packet import UpdateStatus

STATS = {"coll_mbit": 5.0, "owned_loss_mbit": 2.0, "U_mature": 3.0, "V_mature": 4.0}


def _start(cfg, **fed):
    """One client, one update carrying known statistics; further updates are prevented (steps_since reset)."""
    cs, srv, proto, env = build(cfg, 1, **fed)
    c = cs[0]
    perturb(c)
    c.steps_since, c.trans_since = 5, 5
    for k, v in STATS.items():
        env.win[k][0] = v
    return c, srv, proto, env


def _first_update_created(c, proto, env):
    run(proto, env, 0, 1)
    assert proto.n_created == 1
    c.steps_since = 0                                                   # no second update during the test
    assert all(env.win[k][0] == 0.0 for k in STATS)                     # the stats now belong to the update


def _window(env):
    return {k: float(env.win[k][0]) for k in STATS}


# ----------------------------------------------------------------------------- 9
def test_rejected_update_stats_accounted_exactly_once(cfg):
    c, srv, proto, env = _start(cfg, max_staleness=0)
    srv.version = 4                                                     # forces stale_at_receipt
    settled, orig = [], proto._settle
    proto._settle = lambda e, p, t: (settled.append(p), orig(e, p, t))[1]
    _first_update_created(c, proto, env)
    run(proto, env, 1, 30)
    assert proto.n_status["rejected"] == 1 and proto.stats_disposition["returned"] == 1
    assert _window(env) == STATS                                        # handed back once, nothing lost / duplicated
    row = proto.update_log[0]
    assert row["reason"] == "stale_at_receipt" and row["disposition"] == "returned"
    assert len(settled) == 1
    proto._settle(env, settled[0], 99.0)                                # a second settlement attempt is a no-op
    assert _window(env) == STATS and len(proto.update_log) == 1 and proto.n_status["rejected"] == 1


# ----------------------------------------------------------------------------- 10
def test_expired_update_stats_accounted_exactly_once_upload_timeout(cfg):
    c, srv, proto, env = _start(cfg, upload_timeout_s=5.0)
    env.links[0] = link(up=True, success=False)                         # fading: upload can never finish
    _first_update_created(c, proto, env)
    run(proto, env, 1, 30)
    assert proto.n_status["expired"] == 1 and proto.update_log[0]["reason"] == "upload_timeout"
    assert proto.stats_disposition["returned"] == 1 and _window(env) == STATS
    assert c.outstanding is None and srv.n_received == 0                # the server never saw it
    run(proto, env, 30, 60)
    assert proto.n_status["expired"] == 1 and _window(env) == STATS     # still exactly once


def test_expired_update_stats_accounted_exactly_once_buffer_expiry(cfg):
    c, srv, proto, env = _start(cfg, agg_buffer_size=99, agg_timeout_s=1e9, max_buffer_age_s=10.0)
    _first_update_created(c, proto, env)
    run(proto, env, 1, 40)
    assert proto.update_log[0]["reason"] == "expired_in_buffer" and proto.n_status["expired"] == 1
    assert srv.n_expired == 1 and srv.n_accepted == 0 and srv.version == 0
    assert proto.stats_disposition["returned"] == 1 and _window(env) == STATS
    assert c.outstanding is None and c.stats["failed"] == 1             # client learns the outcome over the downlink


# ----------------------------------------------------------------------------- 11
def test_successful_update_not_double_counted(cfg):
    c, srv, proto, env = _start(cfg, agg_buffer_size=1)
    _first_update_created(c, proto, env)
    run(proto, env, 1, 30)
    assert proto.n_status["aggregated"] == 1 and proto.stats_disposition["consumed"] == 1
    assert proto.stats_disposition["returned"] == 0
    assert _window(env) == {k: 0.0 for k in STATS}                      # consumed => NOT returned to the window
    assert proto.stats_mass["consumed"]["coll_mbit"] == 5.0 and proto.stats_mass["returned"]["coll_mbit"] == 0.0
    s = proto.summary()
    assert s["accepted_updates"] == s["aggregated_updates"] == 1 and s["failed_updates"] == 0
    assert proto.statistics_accounting()["balanced"]


# ----------------------------------------------------------------------------- 12
def test_old_episode_stats_cannot_contaminate_next_episode(cfg):
    c, srv, proto, env = _start(cfg, agg_buffer_size=99, agg_timeout_s=1e9, max_buffer_age_s=10.0)
    _first_update_created(c, proto, env)
    run(proto, env, 1, 3)
    env.new_episode()                                                   # episode 1 ends while the update is still buffered
    env.win["coll_mbit"][0] = 1.0                                       # fresh data of episode 2
    run(proto, env, 3, 40)                                              # the old update expires in episode 2
    assert proto.update_log[0]["episode"] == 1 and proto.update_log[0]["disposition"] == "discarded_episode"
    assert proto.stats_disposition["discarded_episode"] == 1 and env.stats_refused_episode == 1
    assert env.win["coll_mbit"][0] == 1.0 and env.win["V_mature"][0] == 0.0   # episode-2 window untouched by episode-1 stats


def test_real_env_refuses_other_episode_stats(smoke_cfg):
    env = MultiAUVIoUTEnv(smoke_cfg)
    env.reset(0)
    ep1 = env.episode_id
    env.reset(1)
    assert env.episode_id == ep1 + 1
    before = {k: v.copy() for k, v in env.win.items()}
    assert env.return_window_stats(0, {"coll_mbit": 9.0, "V_mature": 9.0}, ep1) is False
    assert env.stats_refused_episode == 1
    assert all(np.array_equal(before[k], env.win[k]) for k in before)
    assert env.return_window_stats(0, {"coll_mbit": 9.0}, env.episode_id) is True and env.win["coll_mbit"][0] == 9.0
