"""Protocol 2.2 re-audit (phase 7): properties that were not directly covered by an existing test."""
import math

import pytest
import torch

from fed_fixtures import Harness, link, perturb
from fed_matd3_iout.federated.update_packet import ControlMessage, UpdateStatus

S = dict(coll_mbit=5.0, owned_loss_mbit=2.0, U_mature=3.0, V_mature=4.0)


def _one_update(cfg, **fed):
    h = Harness(cfg, 2, "async", **fed)
    h.hold(False)
    c = h.clients[0]
    perturb(c)
    c.steps_since = c.trans_since = 5
    h.env.inject(0, **S)
    h.env.links[0] = link(rate=h.rate_for_steps(2))
    h.run_until(lambda: c.outstanding is not None)
    return h, c, c.outstanding


def test_older_commit_cannot_overwrite_a_newer_installed_global(cfg):
    h, c, p = _one_update(cfg)
    glob = torch.full_like(c.reference, 0.123)
    c.install_global(glob, 5, 1.0)                                     # a newer global arrives while the update is unresolved
    ref_after_install = c.reference.clone()
    msg = ControlMessage("commit", p.uid, 0, p.update_id, 4, "aggregated", 0.0)
    assert c.handle_control(msg, 1.0) == "commit"
    assert torch.equal(c.reference, ref_after_install) and not torch.equal(c.reference, p.snapshot)   # reference stays the global
    assert c.base_version == 5 and c.outstanding is None


def test_unknown_and_duplicate_control_messages_change_nothing(cfg):
    h, c, p = _one_update(cfg)
    before = (c.reference.clone(), c.steps_since, c.trans_since, c.outstanding.uid, c.base_version)
    assert c.handle_control(ControlMessage("commit", "c0-u999", 0, 999, 9, "x", 0.0), 1.0) == "unknown"
    assert c.handle_control(ControlMessage("reject", "c7-u1", 7, 1, 9, "x", 0.0), 1.0) == "unknown"
    after = (c.reference, c.steps_since, c.trans_since, c.outstanding.uid, c.base_version)
    assert torch.equal(before[0], after[0]) and before[1:] == after[1:] and c.stats["unknown_ctrl"] == 2
    ok = ControlMessage("commit", p.uid, 0, p.update_id, 1, "aggregated", 0.0)
    assert c.handle_control(ok, 2.0) == "commit"
    assert c.handle_control(ok, 3.0) == "duplicate" and c.stats["commits"] == 1


@pytest.mark.parametrize("field", ["stats", "n_samples"])
def test_non_finite_metadata_is_rejected_at_receipt_and_never_aggregated(cfg, field):
    h = Harness(cfg, 2, "async", agg_buffer_size=1)
    h.hold(False)
    c = h.clients[0]
    perturb(c)
    c.steps_since = c.trans_since = 5
    h.env.inject(0, **S)
    theta0 = h.server.theta.clone()
    h.env.links[0] = link(rate=h.rate_for_steps(2))
    h.run_until(lambda: c.outstanding is not None)
    p = c.outstanding
    if field == "stats":
        p.stats["U_mature"] = float("nan")
    else:
        p.n_samples = float("inf")
    h.run(40)
    assert p.status is UpdateStatus.REJECTED and p.reason == "non_finite_metadata"
    assert h.server.version == 0 and torch.equal(h.server.theta, theta0) and torch.isfinite(h.server.theta).all()
    assert torch.equal(c.reference, h.thetas[0])                       # rejection never advanced the reference


def test_nonfinite_statistics_never_enter_the_running_window():
    from fed_matd3_iout.environment.multi_auv_iout_env import MultiAUVIoUTEnv
    from fed_matd3_iout.utils.config import load_config
    env = MultiAUVIoUTEnv(load_config())
    env.reset(0)
    assert env.return_window_stats(0, {"coll_mbit": float("nan"), "owned_loss_mbit": 1.0}, env.episode_id) is False
    assert all(math.isfinite(float(v.sum())) for v in env.win.values())
    assert env.return_window_stats(0, {"coll_mbit": 1.0, "owned_loss_mbit": 1.0}, env.episode_id) is True


def test_packet_statistics_are_a_frozen_copy_of_the_window_and_survive_the_whole_lifecycle(cfg):
    h = Harness(cfg, 2, "async", agg_buffer_size=1)
    h.hold(False)
    c = h.clients[0]
    perturb(c)
    c.steps_since = c.trans_since = 5
    h.env.inject(0, **S)
    h.env.links[0] = link(rate=h.rate_for_steps(2))
    h.run_until(lambda: c.outstanding is not None)
    p = c.outstanding
    frozen = dict(p.stats)
    h.env.win["coll_mbit"][0] += 99.0                                  # later activity in the environment window...
    h.run(60)
    assert p.status is UpdateStatus.AGGREGATED and p.stats == frozen    # ...never reaches the packet; no code path writes pkt.stats
