"""Protocol 2.1: bounded retry/back-off after REJECTED / EXPIRED updates. All scenarios are deterministic (no RNG, no jitter)."""
import math

import pytest
import torch

from fed_fixtures import Harness, link, perturb
from fed_matd3_iout.federated.update_packet import UpdateStatus
from fed_matd3_iout.utils.config import ConfigError, load_config, validate_config

S = dict(coll_mbit=5.0, owned_loss_mbit=2.0, U_mature=3.0, V_mature=4.0)
RETRY = dict(enabled=True, base_s=20.0, factor=2.0, max_s=80.0)


def _poisoned(cfg, retry=RETRY, n=2, **fed):
    """Client 0 persistently produces a non-finite update (rejected at receipt every time)."""
    h = Harness(cfg, n, "async", retry=retry, agg_buffer_size=1, **fed)
    c = h.clients[0]
    perturb(c)
    c.steps_since = c.trans_since = 5
    with torch.no_grad():
        next(c.agent.actor.parameters()).view(-1)[0] = float("nan")
    return h, c


def _ok(h):
    """All lifecycle invariants; 'all_parameters_finite' is excluded because client 0's LOCAL actor is poisoned on purpose."""
    inv = h.check_invariants()
    inv.pop("all_parameters_finite")
    assert all(torch.isfinite(h.server.theta).all() for _ in [0]) and all(inv.values()), inv
    return True


def _created_times(h):
    return [p.t_created for p in h.pkts.values() if p.client_id == 0]


def test_schedule_formula_and_cap(cfg):
    h = Harness(cfg, 1, "async", retry=RETRY)
    c = h.clients[0]
    assert [c.backoff_delay(k) for k in range(0, 7)] == [0, 20, 40, 80, 80, 80, 80]


def test_persistent_rejection_is_bounded_and_never_advances_reference(cfg):
    T = 400
    h, c = _poisoned(cfg)
    ref0 = c.reference.clone()
    h.run(T)
    n_with = h.proto.n_created
    # analytic upper bound on attempts in T seconds: gaps are >= 20, 40, 80, 80, ... after each learned failure
    gaps, t, bound = [20, 40, 80], 0.0, 1
    k = 0
    while True:
        t += gaps[min(k, 2)]
        if t > T:
            break
        bound += 1
        k += 1
    assert 2 <= n_with <= bound, (n_with, bound)
    s = h.proto.summary()
    assert s["rejected_updates"] >= n_with - 1 and s["aggregated_updates"] == 0 and h.server.version == 0
    assert torch.equal(c.reference, ref0)                                        # no reference movement after any rejection
    assert s["retry_attempts"] == n_with - 1 and s["retry_deferred_updates"] >= 1 and s["retry_deferred_steps"] > 0
    assert s["rejected_bytes"] == pytest.approx(s["rejected_updates"] * h.model_bytes)
    assert s["wasted_uplink_bytes"] == pytest.approx(s["rejected_bytes"] + s["expired_bytes"])
    assert s["backoff_s_total"] > 0
    assert _ok(h)
    # contrast: without back-off the same client re-uploads as soon as the outcome is known
    h0, c0 = _poisoned(cfg, retry=None)
    h0.run(T)
    assert h0.proto.n_created > 3 * n_with
    assert h0.proto.summary()["rejected_bytes"] > 3 * s["rejected_bytes"]
    s0 = h0.proto.summary()                                                       # retries are COUNTED even when not limited
    assert s0["retry_attempts"] == h0.proto.n_created - 1 and s0["retry_deferred_updates"] == 0 and s0["backoff_s_total"] == 0.0


def test_retry_only_delays_and_starts_after_the_deadline(cfg):
    h, c = _poisoned(cfg)
    seen = []
    for _ in range(300):
        h.step()
        if c.outstanding is None and c.consecutive_failures and (not seen or seen[-1][0] != c.consecutive_failures):
            seen.append((c.consecutive_failures, c.retry_not_before, float(h.t)))
    times = _created_times(h)
    assert len(times) >= 3
    for (k, nb, _t_learned), t_next in zip(seen, times[1:]):
        assert t_next >= nb                                                      # never earlier than the deadline
        assert t_next - nb <= 3.0                                                # ... and promptly at it (no extra idling)
    assert [round(nb - (tl - 1)) for _, nb, tl in seen][:3] == [20, 40, 80]   # h.t is already incremented


def test_deferral_counts_once_per_episode_of_deferral(cfg):
    h, c = _poisoned(cfg)
    h.run(200)
    s = c.stats
    assert s["retry_deferred"] <= s["retries"] + 1                               # one deferral episode per failure
    assert s["retry_deferred_steps"] >= s["retry_deferred"]


def test_commit_resets_backoff(cfg):
    h, c = _poisoned(cfg)
    h.run_until(lambda: c.consecutive_failures >= 2, 200)
    assert c.consecutive_failures >= 2 and c.retry_not_before > -math.inf
    with torch.no_grad():                                                        # the fault clears
        c.agent.actor.load_state_dict(h.clients[1].agent.actor.state_dict())
    perturb(c, 0.01)
    h.run_until(lambda: c.stats["commits"] >= 1, 400)
    assert c.stats["commits"] == 1 and c.consecutive_failures == 0 and c.retry_not_before == -math.inf
    assert c.backoff_delay(0) == 0.0
    assert all(h.check_invariants().values())


def test_new_global_model_resets_backoff_and_reference_is_the_global(cfg):
    h, c = _poisoned(cfg)
    h.run_until(lambda: c.consecutive_failures >= 1 and c.outstanding is None, 100)
    assert c.retry_not_before > h.t
    theta = h.server.theta.clone()
    c.install_global(theta, 1, 1.0)
    assert c.consecutive_failures == 0 and c.retry_not_before == -math.inf
    assert torch.equal(c.reference, theta)                                       # reference = installed global, not a rejected snapshot


def test_upload_timeout_expiry_also_backs_off_and_counts_partial_bytes(cfg):
    h = Harness(cfg, 1, "async", retry=RETRY, upload_timeout_s=5.0)
    c = h.clients[0]
    perturb(c); c.steps_since = c.trans_since = 5
    h.env.inject(0, **S)
    h.env.links[0] = link(up=True, success=False)                                # nothing ever gets through
    h.run(40)
    s = h.proto.summary()
    assert s["expired_updates"] >= 1 and s["transmitted_updates"] == 0 and s["rejected_bytes"] == 0.0
    assert c.consecutive_failures >= 1 and c.stats["retry_deferred"] >= 1
    assert s["expired_bytes"] == pytest.approx(sum(p.bytes_sent for p in h.pkts.values() if p.status is UpdateStatus.EXPIRED))
    assert s["expired_bytes"] < s["expired_updates"] * h.model_bytes             # partial / zero, not a full model
    assert all(h.check_invariants().values())


def test_statistics_returned_exactly_once_per_failed_update_under_backoff(cfg):
    h, c = _poisoned(cfg)
    h.env.inject(0, **S)
    h.run(300)
    acc = h.proto.statistics_accounting()
    assert acc["balanced"] and h.proto.stats_disposition["consumed"] == 0
    assert _ok(h)                                                                # includes stats conservation


def test_backoff_is_deterministic(cfg):
    runs = []
    for _ in range(2):
        h, c = _poisoned(cfg)
        h.run(300)
        runs.append((_created_times(h), h.proto.summary()["rejected_bytes"], c.stats["retry_deferred_steps"]))
    assert runs[0] == runs[1]


def test_healthy_client_is_unaffected(cfg):
    """A client that never fails must see identical behaviour with and without back-off."""
    out = []
    for r in (RETRY, None):
        h = Harness(cfg, 1, "async", retry=r, agg_buffer_size=1)
        c = h.clients[0]
        perturb(c); c.steps_since = c.trans_since = 5
        h.env.inject(0, **S)
        h.run(60)
        out.append((h.proto.n_created, h.proto.n_status["aggregated"], c.stats["retry_deferred"], h.server.version))
    assert out[0] == out[1] and out[0][2] == 0


def test_config_validation_and_protocol_20_has_no_retry(cfg):
    assert cfg.federated.retry.enabled and cfg.federated.retry.max_s >= cfg.federated.retry.base_s
    import copy
    bad = copy.deepcopy(cfg); bad.federated.retry.factor = 0.5
    with pytest.raises(ConfigError, match="retry"):
        validate_config(bad)
    from pathlib import Path
    old = load_config(cfg_dir=Path(__file__).resolve().parents[1] / "configs" / "protocols" / "v2.0")
    assert old.federated.get("retry") is None
    h = Harness(old, 1, "async")
    assert h.clients[0].retry is None                                            # protocol 2.0 semantics preserved
