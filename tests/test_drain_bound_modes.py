"""D8 drain bound: asynchronous vs synchronous server timers, max_buffer_age_s cap, propagation, final step.
The bound covers SERVER-side resolution only; it never promises that the final downlink notification reaches the client."""
import pytest

from fed_matd3_iout.federated.drain_bound import drain_bound_components, server_side_drain_bound_s
from fed_matd3_iout.utils.config import load_config

PROP = 110.0 / 1500.0


def test_async_uses_agg_timeout(cfg):
    c = drain_bound_components(cfg, "async")
    assert c["timeout_key"] == "agg_timeout_s" and c["buffered_dwell_s"] == pytest.approx(120.0)
    assert c["bound_s"] == pytest.approx(400.0 + PROP + 120.0 + float(cfg["env"]["dt_s"]))
    assert c["bound_s"] == pytest.approx(521.0733, abs=1e-3)


def test_sync_uses_sync_timeout(cfg):
    c = drain_bound_components(cfg, "sync")
    assert c["timeout_key"] == "sync_timeout_s" and c["buffered_dwell_s"] == pytest.approx(300.0)
    assert c["bound_s"] == pytest.approx(400.0 + PROP + 300.0 + float(cfg["env"]["dt_s"]))
    assert c["bound_s"] > drain_bound_components(cfg, "async")["bound_s"]


def test_default_mode_follows_config():
    for mode in ("async", "sync"):
        c = load_config(overrides={"federated.mode": mode})
        assert server_side_drain_bound_s(c) == pytest.approx(drain_bound_components(c, mode)["bound_s"])


def test_max_buffer_age_caps_the_buffered_dwell(cfg):
    """load_config rejects max_buffer_age_s below the timeouts, so the cap is exercised on a plain config mapping."""
    import copy
    c = copy.deepcopy(cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg))
    c["federated"].update({"max_buffer_age_s": 50.0, "agg_timeout_s": 120.0, "sync_timeout_s": 300.0})
    assert drain_bound_components(c, "async")["buffered_dwell_s"] == pytest.approx(50.0)
    assert drain_bound_components(c, "sync")["buffered_dwell_s"] == pytest.approx(50.0)
    c["federated"]["max_buffer_age_s"] = 600.0
    assert drain_bound_components(c, "async")["buffered_dwell_s"] == pytest.approx(120.0)


def test_each_term_enters_the_bound(cfg):
    base = server_side_drain_bound_s(cfg, "async")
    assert server_side_drain_bound_s(load_config(overrides={"federated.upload_timeout_s": 500.0}), "async") == pytest.approx(base + 100.0)
    assert server_side_drain_bound_s(load_config(overrides={"env.dt_s": float(cfg["env"]["dt_s"]) + 1.0}), "async") == pytest.approx(base + 1.0)


def test_unknown_mode_is_rejected(cfg):
    with pytest.raises(ValueError):
        drain_bound_components(cfg, "semi")


def test_configured_drain_does_not_cover_the_sync_bound_and_nothing_hides_it(cfg):
    """The provisional 600 s horizon covers the async bound (521.1 s) but NOT the sync bound (701.1 s): a sync pilot needs an explicit drain value."""
    assert float(cfg["federated"]["final_drain_s"]) >= server_side_drain_bound_s(cfg, "async")
    assert float(cfg["federated"]["final_drain_s"]) < server_side_drain_bound_s(cfg, "sync")
