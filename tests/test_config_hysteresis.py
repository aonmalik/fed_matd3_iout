"""Regression tests 17-18: smoke configuration consistency and offload-hysteresis behaviour (audit item 5)."""
import copy

import pytest

from fed_matd3_iout.environment.multi_auv_iout_env import MultiAUVIoUTEnv
from fed_matd3_iout.utils.config import ConfigError, load_config, validate_config


# ----------------------------------------------------------------------------- 17
def test_smoke_offload_hysteresis_is_ordered(smoke_cfg):
    e = smoke_cfg.env
    assert 0.0 <= e.offload_lo < e.offload_hi <= 1.0
    validate_config(smoke_cfg)


def test_main_config_unchanged_and_valid(cfg):
    assert cfg.env.offload_hi == 0.5 and cfg.env.offload_lo == 0.05      # main-experiment meaning preserved
    assert cfg.allocation.contact_time_s == cfg.env.max_hover_steps * cfg.env.dt_s
    validate_config(cfg)


def test_smoke_contact_time_matches_hover_budget(smoke_cfg):
    assert smoke_cfg.allocation.contact_time_s == pytest.approx(smoke_cfg.env.max_hover_steps * smoke_cfg.env.dt_s)


def test_inconsistent_configs_are_rejected(cfg):
    bad = copy.deepcopy(cfg)
    bad.env.offload_hi, bad.env.offload_lo = 0.02, 0.05                  # the audited smoke defect
    with pytest.raises(ConfigError, match="offload"):
        validate_config(bad)
    bad = copy.deepcopy(cfg)
    bad.allocation.contact_time_s = 10 * bad.env.max_hover_steps
    with pytest.raises(ConfigError, match="contact_time_s"):
        validate_config(bad)
    bad = copy.deepcopy(cfg)
    bad.federated.max_buffer_age_s = 1.0
    with pytest.raises(ConfigError, match="max_buffer_age_s"):
        validate_config(bad)
    with pytest.raises(ConfigError):                                      # load_config validates overrides too
        load_config(overrides={"env.offload_hi": 0.01})


# ----------------------------------------------------------------------------- 18
def _env(smoke_cfg):
    env = MultiAUVIoUTEnv(smoke_cfg)
    env.reset(0)
    a = env.auvs[0]
    a.hover_left = 0
    a.to_usv = False
    return env, a


def test_hysteresis_does_not_chatter(smoke_cfg):
    env, a = _env(smoke_cfg)
    hi, lo = smoke_cfg.env.offload_hi, smoke_cfg.env.offload_lo
    switches, modes = 0, []

    def at(fill):
        nonlocal switches
        a.onboard_mbit = fill * a.cap_mbit
        before = a.to_usv
        env._update_mode(a)
        switches += int(a.to_usv != before)
        modes.append(a.to_usv)

    for f in (0.0, lo, (lo + hi) / 2, hi * 0.99):                        # below the upper threshold: stays sensing
        at(f)
    assert not any(modes) and switches == 0
    at(hi)                                                               # crosses hi: exactly one switch to offload
    assert a.to_usv and switches == 1
    for f in (hi, hi, (lo + hi) / 2, hi * 0.8, lo * 1.5, lo * 1.01):     # inside the band: no flip back, no repeat flips
        at(f)
    assert a.to_usv and switches == 1
    at(lo)                                                               # drops to lo: exactly one switch back
    assert not a.to_usv and switches == 2
    for f in (lo, (lo + hi) / 2, hi * 0.99):                             # back inside the band: stays sensing
        at(f)
    assert not a.to_usv and switches == 2


def test_no_immediate_remode_after_switch_at_same_fill(smoke_cfg):
    env, a = _env(smoke_cfg)
    hi = smoke_cfg.env.offload_hi
    a.onboard_mbit = hi * a.cap_mbit
    env._update_mode(a)
    assert a.to_usv
    for _ in range(5):                                                   # same fill, repeated calls: no oscillation
        env._update_mode(a)
        assert a.to_usv
    smoke_bad = copy.deepcopy(smoke_cfg)                                 # contrast: the audited (inverted) band WOULD chatter
    smoke_bad.env.offload_hi, smoke_bad.env.offload_lo = 0.02, 0.05
    env2 = MultiAUVIoUTEnv(smoke_bad)
    env2.reset(0)
    b = env2.auvs[0]
    b.hover_left = 0
    b.onboard_mbit = 0.03 * b.cap_mbit                                   # lo > fill >= hi: switches on, then straight off
    seen = []
    for _ in range(4):
        env2._update_mode(b)
        seen.append(b.to_usv)
    assert seen == [True, False, True, False]
