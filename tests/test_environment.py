import numpy as np
import pytest
from fed_matd3_iout.environment.multi_auv_iout_env import MultiAUVIoUTEnv
from fed_matd3_iout.environment.movement import action_to_vel, step_position
from fed_matd3_iout.environment.energy import propulsion_energy, propulsion_power
from fed_matd3_iout.environment.safety import ProximityTracker, proximity_penalty


def run(env, n, seed=0, act=None):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        o, r, te, tr, info = env.step(act if act is not None else rng.uniform(-1, 1, (env.M, 2)))
        out.append(r)
    return np.array(out)


@pytest.mark.parametrize("M,N", [(1, 5), (3, 20), (5, 3), (10, 40)])
def test_variable_sizes(cfg, M, N):
    cfg.env.n_auvs, cfg.env.n_sensors, cfg.env.episode_steps = M, N, 30
    env = MultiAUVIoUTEnv(cfg)
    obs, info = env.reset(0)
    assert obs.shape == (M, env.obs_dim) and info["state"].shape == (env.state_dim,)
    o, r, te, tr, info = env.step(np.zeros((M, 2)))
    assert r.shape == (M,) and np.all(np.isfinite(o)) and env.usv is not None


def test_exactly_one_usv(cfg):
    env = MultiAUVIoUTEnv(cfg); env.reset(0)
    from fed_matd3_iout.environment.usv import USV
    assert isinstance(env.usv, USV) and not isinstance(env.usv, list)


def test_reproducible_and_scenarios_differ(cfg):
    cfg.env.episode_steps = 40
    a, b, c = MultiAUVIoUTEnv(cfg), MultiAUVIoUTEnv(cfg), MultiAUVIoUTEnv(cfg)
    a.reset(5); b.reset(5); c.reset(6)
    ra, rb, rc = run(a, 40), run(b, 40), run(c, 40)
    assert np.array_equal(ra, rb) and not np.array_equal(ra, rc)
    assert a.accounting() == b.accounting()


def test_streams_are_independent(cfg):
    """Changing only the fading seed must not change layout or arrivals."""
    e1 = MultiAUVIoUTEnv(cfg); e1.reset(0)
    cfg.seeds.fading = 9999
    e2 = MultiAUVIoUTEnv(cfg); e2.reset(0)
    assert np.array_equal(np.array([s.pos for s in e1.sensors]), np.array([s.pos for s in e2.sensors]))
    assert [s.queue.occ for s in e1.sensors] == [s.queue.occ for s in e2.sensors]


def test_boundary_enforced(cfg):
    p, moved, viol, trav = step_position(np.array([0.5, 0.5]), 2.0, np.pi, 1.0, 200.0)
    assert viol and not moved and np.allclose(p, [0.5, 0.5]) and trav == 0.0
    env = MultiAUVIoUTEnv(cfg); env.reset(0)
    run(env, 100, act=np.tile([1.0, 1.0], (env.M, 1)))          # always heading toward -x
    assert all(0 <= a.pos[0] <= env.L and 0 <= a.pos[1] <= env.L for a in env.auvs)


def test_action_mapping_bounds(cfg):
    v, th = action_to_vel(np.array([-1.0, 0.0]), 1.2, 2.2)
    assert v == pytest.approx(1.2)
    v, th = action_to_vel(np.array([1.0, 0.5]), 1.2, 2.2)
    assert v == pytest.approx(2.2) and th == pytest.approx(np.pi / 2)
    v, _ = action_to_vel(np.array([9.0, 0.0]), 1.2, 2.2)           # out-of-range action is clipped
    assert v == pytest.approx(2.2)


def test_energy_model():
    assert propulsion_power(0.0, 20.0, 8.0) == 20.0               # hover: baseline only
    assert propulsion_power(2.0, 20.0, 8.0) == pytest.approx(20 + 8 * 8)
    assert propulsion_energy(2.0, 3.0, 20.0, 8.0) == pytest.approx(3 * 84)


def test_energy_components_disjoint(cfg):
    cfg.env.episode_steps = 80
    env = MultiAUVIoUTEnv(cfg); env.reset(1)
    run(env, 80)
    e = env.energy
    assert env.energy.total() == pytest.approx(sum(e.prop) + sum(e.tx) + sum(e.rx) + sum(e.comp))
    assert sum(e.prop) > 0


def test_safety_tracker():
    pos = np.array([[0.0, 0.0], [3.0, 0.0], [100.0, 100.0]])
    assert proximity_penalty(pos, 5.0)[0] == pytest.approx(2.0)
    t = ProximityTracker(3)
    assert t.update(pos, 5.0) == (1, 1) and t.update(pos, 5.0) == (1, 0)     # one entry event, two pair-steps


def test_observation_freshness(cfg):
    cfg.acoustic.usv_link.range_m = 1.0                           # link never in range -> beliefs go stale
    cfg.env.episode_steps = 30
    env = MultiAUVIoUTEnv(cfg); env.reset(0)
    run(env, 20, act=np.zeros((env.M, 2)))
    assert all(not a.link_up for a in env.auvs) and env.auvs[0].belief_age >= 19


def test_no_negative_buffers_and_ledger(cfg):
    cfg.env.episode_steps = 200
    env = MultiAUVIoUTEnv(cfg); env.reset(2)
    run(env, 200, seed=3)
    assert all(s.queue.occ >= 0 for s in env.sensors) and all(a.onboard_mbit >= -1e-12 for a in env.auvs)
    assert env.ledger.units == pytest.approx(env.delivered_units)


def test_reward_components_logged(cfg):
    env = MultiAUVIoUTEnv(cfg); env.reset(0); run(env, 20)
    assert set(env.rew_comp_sum) == {"utility", "overflow", "energy", "collision", "boundary", "shape"}


def test_non_iid_regional_traffic(cfg):
    cfg.traffic.mode = "regional"; cfg.env.n_sensors = 60
    env = MultiAUVIoUTEnv(cfg); env.reset(0)
    west = [s.lam_base for s in env.sensors if s.pos[0] < 50]; east = [s.lam_base for s in env.sensors if s.pos[0] > 150]
    assert np.mean(east) > np.mean(west)


def test_rate_change(cfg):
    cfg.traffic.rate_change_step, cfg.traffic.rate_change_factor = 5, 3.0
    env = MultiAUVIoUTEnv(cfg); env.reset(0)
    l0 = env.sensor_lams().copy(); run(env, 6)
    assert np.allclose(env.sensor_lams(), 3.0 * l0)


def test_env_arrivals_match_lambda(cfg):
    """Generated data over an episode ~ sum_j lambda_j * T (Poisson arrivals actually used by the environment)."""
    cfg.env.episode_steps = 400
    cfg.env.initial_fill = [0.0, 0.0]
    cfg.env.n_sensors = 12
    env = MultiAUVIoUTEnv(cfg); env.reset(0)
    run(env, 400, act=np.zeros((env.M, 2)))
    gen_units = sum(s.queue.generated for s in env.sensors)
    expected = float(sum(s.lam_base for s in env.sensors)) * 400
    assert gen_units == pytest.approx(expected, rel=0.03)


def test_hover_duration_rule(cfg):
    """Paper 2 Eq.(6): h = max(1, ceil(Q b / (B R dt))), computed once at contact."""
    cfg.env.n_auvs, cfg.env.n_sensors = 1, 1
    cfg.env.initial_fill = [0.2, 0.2]                              # 1000 units
    env = MultiAUVIoUTEnv(cfg); env.reset(0)
    env.sensors[0].lam_base = 0.0                                  # no arrivals -> occupancy known
    a = env.auvs[0]
    assert a.target == 0
    a.pos = env.sensors[0].pos.copy() + np.array([1.0, 0.0])
    occ = env.sensors[0].queue.occ
    h = max(1, int(np.ceil(env.q * occ / (env.sensor_rate * env.dt))))
    env.step(np.zeros((1, 2)))
    assert a.hover_sensor in (0, -1)
    assert a.hover_left == h - 1 or (h == 1 and a.hover_left == 0)


def test_reward_collision_penalty(cfg):
    cfg.env.n_auvs, cfg.env.n_sensors = 2, 5
    env = MultiAUVIoUTEnv(cfg); env.reset(0)
    env.auvs[1].pos = env.auvs[0].pos + np.array([1.0, 0.0])
    env.step(np.zeros((2, 2)))
    assert env.rew_comp_sum["collision"] < 0 and env.prox.events >= 1


def test_compute_energy_accounting(smoke_cfg):
    import torch
    from fed_matd3_iout.baselines.registry import build_cfg
    from fed_matd3_iout.training.trainer import Trainer
    cfg = build_cfg(smoke_cfg, "s3a_fedmatd3")
    t = Trainer(cfg, torch.device("cpu"), None, 0)
    c = t.train(1)
    steps = sum(cl.total_steps for cl in t.clients)
    assert steps > 0 and c[0]["E_comp_j"] == pytest.approx(steps * cfg.energy.e_grad_j)   # E_comp counted once per gradient step
