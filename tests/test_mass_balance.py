import numpy as np
import pytest
from fed_matd3_iout.environment.multi_auv_iout_env import MultiAUVIoUTEnv
from fed_matd3_iout.environment.sensor_queue import SensorQueue
from fed_matd3_iout.semantic.mission_criticality import build_categories


def test_queue_mass_balance_random_ops(cfg):
    rng = np.random.default_rng(5)
    q = SensorQueue(0, 5000, 0.0004, 0.6, build_categories(cfg["semantic"]))
    q.initialise(0, 2000, rng)
    for t in range(500):
        q.arrive(t, rng.poisson(rng.uniform(0, 30)), rng)
        if rng.random() < 0.4:
            q.serve(rng.uniform(0, 400))
        assert q.occ >= 0 and q.occ <= 5000 + 1e-9
        assert q.residual_check() == pytest.approx(0, abs=1e-6)      # G = D + L + B_res


def test_env_end_to_end_mass_balance(cfg):
    cfg.env.episode_steps = 150
    env = MultiAUVIoUTEnv(cfg)
    env.reset(3)
    rng = np.random.default_rng(0)
    for _ in range(150):
        env.step(rng.uniform(-1, 1, (env.M, 2)))
    acc = env.accounting()
    assert abs(acc["sensor_balance_err"]) < 1e-9
    assert abs(acc["e2e_balance_err"]) < 1e-9      # G = delivered + onboard + lost + residual
    assert all(s.queue.occ >= 0 for s in env.sensors)
