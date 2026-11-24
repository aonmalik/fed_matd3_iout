import itertools
import numpy as np
import pytest
from fed_matd3_iout.allocation.semantic_service_cost import AllocState, cost_matrix
from fed_matd3_iout.allocation.task_allocator import TaskAllocator


def mkstate(M=3, N=5, seed=0, cur=None):
    r = np.random.default_rng(seed)
    return AllocState(auv_pos=r.uniform(0, 200, (M, 2)), auv_load=r.uniform(0, 1, M), auv_free_mbit=np.full(M, 10.0),
                      cur_target=np.full(M, -1) if cur is None else np.array(cur), sensor_pos=r.uniform(0, 200, (N, 2)),
                      b=r.uniform(0, 5000, N), lam=r.uniform(3, 30, N), b_max=5000.0, q=0.0004,
                      density=r.uniform(0.1, 1, N), last_served=np.full(N, -1), t_step=10, episode_steps=1000,
                      rate_mbps=0.05, d_max=283.0)


def test_one_to_one_and_optimal(cfg):
    st = mkstate(3, 5)
    al = TaskAllocator(cfg.allocation)
    res = al.allocate(st, [0, 1, 2], list(range(5)))
    tg = [v for v in res.values() if v >= 0]
    assert len(tg) == len(set(tg)) == 3                          # no sensor assigned twice
    C, _ = cost_matrix(st, al.w, "exact", al.v_nom, al.contact_time, al.tol)
    best = min(sum(C[i, p[i]] for i in range(3)) for p in itertools.permutations(range(5), 3))
    got = sum(C[i, res[i]] for i in range(3))
    assert got == pytest.approx(best)                            # Hungarian == brute-force optimum


def test_more_auvs_than_sensors_leaves_idle(cfg):
    st = mkstate(4, 2)
    res = TaskAllocator(cfg.allocation).allocate(st, [0, 1, 2, 3], [0, 1])
    assert sum(1 for v in res.values() if v >= 0) == 2 and sum(1 for v in res.values() if v < 0) == 2


def test_no_free_sensors_or_no_participants(cfg):
    st = mkstate()
    al = TaskAllocator(cfg.allocation)
    assert al.allocate(st, [0, 1], []) == {0: -1, 1: -1}
    assert al.allocate(st, [], [0, 1]) == {}


def test_cost_terms_normalised(cfg):
    st = mkstate()
    al = TaskAllocator(cfg.allocation)
    _, d = cost_matrix(st, al.w, "exact", al.v_nom, al.contact_time, al.tol)
    for k in ("B", "U", "T", "L", "S", "W", "A"):
        assert d[k].min() >= 0 and d[k].max() <= 1 + 1e-9, k


def test_backlog_term_prefers_fuller_sensor(cfg):
    st = mkstate(1, 2)
    st.auv_pos[:] = 100; st.sensor_pos[:] = 100                 # equal distance -> equal travel
    st.b = np.array([4000.0, 100.0]); st.lam[:] = 5.0; st.density[:] = 0.5
    res = TaskAllocator(cfg.allocation).allocate(st, [0], [0, 1])
    assert res[0] == 0


def test_semantic_term_and_ablation(cfg):
    st = mkstate(1, 2)
    st.auv_pos[:] = 100; st.sensor_pos[:] = 100
    st.b[:] = 2000.0; st.lam[:] = 5.0; st.density = np.array([0.2, 1.0])
    assert TaskAllocator(cfg.allocation, use_semantic=True).allocate(st, [0], [0, 1])[0] == 1
    cfg.allocation.weights.semantic = 0.0                         # component switched off -> tie broken by solver, term gone
    C, _ = cost_matrix(st, dict(cfg.allocation.weights), "exact", 1.7, 60.0, 1e-9)
    assert C[0, 0] == pytest.approx(C[0, 1])


def test_assignment_continuity(cfg):
    st = mkstate(2, 3, seed=4, cur=[1, 0])
    w = dict(cfg.allocation.weights)
    w.update(switch=50.0)
    C, d = cost_matrix(st, w, "exact", 1.7, 60.0, 1e-9)
    assert d["S"][0, 1] == 0 and d["S"][0, 0] == 1 and d["S"][1, 0] == 0
    cfg.allocation.weights.update(switch=50.0)
    res = TaskAllocator(cfg.allocation).allocate(st, [0, 1], [0, 1, 2])
    assert res == {0: 1, 1: 0}


def test_ce_vs_exact_models_both_run(cfg):
    st = mkstate()
    for m in ("exact", "ce", "current"):
        cfg.allocation.backlog_model = m
        res = TaskAllocator(cfg.allocation).allocate(st, [0, 1, 2], list(range(5)))
        assert len(set(v for v in res.values() if v >= 0)) == 3


def test_greedy_reassigns_only_freed(cfg):
    st = mkstate(3, 5, cur=[0, 1, 2])
    cfg.allocation.method = "greedy"
    res = TaskAllocator(cfg.allocation).allocate(st, [0, 1, 2], [3, 4], freed=1)
    assert res[0] == 0 and res[2] == 2 and res[1] in (3, 4)
