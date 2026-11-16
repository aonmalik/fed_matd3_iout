import numpy as np
import pytest
import torch
from fed_matd3_iout.baselines.registry import ABLATIONS, METHODS, build_cfg, make_trainer
from fed_matd3_iout.evaluation.evaluator import evaluate, summarise
from fed_matd3_iout.evaluation.metrics import episode_metrics
from fed_matd3_iout.evaluation.report_generator import comparisons, seed_scores, table
from fed_matd3_iout.evaluation.statistical_tests import cohens_d_paired, holm_bonferroni, mean_sd_ci, paired_test
from fed_matd3_iout.experiment import run_one
from fed_matd3_iout.training.trainer import Trainer

CPU = torch.device("cpu")


def tiny(smoke_cfg):
    smoke_cfg.experiment.train_episodes = 2
    smoke_cfg.experiment.eval_scenarios = 2
    smoke_cfg.matd3.ppo.rollout_steps = 20
    smoke_cfg.matd3.ppo.epochs = 1
    return smoke_cfg


def test_full_pipeline_federated(smoke_cfg):
    """Arrivals, allocation, motion, service, local MATD3, intermittent link, upload, USV aggregation, download, evaluation."""
    cfg = build_cfg(tiny(smoke_cfg), "s3a_fedmatd3")
    cfg.experiment.train_episodes = 3
    t = Trainer(cfg, CPU, None, 0)
    t.train(3)
    s = t.proto.summary()
    assert s["received_updates"] > 0 and s["aggregation_rounds"] > 0 and s["bytes_down"] > 0 and s["avg_local_steps"] > 0
    assert 0.0 < s["link_up_fraction"] < 1.0                                         # intermittent, not continuous
    # nothing aggregated before it was received; every aggregated update was logged as received first
    recv = {(e["client"], e["update"]): e["t"] for e in t.server.events if e["ok"]}
    for rec in t.server.history:
        for c, u in zip(rec["clients"], rec["updates"]):
            assert (c, u) in recv and recv[(c, u)] <= rec["t"]
    assert all(c.base_version <= t.server.version for c in t.clients)
    # every received update is exactly one of: aggregated, rejected, or still buffered
    assert t.server.n_received == t.server.n_accepted + t.server.n_rejected + len(t.server.buffer)
    rows = evaluate(cfg, t.global_policies(), 2) + evaluate(cfg, t.local_policies(), 2)
    assert all(r["mass_balance_err"] < 1e-6 for r in rows)
    # evaluation scenarios are matched: two independent evaluations of the same policy are identical
    a = evaluate(cfg, t.global_policies(), 2); b = evaluate(cfg, t.global_policies(), 2)
    assert [r["U_total"] for r in a] == [r["U_total"] for r in b]


@pytest.mark.parametrize("method", list(METHODS))
def test_every_baseline_runs(smoke_cfg, method, tmp_path):
    r = run_one(tiny(smoke_cfg), method, 0, tmp_path, n_episodes=2, verbose=False)
    assert len(r["global"]) == 2 and r["global"][0]["mass_balance_err"] < 1e-6
    fed = METHODS[method]["overrides"].get("federated.enabled", True) is not False and METHODS[method]["trainer"] == "matd3"
    assert bool(r["comm"]) == fed                                                    # only federated methods communicate models
    assert (tmp_path / r["run_dir"].split("/")[-1] / "run_manifest.json").exists()


@pytest.mark.parametrize("name", list(ABLATIONS))
def test_every_ablation_runs_and_changes_only_its_switch(smoke_cfg, name, tmp_path):
    base = build_cfg(tiny(smoke_cfg), "s3a_fedmatd3")
    abl = build_cfg(tiny(smoke_cfg), "s3a_fedmatd3", ABLATIONS[name])
    from fed_matd3_iout.utils.config import to_plain
    def flat(d, p=""):
        o = {}
        for k, v in d.items():
            o.update(flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
        return o
    fb, fa = flat(to_plain(base)), flat(to_plain(abl))
    changed = {k for k in fb if fb[k] != fa[k]}
    assert changed <= set(ABLATIONS[name])                                           # nothing unrelated is touched
    r = run_one(smoke_cfg, "s3a_fedmatd3", 0, tmp_path, overrides=ABLATIONS[name], tag=name, n_episodes=2, verbose=False)
    assert len(r["global"]) == 2


def test_sync_mode_barrier(smoke_cfg):
    cfg = build_cfg(tiny(smoke_cfg), "s3a_fedmatd3", {"federated.mode": "sync"})
    t = Trainer(cfg, CPU, None, 0); t.train(3)
    assert t.server.n_received > 0
    # barrier: a client never has two updates received against the same global base version
    seen = [(e["client"], e["base"]) for e in t.server.events if e["ok"]]
    assert len(seen) == len(set(seen))
    ca = build_cfg(tiny(smoke_cfg), "s3a_fedmatd3")                                  # async control: no such barrier
    ta = Trainer(ca, CPU, None, 0); ta.train(3)
    assert ta.server.n_received > 0


def test_failed_link_means_no_aggregation(smoke_cfg):
    cfg = build_cfg(tiny(smoke_cfg), "s3a_fedmatd3", {"acoustic.usv_link.range_m": 1.0})   # never in contact
    t = Trainer(cfg, CPU, None, 0); t.train(3)
    s = t.proto.summary()
    assert s["received_updates"] == 0 and s["aggregation_rounds"] == 0 and s["bytes_total"] == 0
    assert s["avg_local_steps"] > 0                                                    # local learning continues while disconnected


@pytest.mark.parametrize("M,N", [(1, 10), (3, 20), (10, 40)])
def test_scalability_one_usv(smoke_cfg, M, N):
    cfg = tiny(smoke_cfg)
    cfg.env.n_auvs, cfg.env.n_sensors, cfg.env.episode_steps = M, N, 15
    cfg.matd3.warmup = 10
    cfg = build_cfg(cfg, "s3a_fedmatd3")
    t = Trainer(cfg, CPU, None, 0); t.train(2)
    assert len(t.agents) == M and t.env.usv is not None
    assert all(r["mass_balance_err"] < 1e-6 for r in evaluate(cfg, t.global_policies(), 1))


def test_metric_definitions_and_denominators(smoke_cfg):
    cfg = build_cfg(tiny(smoke_cfg), "centralized_matd3")
    t = Trainer(cfg, CPU, None, 0); t.train(2)
    m = evaluate(cfg, t.local_policies(), 1)[0]
    assert m["collection_ratio"] == pytest.approx(m["collected_mbit"] / m["generated_mbit"])
    assert m["delivery_ratio"] == pytest.approx(m["delivered_mbit"] / m["generated_mbit"])
    assert 0 <= m["U_norm"] <= 2.5 + 1e-9 and 0 <= m["critical_pdr"] <= 1 and 0 <= m["voi_retention"] <= 1 + 1e-12
    assert m["E_total_j"] == pytest.approx(m["E_prop_j"] + m["E_comm_j"] + m["E_comp_j"])


def test_statistics_helpers():
    s = mean_sd_ci([1, 2, 3, 4, 5])
    assert s["mean"] == 3 and s["sd"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1)) and s["ci_lo"] < 3 < s["ci_hi"]
    a, b = np.array([5., 6, 7, 8, 9]), np.array([4., 5.2, 6, 7.1, 8])
    r = paired_test(a, b)
    assert r["mean_diff"] > 0 and r["t_p"] < 0.05 and r["cohens_d"] > 0
    adj = holm_bonferroni([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    with pytest.raises(AssertionError):
        paired_test(np.ones(3), np.ones(4))                                          # unmatched samples are rejected


def test_report_tables_and_comparisons():
    mk = lambda v: {s: [{k: v + s + i for k in ("U_total", "U_norm", "delivered_mbit")} for i in range(2)] for s in range(4)}
    res = {"proposed": mk(10.0), "base": mk(8.0)}
    rows = table(res, ["U_total"])
    assert rows[0]["method"] == "proposed" and "±" in rows[0]["U_total"]
    cmp = comparisons(res, "proposed")
    assert cmp and all("t_p_holm" in c and c["mean_diff"] == pytest.approx(2.0) for c in cmp)
