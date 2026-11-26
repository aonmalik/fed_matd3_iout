"""Diagnostics of the (unchanged) semantic aggregation: correctness of the numbers and 'diagnostics never alter learning'."""
import numpy as np
import pytest
import torch

from fed_fixtures import Harness, perturb
from fed_matd3_iout.federated.aggregation import aggregate, l2_clip, normalise_with_cap, raw_weights
from fed_matd3_iout.validation.aggregation_diagnostics import diagnose_rounds


def rec(clients, M, n=None, stale=None, w=None, cf_tv=0.0, cf_eff=0.0, eps=0.1):
    k = len(clients)
    n = n or [1000.0] * k
    stale = stale or [1.0] * k
    z = raw_weights(n, stale, M, 1.0, 1.0, eps)
    w = normalise_with_cap(z, 1.0) if w is None else w
    return {"clients": clients, "relevance": M, "n": n, "stale_factor": stale, "w": list(w), "cf_rel_tv": cf_tv, "cf_rel_effect": cf_eff}


def test_single_update_rounds_are_counted_and_inert(cfg):
    hist = [rec([0], [0.3]), rec([1], [0.0]), rec([0, 1], [0.2, 0.6]), rec([0, 1, 2], [0.5, 0.5, 0.5])]
    d = diagnose_rounds(hist, cfg.federated)
    assert d["n_rounds"] == 4 and d["single_update_rounds"] == 2 and d["single_update_round_fraction"] == 0.5
    assert d["participants_per_round"]["histogram"] == {1: 2, 2: 1, 3: 1}
    assert d["verdict"]["fraction_of_rounds_where_relevance_can_act"] == 0.5


def test_log_variance_shares_sum_to_one_and_attribute_correctly(cfg):
    f = cfg.federated
    r = rec([0, 1, 2], [0.0, 0.5, 1.0], n=[800.0, 1000.0, 1200.0], stale=[1.0, 0.74, 0.55])
    d = diagnose_rounds([r], f)
    sh = d["weight_variance_shares_multi_rounds"]
    assert sum(v["mean"] for v in sh.values()) == pytest.approx(1.0, abs=1e-9)
    only_rel = diagnose_rounds([rec([0, 1, 2], [0.0, 0.5, 1.0])], f)["weight_variance_shares_multi_rounds"]
    assert only_rel["relevance"]["mean"] == pytest.approx(1.0) and only_rel["sample_count"]["mean"] == pytest.approx(0.0)


def test_equal_relevance_has_no_contribution_and_verdict_says_so(cfg):
    d = diagnose_rounds([rec([0, 1], [0.0, 0.0]) for _ in range(5)], cfg.federated)
    assert d["relevance_factor_g"]["within_round_spread_max_over_min"]["max"] == 1.0
    assert "NOT expected to be measurable" in d["verdict"]["reading"]


def test_large_relevance_spread_is_reported_as_measurable(cfg):
    hist = [rec([0, 1], [0.05, 1.0], cf_tv=0.3, cf_eff=0.2) for _ in range(5)]
    assert "measurable mechanism" in diagnose_rounds(hist, cfg.federated)["verdict"]["reading"]


def test_per_client_and_weight_variation(cfg):
    hist = [rec([0, 1], [0.2, 0.4]), rec([0, 1], [0.6, 0.4]), rec([1], [0.4])]
    d = diagnose_rounds(hist, cfg.federated)
    assert set(d["per_client"]) == {0, 1} and d["per_client"][1]["rounds_participated"] == 3
    assert d["per_client"][0]["weight_cv"] is not None and d["weight_entropy_per_round"]["std"] >= 0


def test_server_records_counterfactual_and_does_not_change_the_model(cfg):
    """The relevance counterfactual is diagnostic only: theta' = theta + sum_i w_i Clip(Delta_i) with the NORMAL weights."""
    h = Harness(cfg, 2, "async", agg_buffer_size=2, aggregation="s3a", max_client_weight=1.0)
    c0, c1 = h.clients
    for c in (c0, c1):
        perturb(c, 0.1 if c.id == 0 else 0.3); c.steps_since = c.trans_since = 5
    h.env.inject(0, coll_mbit=9.0, owned_loss_mbit=1.0, U_mature=8.0, V_mature=10.0)    # high relevance
    h.env.inject(1, coll_mbit=1.0, owned_loss_mbit=9.0, U_mature=1.0, V_mature=10.0)    # low relevance
    theta0 = h.server.theta.clone()
    h.run_until(lambda: h.server.version == 1, 300)
    r = h.server.history[0]
    assert r["cf_rel_tv"] > 0 and r["cf_rel_effect"] > 0 and abs(sum(r["w_norel"]) - 1) < 1e-9 and r["w_norel"] == pytest.approx([0.5, 0.5], abs=1e-9)
    expect = theta0.clone()
    for p, w in zip((h.pkts["c0-u1"], h.pkts["c1-u1"]), r["w"]):
        expect += w * l2_clip(p.delta, None)[0]
    assert torch.allclose(h.server.theta, expect, atol=1e-6)                               # applied update uses the real weights only


def test_fedavg_has_no_relevance_counterfactual(cfg):
    h = Harness(cfg, 1, "async", agg_buffer_size=1, aggregation="fedavg")
    c = h.clients[0]; perturb(c); c.steps_since = c.trans_since = 5
    h.run_until(lambda: h.server.version == 1, 300)
    assert h.server.history[0]["cf_rel_tv"] == 0.0 and h.server.history[0]["cf_rel_effect"] == 0.0
