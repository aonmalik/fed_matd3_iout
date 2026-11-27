"""Diagnostics-only additions: relevance status classification, per-update/per-round logging, episode-boundary reporting and the extended
statistics mass balance.  None of these tests (or the code they cover) may change rho_sem, rho_svc, M, weights, attribution or settlement."""
import json

import numpy as np
import pytest

from fed_fixtures import build, perturb, run
from fed_matd3_iout.environment.multi_auv_iout_env import MultiAUVIoUTEnv
from fed_matd3_iout.federated import communication_protocol as cp
from fed_matd3_iout.federated.relevance import (SEM_COLLECTED_NO_MATURE, SEM_MATURE_EVIDENCE, SEM_NO_COLLECTION, SEM_STATUSES, SVC_EVIDENCE,
                                                SVC_NO_FLOW, SVC_STATUSES, relevance, relevance_status, rho_sem, rho_svc)
from fed_matd3_iout.semantic.attribution import BatchBook
from fed_matd3_iout.semantic.packet import Batch
from fed_matd3_iout.validation.aggregation_diagnostics import diagnose_rounds, summarize_relevance_status

ADD = ("coll_mbit", "owned_loss_mbit", "V_mature", "U_mature", "units_del_mature", "units_coll_mature", "n_mature_batches")
FULL = {"coll_mbit": 5.0, "owned_loss_mbit": 2.0, "V_mature": 4.0, "U_mature": 3.0, "units_del_mature": 6.0, "units_coll_mature": 7.0,
        "n_mature_batches": 3.0}


def B(pkt, t, units=10.0, mu=0.01, sigma=1.0, c=1.0):
    return Batch(pkt_id=pkt, sensor_id=0, t_gen=t - 1.0, units=units, sigma=sigma, category="routine", mu=mu, c=c, t_collect=t, collected_by=0)


# ------------------------------------------------------------------------------------------------ classification
def test_status_categories_are_exclusive_and_consistent_with_rho():
    cases = {
        "nothing": ({}, SEM_NO_COLLECTION, SVC_NO_FLOW),
        "collected_not_mature": ({"coll_mbit": 2.0, "n_immature_batches": 1.0}, SEM_COLLECTED_NO_MATURE, SVC_EVIDENCE),
        "immature_from_earlier_window": ({"n_immature_batches": 2.0}, SEM_COLLECTED_NO_MATURE, SVC_NO_FLOW),
        "mature": ({"V_mature": 4.0, "U_mature": 3.0, "coll_mbit": 1.0}, SEM_MATURE_EVIDENCE, SVC_EVIDENCE),
        "mature_without_new_collection": ({"V_mature": 4.0, "U_mature": 0.0}, SEM_MATURE_EVIDENCE, SVC_NO_FLOW),
        "no_collection_but_owned_overflow": ({"owned_loss_mbit": 3.0}, SEM_NO_COLLECTION, SVC_EVIDENCE),
    }
    for name, (st, sem, svc) in cases.items():
        r = relevance_status(st)
        assert (r["sem"], r["svc"]) == (sem, svc), name
        assert r["sem"] in SEM_STATUSES and r["svc"] in SVC_STATUSES
        assert r["rho_sem_defined"] == (rho_sem(st) is not None) == (r["sem"] == SEM_MATURE_EVIDENCE), name
        assert r["rho_svc_defined"] == (rho_svc(st) is not None) == (r["svc"] == SVC_EVIDENCE), name


def test_undefined_is_not_conflated_with_zero_relevance():
    undefined = {"owned_loss_mbit": 0.0, "coll_mbit": 0.0}                       # no evidence at all
    zero_with_evidence = {"owned_loss_mbit": 3.0, "coll_mbit": 0.0}              # owned sensors overflowed, nothing collected: rho_svc == 0
    for kind in ("service", "s3a"):
        assert relevance(undefined, kind, 0.5) == relevance(zero_with_evidence, kind, 0.5) == 0.0     # M is identical (unchanged behaviour) ...
    a, b = relevance_status(undefined), relevance_status(zero_with_evidence)
    assert a["svc"] == SVC_NO_FLOW and not a["svc_zero_with_evidence"]            # ... but the diagnostic tells them apart
    assert b["svc"] == SVC_EVIDENCE and b["svc_zero_with_evidence"] and rho_svc(zero_with_evidence) == 0.0


def test_status_does_not_change_the_relevance_formula():
    st = dict(FULL)
    before = [relevance(st, k, 0.5) for k in ("semantic", "service", "s3a", "fedavg")]
    relevance_status(st)
    assert before == [relevance(st, k, 0.5) for k in ("semantic", "service", "s3a", "fedavg")]
    assert relevance(st, "s3a", 0.5) == pytest.approx(0.5 * 3.0 / 4.0 + 0.5 * 5.0 / 7.0)
    unit = relevance_status({"units_coll_mature": 2.0, "units_del_mature": 1.0}, use_semantics=False)
    assert unit["sem"] == SEM_MATURE_EVIDENCE and unit["attribution_mode"] == "unit_based"       # neutral label, mode stated explicitly
    assert relevance_status(dict(FULL))["attribution_mode"] == "semantic_utility"
    assert "semantic" not in SEM_MATURE_EVIDENCE                                                  # the label must not imply semantic attribution
    # unit-based mode looks at units, not utility: V_mature alone is NOT evidence there, and vice versa
    assert relevance_status({"V_mature": 4.0, "U_mature": 1.0}, use_semantics=False)["rho_sem_defined"] is False
    assert relevance_status({"units_coll_mature": 2.0}, use_semantics=True)["rho_sem_defined"] is False


# ------------------------------------------------------------------------------------------------ immature / mature, exactly once, no double count
def test_immature_batches_are_open_not_failures_and_claimed_exactly_once():
    bk = BatchBook(1, maturity_eps=0.05)                                    # horizon ln(20)/0.01 = 299.6 s
    bk.on_collect(0, B(1, 100.0))
    c1 = bk.claim(0, 150.0)
    assert c1["V_mature"] == 0.0 and c1["n_mature_batches"] == 0.0 and c1["n_immature_batches"] == 1.0
    st = {"coll_mbit": 0.0, **c1}
    assert relevance_status(st)["sem"] == SEM_COLLECTED_NO_MATURE             # open batch is NOT 'no collection' and NOT a failure
    bk.on_deliver(0, B(1, 100.0, units=4.0), utility=3.0)
    c2 = bk.claim(0, 500.0)                                                    # matured by age: 4/10 delivered
    assert c2["n_mature_batches"] == 1.0 and c2["V_mature"] == 10.0 and c2["U_mature"] == 3.0
    assert relevance_status({"coll_mbit": 0.0, **c2})["sem"] == SEM_MATURE_EVIDENCE
    c3 = bk.claim(0, 900.0)                                                    # nothing left: no double counting
    assert c3["V_mature"] == 0.0 and c3["n_mature_batches"] == 0.0 and c3["n_immature_batches"] == 0.0
    assert bk.claimed_V[0] == 10.0 and bk.claimed_entries[0] == 1
    bk.on_deliver(0, B(1, 100.0, units=6.0), utility=1.0)                      # late remainder is not re-attributed
    assert bk.claim(0, 1000.0)["V_mature"] == 0.0 and bk.late_units[0] == 6.0
    assert bk.claimed_U[0] + bk.pending(0)["U"] + bk.late_U[0] == pytest.approx(4.0)       # mass balance of delivered utility


def test_fully_delivered_batch_matures_immediately_and_bounds_hold():
    bk = BatchBook(1, 0.05)
    bk.on_collect(0, B(1, 10.0, units=5.0, sigma=0.8, c=2.0))
    bk.on_deliver(0, B(1, 10.0, units=5.0, sigma=0.8, c=2.0), utility=4.0)
    c = bk.claim(0, 11.0)
    assert c["n_mature_batches"] == 1.0 and 0.0 <= c["U_mature"] <= c["V_mature"] == 8.0
    assert 0.0 <= rho_sem(c) <= 1.0


# ------------------------------------------------------------------------------------------------ episode boundary
def test_episode_reset_reports_discarded_partial_window_and_censored_batches(smoke_cfg):
    env = MultiAUVIoUTEnv(smoke_cfg)
    env.reset(0)
    assert env.episode_boundary_diagnostics()["n_boundaries"] == 0
    env.win["coll_mbit"][0] = 2.5
    env.win["owned_loss_mbit"][1] = 1.5
    env.book.on_collect(0, B(7, 5.0, units=4.0, sigma=0.5, c=2.0))
    env.book.on_deliver(0, B(7, 5.0, units=1.0, sigma=0.5, c=2.0), utility=0.7)
    ep = env.episode_id
    env.reset(1)
    d = env.episode_boundary_diagnostics()
    assert d["n_boundaries"] == 1 and d["current_episode_id"] == ep + 1
    assert d["dropped_partial_window"]["coll_mbit"] == 2.5 and d["dropped_partial_window"]["owned_loss_mbit"] == 1.5
    rec = d["log"][0]
    assert rec["ended_episode_id"] == ep and rec["partial_window_coll_mbit_by_auv"][0] == 2.5
    assert rec["censored_open_batches"]["n_entries"] == 1 and rec["censored_open_batches"]["V"] == pytest.approx(4.0)
    assert rec["censored_open_batches"]["U_delivered_so_far"] == pytest.approx(0.7) and d["censored_V_total"] == pytest.approx(4.0)
    assert all(float(v.sum()) == 0.0 for v in env.win.values())                # unchanged behaviour: the new window is empty
    assert env.book.pending(0)["n"] == 0                                       # ... and the new book is empty


def test_boundary_diagnostics_accumulate_and_report_refused_stats(smoke_cfg):
    env = MultiAUVIoUTEnv(smoke_cfg)
    env.reset(0)
    ep1 = env.episode_id
    env.win["coll_mbit"][0] = 1.0
    env.reset(1)
    env.win["coll_mbit"][0] = 2.0
    env.reset(2)
    d = env.episode_boundary_diagnostics()
    assert d["n_boundaries"] == 2 and d["dropped_partial_window"]["coll_mbit"] == 3.0
    assert env.return_window_stats(0, {"coll_mbit": 9.0, "V_mature": 4.0, "U_mature": float("nan")}, ep1) is False
    d = env.episode_boundary_diagnostics()
    assert d["stats_refused_episode_count"] == 1 and d["stats_refused_episode_mass"]["coll_mbit"] == 9.0
    assert d["stats_refused_episode_mass"]["V_mature"] == 4.0 and d["stats_refused_episode_mass"]["U_mature"] == 0.0   # non-finite never summed
    assert all(float(v.sum()) == 0.0 for v in env.win.values())
    json.dumps(d)


# ------------------------------------------------------------------------------------------------ extended mass balance (settlement unchanged)
def test_acc_keys_cover_all_additive_window_keys():
    assert tuple(cp._ACC_KEYS) == ADD
    assert set(ADD) <= set(MultiAUVIoUTEnv.WINDOW_KEYS)


def _start(cfg, **fed):
    cs, srv, proto, env = build(cfg, 1, **fed)
    c = cs[0]
    perturb(c)
    c.steps_since, c.trans_since = 5, 5
    for k, v in FULL.items():
        env.win[k][0] = v
    return c, srv, proto, env


def _win(env):
    return {k: float(env.win[k][0]) for k in ADD}


@pytest.mark.parametrize("scenario", ["rejected", "expired_buffer", "aggregated"])
def test_all_additive_keys_conserved_for_rejected_expired_and_aggregated(cfg, scenario):
    kw = {"rejected": dict(max_staleness=0), "expired_buffer": dict(agg_buffer_size=99, agg_timeout_s=1e9, max_buffer_age_s=10.0),
          "aggregated": dict(agg_buffer_size=1)}[scenario]
    c, srv, proto, env = _start(cfg, **kw)
    if scenario == "rejected":
        srv.version = 4
    run(proto, env, 0, 1)
    assert proto.n_created == 1
    c.steps_since = 0
    assert all(env.win[k][0] == 0.0 for k in ADD)                              # the update owns the statistics
    run(proto, env, 1, 40)
    disp = {"rejected": "returned", "expired_buffer": "returned", "aggregated": "consumed"}[scenario]
    other = "consumed" if disp == "returned" else "returned"
    assert proto.stats_disposition[disp] == 1 and proto.stats_disposition[other] == 0
    for k in ADD:
        assert proto.stats_mass[disp][k] == FULL[k] and proto.stats_mass[other][k] == 0.0 and proto.stats_mass["discarded_episode"][k] == 0.0
    assert _win(env) == (FULL if disp == "returned" else {k: 0.0 for k in ADD})     # exactly once: returned in full, or consumed and gone
    assert proto.statistics_accounting()["balanced"]


def test_old_episode_stats_all_keys_end_up_in_discarded_mass(cfg):
    c, srv, proto, env = _start(cfg, agg_buffer_size=99, agg_timeout_s=1e9, max_buffer_age_s=10.0)
    run(proto, env, 0, 1)
    c.steps_since = 0
    run(proto, env, 1, 3)
    env.new_episode()
    run(proto, env, 3, 40)
    assert proto.stats_disposition["discarded_episode"] == 1
    assert all(proto.stats_mass["discarded_episode"][k] == FULL[k] for k in ADD) and all(env.win[k][0] == 0.0 for k in ADD)


# ------------------------------------------------------------------------------------------------ logging and summaries
def _two_client_round(cfg, stats0, stats1):
    cs, srv, proto, env = build(cfg, 2, agg_buffer_size=2)
    for c, st in zip(cs, (stats0, stats1)):
        perturb(c)
        c.steps_since, c.trans_since = 5, 5
        for k, v in st.items():
            env.win[k][c.id] = v
    run(proto, env, 0, 80)
    return srv


def test_server_logs_status_per_update_aligned_with_rho(cfg):
    srv = _two_client_round(cfg, FULL, {"owned_loss_mbit": 2.0})
    assert len(srv.history) >= 1
    r = srv.history[0]
    assert len(r["relevance_status"]) == len(r["clients"]) == len(r["rho_sem"]) == len(r["rho_svc"]) == len(r["relevance"])
    for st, rs, rv in zip(r["relevance_status"], r["rho_sem"], r["rho_svc"]):
        assert st["rho_sem_defined"] == (rs is not None) and st["rho_svc_defined"] == (rv is not None)
    by_client = dict(zip(r["clients"], r["relevance_status"]))
    assert by_client[0]["sem"] == SEM_MATURE_EVIDENCE and by_client[1]["sem"] == SEM_NO_COLLECTION
    assert by_client[1]["svc"] == SVC_EVIDENCE and by_client[1]["svc_zero_with_evidence"]
    json.dumps(r["relevance_status"])


def test_round_summary_is_consistent_and_reproducible(cfg):
    srv = _two_client_round(cfg, FULL, {"owned_loss_mbit": 2.0})
    s1, s2 = summarize_relevance_status(srv.history), summarize_relevance_status(json.loads(json.dumps(srv.history, default=float)))
    assert s1 == s2 and s1["available"] and s1["counts_consistent"]
    assert sum(s1["semantic_status_counts"].values()) == sum(s1["service_status_counts"].values()) == s1["n_client_updates"]
    assert diagnose_rounds(srv.history, cfg["federated"])["relevance_status_summary"] == s1


def test_summary_counts_rounds_without_evidence_and_unavailable_history():
    z = {"sem": SEM_NO_COLLECTION, "svc": SVC_NO_FLOW, "rho_sem_defined": False, "rho_svc_defined": False, "svc_zero_with_evidence": False}
    e = {"sem": SEM_NO_COLLECTION, "svc": SVC_EVIDENCE, "rho_sem_defined": False, "rho_svc_defined": True, "svc_zero_with_evidence": True}
    h = [{"clients": [0, 1], "relevance": [0.0, 0.0], "relevance_status": [z, e]},
         {"clients": [0, 1], "relevance": [0.0, 0.0], "relevance_status": [z, z]},
         {"clients": [0], "relevance": [0.3], "relevance_status": [z]}]
    s = summarize_relevance_status(h)
    assert s["multi_client_rounds"] == 2 and s["multi_rounds_every_relevance_zero"] == 2 and s["multi_rounds_with_distinct_relevance"] == 0
    assert s["multi_rounds_no_component_evidence_for_any_client"] == 1 and s["service_zero_with_evidence"] == 1 and s["counts_consistent"]
    assert summarize_relevance_status([{"clients": [0], "relevance": [0.0]}])["available"] is False           # older history: never guessed
    assert summarize_relevance_status([])["available"] is False


def test_summary_refuses_records_with_mismatched_array_lengths():
    ok = {"sem": SEM_NO_COLLECTION, "svc": SVC_NO_FLOW, "rho_sem_defined": False, "rho_svc_defined": False, "svc_zero_with_evidence": False}
    good = {"round": 1, "clients": [0, 1], "relevance": [0.0, 0.0], "relevance_status": [ok, ok]}
    for name, bad in {"status_short": {"round": 2, "clients": [0, 1], "relevance": [0.0, 0.0], "relevance_status": [ok]},
                      "relevance_short": {"round": 2, "clients": [0, 1], "relevance": [0.0], "relevance_status": [ok, ok]},
                      "clients_long": {"round": 2, "clients": [0, 1, 2], "relevance": [0.0, 0.0], "relevance_status": [ok, ok]}}.items():
        s = summarize_relevance_status([good, bad])
        assert s["available"] is False and s["n_inconsistent_rounds"] == 1 and "inconsistent_records" in s["reason"], name
        assert s["inconsistent_rounds"][0]["round_index"] == 1 and s["inconsistent_rounds"][0]["round"] == 2
        assert "semantic_status_counts" not in s and "n_client_updates" not in s                   # no misleading partial counts
    assert summarize_relevance_status([good])["available"] is True
