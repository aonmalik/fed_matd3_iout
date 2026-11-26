"""Eight deterministic adverse-condition experiments for the federated update lifecycle.

Each scenario returns ``{"name", "checks": {description: bool}, "details": {...}}``.  ``checks`` always includes the global
invariants of ``Harness.check_invariants`` (prefixed ``inv:``) evaluated at the END of the scenario, plus scenario-specific
checks on model references, update counters and statistics.  Nothing here is random: links are scripted step by step.
"""
from __future__ import annotations

import numpy as np
import torch

from ..agents.actor import flatten_params, load_flat_params
from ..federated.update_packet import ControlMessage, UpdateStatus
from .fixtures import Harness, link, perturb, ready

S = {"coll_mbit": 5.0, "V_mature": 4.0}


def _finish(name, h, checks, details=None):
    inv = h.check_invariants()
    checks = dict(checks)
    checks.update({f"inv:{k}": v for k, v in inv.items()})
    return {"name": name, "passed": all(checks.values()), "checks": checks, "details": details or {}}


def _log(h):
    return {u["uid"]: u for u in h.proto.update_log}


# --------------------------------------------------------------------------------------------- 1
def delayed_upload(cfg):
    """Client 1 completes while client 0 is out of contact; client 0's slow upload then arrives stale but acceptable."""
    h = Harness(cfg, 2, "async", agg_buffer_size=1, max_staleness=8)
    c0, c1 = h.clients
    perturb(c0); perturb(c1, 0.2); ready(c0); ready(c1)
    h.env.links[0] = link(up=False)                                    # c0 out of range
    h.env.links[1] = link(rate=h.rate_for_steps(3))
    h.run(20)
    ref0 = c0.reference.clone()
    ch = {"c1 committed while c0 was away": h.server.version == 1 and h.proto.n_status["aggregated"] == 1,
          "c0 created no update while out of contact": c0.outstanding is None and h.proto.n_created == 1,
          "c0 reference untouched while away": torch.equal(c0.reference, ref0)}
    h.env.links[0] = link(rate=h.rate_for_steps(6))                    # contact returns: slow, multi-step upload
    h.env.links[1] = link(up=False)
    h.run_until(lambda: h.proto.n_status["aggregated"] == 2)
    h.run(30)
    u0 = h.pkts["c0-u1"]
    rec = h.server.history[-1]
    ch.update({"delayed update aggregated": u0.status is UpdateStatus.AGGREGATED,
               "recorded staleness is server_version - base_version (>=1)": rec["staleness"] == [h.server.version - 1 - u0.base_version]
               and rec["staleness"][0] >= 1,
               "stale factor < 1 applied": rec["stale_factor"][0] < 1.0,
               "client resolved exactly once": u0.client_resolved and c0.stats["commits"] == 1,
               "upload took several steps": u0.t_received - u0.t_upload_start >= 3.0})
    return _finish("delayed_upload", h, ch, {"staleness": rec["staleness"], "latency_s": u0.t_received - u0.t_upload_start})


# --------------------------------------------------------------------------------------------- 2
def stale_rejection(cfg):
    """An update that becomes older than max_staleness is rejected; its snapshot never becomes the reference."""
    h = Harness(cfg, 2, "async", agg_buffer_size=1, max_staleness=1)
    c0, c1 = h.clients
    perturb(c0); ready(c0)
    h.env.inject(0, **S)
    h.env.links[0] = link(rate=h.rate_for_steps(8))
    h.run(2)                                                           # c0 upload started (snapshot taken)
    c0.steps_since = 0                                                 # (no immediate retry: isolate the first update)
    h.env.links[0] = link(up=False)                                    # contact lost mid-upload
    for k in range(3):                                                 # meanwhile the USV advances 3 versions via c1
        perturb(c1, 0.05); ready(c1)
        h.env.links[1] = link(rate=h.rate_for_steps(2))
        h.run_until(lambda k=k: h.server.version == k + 1, 100)
        h.run_until(lambda: h.clients[1].outstanding is None, 100)     # c1 must hear its commit before its next update
    h.env.links[1] = link(up=False)
    h.env.links[0] = link(rate=h.rate_for_steps(2))
    h.run(60)
    u0 = h.pkts["c0-u1"]
    ch = {"stale update rejected": u0.status is UpdateStatus.REJECTED and u0.reason in ("stale_at_receipt", "stale_at_aggregation"),
          "reference is NOT the rejected snapshot": not torch.equal(c0.reference, h.snapshots[u0.uid]),
          "rejected update not aggregated": h.server.n_accepted == 3 and u0.uid not in {f"c{c}-u{u}" for r in h.server.history
                                                                                      for c, u in zip(r["clients"], r["updates"])},
          "failure counted once": c0.stats["failed"] == 1 and h.proto.n_status["rejected"] == 1,
          "statistics returned to the window exactly once": u0.stats_disposition == "returned"
          and h.env.win["coll_mbit"][0] == S["coll_mbit"]}
    return _finish("stale_rejection", h, ch, {"reason": u0.reason})


# --------------------------------------------------------------------------------------------- 3
def server_rejection(cfg):
    """A corrupted (non-finite) update is refused at receipt: the global actor stays finite and unchanged."""
    h = Harness(cfg, 2, "async", agg_buffer_size=1)
    c0 = h.clients[0]
    perturb(c0); ready(c0)
    with torch.no_grad():
        next(c0.agent.actor.parameters()).view(-1)[0] = float("nan")      # client-side corruption
    h.env.inject(0, **S)
    theta0 = h.server.theta.clone()
    ref0 = c0.reference.clone()
    h.run(1); h.hold()                                                 # first update created; no retry loop in this scenario
    h.run(40)
    u = h.pkts["c0-u1"]
    ch = {"rejected at receipt as non_finite": u.status is UpdateStatus.REJECTED and u.reason == "non_finite",
          "global actor unchanged and finite": torch.equal(h.server.theta, theta0) and bool(torch.isfinite(h.server.theta).all()),
          "no aggregation round": h.server.version == 0,
          "client reference unchanged": torch.equal(c0.reference, ref0),
          "client learned the outcome (unblocked), counters retained": c0.outstanding is None and c0.stats["failed"] == 1
          and c0.steps_since == 5,
          "statistics returned exactly once": u.stats_disposition == "returned" and h.env.win["coll_mbit"][0] == S["coll_mbit"]}
    h.clients[0].agent.actor  # (the corrupted local actor is outside the federation's responsibility; reference is the clean one)
    inv_skip = {"all_parameters_finite"}
    out = _finish("server_rejection", h, ch)
    for k in inv_skip:                                                  # local actor intentionally NaN in this scenario
        out["checks"].pop(f"inv:{k}", None)
    out["passed"] = all(out["checks"].values())
    return out


# --------------------------------------------------------------------------------------------- 4
def update_expiration(cfg):
    """(a) upload timeout (client side) and (b) buffer expiry (USV side): nothing committed, nothing lost, client unblocked."""
    res = {}
    # (a) upload never completes (fading), client abandons it
    h = Harness(cfg, 1, "async", upload_timeout_s=5.0)
    c = h.clients[0]; perturb(c); ready(c); h.env.inject(0, **S)
    ref0 = c.reference.clone()
    h.env.links[0] = link(success=False)
    h.run(1); h.hold()
    h.run(30)
    u = h.pkts["c0-u1"]
    ch = {"(a) upload timeout -> EXPIRED": u.status is UpdateStatus.EXPIRED and u.reason == "upload_timeout",
          "(a) reference unchanged": torch.equal(c.reference, ref0) and c.steps_since == 5 + 0,
          "(a) statistics returned once": h.env.win["coll_mbit"][0] == S["coll_mbit"] and u.stats_disposition == "returned"}
    # the client is unblocked and its next update carries the retained progress against the SAME reference
    perturb(c, 0.05)                                                   # further local training before the retry
    h.hold(False); c.steps_since = 9; h.env.links[0] = link()
    h.run(30)
    if c.retry is not None:                                            # protocol >= 2.1: the retry is DEFERRED, not lost
        ch["(a) retry deferred inside the back-off window"] = (h.proto.n_created == 1 and c.consecutive_failures == 1
                                                               and c.stats["retry_deferred"] == 1 and c.upload is None)
        h.run(int(c.retry_not_before - h.t) + 40)
    u2 = h.pkts.get("c0-u2")
    ch["(a) client unblocked: new update created and aggregated"] = u2 is not None and u2.status is UpdateStatus.AGGREGATED
    ch["(a) new delta measured against the unchanged reference"] = u2 is not None and torch.allclose(h.deltas["c0-u2"], h.snapshots["c0-u2"] - ref0)
    ch["(a) stats of both updates settled once each"] = len(h.proto.update_log) == 2
    inv_a = h.check_invariants()
    # (b) USV expires a buffered update
    h2 = Harness(cfg, 1, "async", agg_buffer_size=99, agg_timeout_s=1e9, max_buffer_age_s=10.0)
    c = h2.clients[0]; perturb(c); ready(c); h2.env.inject(0, **S)
    ref0 = c.reference.clone()
    h2.run(1); h2.hold()
    h2.run(40)
    u = h2.pkts["c0-u1"]
    ch.update({"(b) buffer expiry -> EXPIRED": u.status is UpdateStatus.EXPIRED and u.reason == "expired_in_buffer",
               "(b) never aggregated, version unchanged": h2.server.n_accepted == 0 and h2.server.version == 0,
               "(b) reference unchanged, counters retained": torch.equal(c.reference, ref0) and c.steps_since == 5,
               "(b) statistics returned once": h2.env.win["coll_mbit"][0] == S["coll_mbit"] and u.stats_disposition == "returned",
               "(b) client informed over the downlink": c.stats["failed"] == 1 and c.outstanding is None})
    inv_b = h2.check_invariants()
    ch.update({f"inv(a):{k}": v for k, v in inv_a.items()})
    ch.update({f"inv(b):{k}": v for k, v in inv_b.items()})
    return {"name": "update_expiration", "passed": all(ch.values()), "checks": ch, "details": {}}


# --------------------------------------------------------------------------------------------- 5
def duplicate_delivery(cfg):
    """The same in-flight upload is delivered twice (and a second duplicate arrives after aggregation)."""
    h = Harness(cfg, 1, "async", agg_buffer_size=1)
    c = h.clients[0]; perturb(c); ready(c, 7); h.env.inject(0, **S)
    h.run_until(lambda: bool(h.proto.inflight))
    tr, i = h.proto.inflight[0]
    h.proto.inflight.append((tr, i))                                   # duplicate delivery of the same transfer
    h.run(10)
    late = h.pkts["c0-u1"]
    h.proto.inflight.append((tr, i))                                   # and a third copy after the outcome
    h.run(10)
    commits = [m for m in c.stats.items() if m[0] == "commits"]
    ch = {"received once": h.server.n_received == 1, "duplicates ignored (2)": h.server.n_duplicates == 2,
          "aggregated exactly once": h.server.n_accepted == 1 and h.server.version == 1 and len(h.server.history) == 1,
          "global moved by exactly one weighted delta": torch.allclose(h.server.theta, h.theta_before[1] + h.deltas["c0-u1"], atol=1e-6),
          "one commit applied by the client": c.stats["commits"] == 1 and late.client_resolved,
          "statistics consumed once": h.proto.stats_disposition["consumed"] == 1 and h.proto.stats_mass["consumed"]["coll_mbit"] == S["coll_mbit"],
          "no window refill": h.env.win["coll_mbit"][0] == 0.0}
    return _finish("duplicate_delivery", h, ch)


# --------------------------------------------------------------------------------------------- 6
def client_reconnection(cfg):
    """Contact lost mid-upload (fragmentation keeps progress) and again while waiting for the commit."""
    h = Harness(cfg, 1, "async", agg_buffer_size=2, allow_fragmentation=True)    # K=2: the update waits in the buffer
    c = h.clients[0]; perturb(c); ready(c, 6); h.env.inject(0, **S)
    ref0 = c.reference.clone()
    rate = h.rate_for_steps(6)
    h.env.links[0] = link(rate=rate)
    h.run(3)
    prog_before = c.upload.progress_bytes if c.upload is not None else None
    h.env.links[0] = link(up=False, rate=rate)                         # outage mid-upload
    h.run(15)
    ch = {"progress persists across the outage": c.upload is not None and c.upload.progress_bytes == prog_before and prog_before > 0,
          "no second update during the outage": h.proto.n_created == 1}
    h.env.links[0] = link(rate=rate)
    h.run_until(lambda: h.pkts["c0-u1"].receipt_acked, 100)            # upload completes; the transport ACK arrives ALONE
    ch["receipt received while update still BUFFERED"] = h.pkts["c0-u1"].status is UpdateStatus.BUFFERED
    ch["transport ACK alone does not advance the reference"] = torch.equal(c.reference, ref0) and c.outstanding is not None
    h.cfg.federated["agg_buffer_size"] = 1                              # now let the USV aggregate ...
    h.step()
    # ... and lose contact again BEFORE the commit has been delivered: client must still be waiting
    h.env.links[0] = link(up=False, rate=rate)
    h.run(10)
    u = h.pkts["c0-u1"]
    ch.update({"aggregated at the USV while client offline": u.status is UpdateStatus.AGGREGATED,
               "client has NOT learned the outcome": not u.client_resolved and c.outstanding is u,
               "reference not advanced while offline": torch.equal(c.reference, ref0),
               "no new update while commit pending": h.proto.n_created == 1,
               "control messages are queued, not lost": len(h.server.outbox[0]) >= 1 or c.ctrl is not None})
    ready(c, 50)                                                       # lots of new local progress, still blocked
    h.run(5)
    ch["still blocked by unresolved update"] = h.proto.n_created == 1
    h.env.links[0] = link(rate=rate)                                   # reconnection
    h.run_until(lambda: u.client_resolved, 200)
    ch.update({"commit delivered after reconnection": u.client_resolved and c.stats["commits"] == 1,
               "reference now the committed snapshot or the installed global": any(
                   torch.equal(c.reference, a) for a in (h.snapshots[u.uid], h.thetas[1]))})
    return _finish("client_reconnection", h, ch)


# --------------------------------------------------------------------------------------------- 7
def asynchronous_aggregation(cfg):
    """Three clients with different contact times; buffer K=2. Re-derive every aggregation independently."""
    h = Harness(cfg, 3, "async", agg_buffer_size=2, agg_timeout_s=15.0, max_staleness=8, max_client_weight=0.8, clip_norm=None)
    for i, c in enumerate(h.clients):
        perturb(c, 0.05 * (i + 1)); ready(c, 5 + i)
        h.env.inject(i, coll_mbit=3.0 + i, V_mature=2.0)
        h.env.links[i] = link(rate=h.rate_for_steps(2 + 3 * i))
    h.run(10)
    for rnd in range(3):
        for i, c in enumerate(h.clients):
            perturb(c, 0.01); ready(c, 6 + i)
        h.run(40)
    ok_formula, ok_w, ok_stale, versions = True, True, True, []
    f = h.cfg.federated
    for rec in h.server.history:
        v = rec["round"]
        versions.append(v)
        d = [h.deltas[f"c{c}-u{u}"] for c, u in zip(rec["clients"], rec["updates"])]
        w = torch.tensor(rec["w"], dtype=torch.float32)
        expect = h.theta_before[v] + float(f["server_lr"]) * sum(wi * di for wi, di in zip(w, d))
        ok_formula &= bool(torch.allclose(h.thetas[v], expect, atol=1e-5))
        cap = max(float(f["max_client_weight"]), 1.0 / len(rec["w"]))     # a cap below 1/|R| is infeasible (documented)
        ok_w &= abs(sum(rec["w"]) - 1.0) < 1e-9 and max(rec["w"]) <= cap + 1e-9 and min(rec["w"]) > 0
        ok_stale &= all(0 <= s <= int(f["max_staleness"]) for s in rec["staleness"])
    ch = {"several rounds happened": h.server.version >= 2,
          "versions strictly consecutive": versions == list(range(1, len(versions) + 1)),
          "theta' = theta + eta_g sum w_i Delta_i (independent recomputation)": ok_formula,
          "weights positive, sum to 1, respect the cap": ok_w,
          "recorded staleness within bounds": ok_stale,
          "every aggregated update appears in exactly one round": (lambda l: len(l) == len(set(l)))(
              [(c, u) for r in h.server.history for c, u in zip(r["clients"], r["updates"])])}
    return _finish("asynchronous_aggregation", h, ch, {"rounds": h.server.version, "avg_staleness": h.proto.summary()["avg_staleness"]})


# --------------------------------------------------------------------------------------------- 8
def episode_transition(cfg):
    """Episode boundary with unresolved updates: buffered / in-flight / uploading updates keep their OWN episode's statistics."""
    h = Harness(cfg, 3, "async", agg_buffer_size=99, agg_timeout_s=1e9, max_buffer_age_s=30.0)
    for i, c in enumerate(h.clients):
        perturb(c, 0.05 * (i + 1)); ready(c)
        h.env.inject(i, **S)
    h.env.links[0] = link(rate=h.rate_for_steps(2))                    # c0: delivered and buffered before the boundary
    h.env.links[1] = link(rate=h.rate_for_steps(20))                   # c1: still uploading at the boundary
    h.env.links[2] = link(up=False)                                    # c2: not yet started (stats stay in the window)
    h.run(8)
    states = [(h.pkts[f"c{i}-u1"].status.value if f"c{i}-u1" in h.pkts else None) for i in range(3)]
    h.env.new_episode()                                                # ---- episode boundary ----
    for i in range(3):
        h.env.inject(i, coll_mbit=1.0, V_mature=1.0)                   # fresh data of episode 2
    pre = {k: float(h.env.win[k].sum()) for k in ("coll_mbit", "V_mature")}
    h.env.links[1] = link(rate=h.rate_for_steps(2))
    h.run(80)                                                          # old updates expire / complete in episode 2
    log = _log(h)
    old = [u for u in h.proto.update_log if u["episode"] == 1]
    ch = {"state at boundary: c0 buffered, c1 uploading, c2 not started": states == ["buffered", "uploading", None],
          "all episode-1 updates ended with disposition discarded_episode": all(u["disposition"] == "discarded_episode" for u in old)
          and len(old) == 2,
          "episode-1 statistics did not leak into the new window": float(h.env.win["coll_mbit"].sum()) == pre["coll_mbit"]
          or h.env.win["coll_mbit"].sum() <= pre["coll_mbit"] + 1e-12,
          "refusals counted": h.env.stats_refused_episode == 2,
          "episode-2 updates (created after the boundary) carry episode-2 stats only": all(
              q.episode_id == 2 and q.stats.get("coll_mbit", 0.0) <= 1.0 + 1e-12 for q in h.pkts.values() if q.t_created >= 8)}
    return _finish("episode_transition", h, ch, {"log": [(k, v["status"], v["disposition"]) for k, v in log.items()]})


def retry_backoff_impact(cfg, horizon=600, heal_at=100) -> dict:
    """MEASURED effect of the back-off (both runs deterministic, same fault) on bytes, airtime, energy and recovery latency.
    (A) persistent fault: client 0's update is non-finite for the whole horizon -> every upload is rejected at receipt.
    (B) transient fault: the same fault heals at t=heal_at -> time until the client's first COMMIT.
    Energy = measured transmit airtime x energy.p_tx_elec_w (uncalibrated assumption of the simulator, see environment.yaml)."""
    p_tx = float(cfg.energy.p_tx_elec_w)
    out = {"horizon_s": horizon, "heal_at_s": heal_at, "p_tx_elec_w": p_tx, "runs": {}}
    for label, retry in (("no_backoff_protocol_2_0", None),
                         ("backoff_protocol_2_1", {k: float(v) if k != "enabled" else bool(v) for k, v in dict(cfg.federated.retry).items()})):
        h = Harness(cfg, 2, "async", retry=retry, agg_buffer_size=1)
        c = h.clients[0]
        perturb(c); ready(c); h.env.inject(0, **S)
        good = flatten_params(c.agent.actor).clone()
        with torch.no_grad():
            next(c.agent.actor.parameters()).view(-1)[0] = float("nan")
        t_commit = None
        for t in range(horizon):
            if t == heal_at:                                               # the fault clears
                load_flat_params(c.agent.actor, good + 0.01)
            h.step()
            if t_commit is None and c.stats["commits"] >= 1:
                t_commit = t
        s = h.proto.summary()
        out["runs"][label] = {"uploads": h.proto.n_created, "rejected": s["rejected_updates"], "rejected_bytes": s["rejected_bytes"],
                              "uplink_bytes": c.stats["bytes_up"], "tx_airtime_s": c.stats["tx_s"], "tx_energy_j": c.stats["tx_s"] * p_tx,
                              "retry_attempts": s["retry_attempts"], "retry_deferred_updates": s["retry_deferred_updates"],
                              "first_commit_t_s": t_commit, "recovery_latency_after_heal_s": None if t_commit is None else t_commit - heal_at,
                              "invariants_hold": all(v for k, v in h.check_invariants().items() if k != "all_parameters_finite")}
    a, b = out["runs"]["no_backoff_protocol_2_0"], out["runs"]["backoff_protocol_2_1"]
    out["bytes_ratio_backoff_over_none"] = b["uplink_bytes"] / max(a["uplink_bytes"], 1e-12)
    out["energy_saved_j"] = a["tx_energy_j"] - b["tx_energy_j"]
    out["extra_recovery_latency_s"] = None if None in (a["first_commit_t_s"], b["first_commit_t_s"]) else b["first_commit_t_s"] - a["first_commit_t_s"]
    out["interpretation"] = ("back-off trades a bounded extra recovery latency after a transient failure (at most the current back-off delay, "
                             "<= federated.retry.max_s) for a large cut in wasted full-model uploads and transmit energy under persistent rejection; "
                             "it never changes the update lifecycle, the reference model or the progress counters")
    return out


SCENARIOS = {f.__name__: f for f in (delayed_upload, stale_rejection, server_rejection, update_expiration, duplicate_delivery,
                                     client_reconnection, asynchronous_aggregation, episode_transition)}


def run_all(cfg) -> dict:
    return {name: fn(cfg) for name, fn in SCENARIOS.items()}


def training_probe(cfg, episodes=3, method="s3a_fedmatd3", seed=0, with_diagnostics=False) -> dict:
    """Short REAL training at the given configuration (default = main protocol scale per episode). Measures cost and
    communication/aggregation behaviour. Not a performance result: no comparison, no evaluation of policy quality."""
    import time

    from ..baselines.registry import build_cfg
    from ..training.trainer import Trainer
    c = build_cfg(cfg, method)
    t0 = time.time()
    t = Trainer(c, torch.device("cpu"), None, seed)
    t.train(episodes)
    wall = time.time() - t0
    s = t.proto.summary()
    hist = t.server.history
    sizes = [len(r["clients"]) for r in hist]
    rel = [x for r in hist for x in r["relevance"]]
    w = [x for r in hist for x in r["w"]]
    acc = t.proto.statistics_accounting()
    return {"method": method, "episodes": episodes, "episode_steps": int(c.env.episode_steps), "wall_s": wall,
            "sec_per_episode": wall / episodes, "gradient_steps_total": int(sum(x.total_steps for x in t.clients)),
            "updates_created": s["attempted_updates"], "aggregated": s["aggregated_updates"], "rejected": s["rejected_updates"],
            "expired": s["expired_updates"], "unresolved": s["unresolved_updates"], "stats_discarded_episode": s["stats_discarded_episode"],
            "comm_tx_s_per_upload": s["comm_tx_s"] / max(s["attempted_updates"], 1),
            "tx_energy_j_per_upload_incl_acks": s["comm_tx_s"] / max(s["attempted_updates"], 1) * float(c.energy.p_tx_elec_w),
            "model_bytes_per_upload": s["bytes_up"] / max(s["transmitted_updates"], 1),
            "aggregation_rounds": s["aggregation_rounds"], "single_update_rounds": int(sum(1 for n in sizes if n == 1)),
            "single_update_round_fraction": (sum(1 for n in sizes if n == 1) / len(sizes)) if sizes else float("nan"),
            "avg_staleness": s["avg_staleness"], "link_up_fraction": s["link_up_fraction"], "bytes_total": s["bytes_total"],
            "relevance_min_mean_max": [min(rel), float(np.mean(rel)), max(rel)] if rel else None,
            "weights_min_mean_max": [min(w), float(np.mean(w)), max(w)] if w else None,
            "accounting_balanced": bool(acc["balanced"]),
            "all_params_finite": bool(torch.isfinite(t.server.theta).all() and all(
                torch.isfinite(p).all() for a in t.agents for p in list(a.actor.parameters()) + list(a.critic.parameters()))),
            "return_mean_per_episode": [r["return_mean"] for r in t.curve],
            **({"aggregation_diagnostics": __import__("fed_matd3_iout.validation.aggregation_diagnostics", fromlist=["x"]).diagnose_rounds(
                hist, c.federated), "retry_counters": {k: s[k] for k in s if k.startswith(("retry", "rejected_b", "expired_b", "wasted", "backoff"))}}
               if with_diagnostics else {})}
