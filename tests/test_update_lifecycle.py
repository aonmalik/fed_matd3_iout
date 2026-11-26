"""Regression tests 1-8: federated update lifecycle (audit item 2)."""
import numpy as np
import pytest
import torch

from fed_matd3_iout.federated.update_packet import ControlMessage, IllegalTransition, UpdateStatus
from fed_fixtures import build, link, perturb, run


def _ready(c, steps=5):
    c.steps_since, c.trans_since = steps, steps


# ----------------------------------------------------------------------------- 1
def test_no_second_update_while_first_unresolved(cfg):
    cs, srv, proto, env = build(cfg, 1, agg_buffer_size=99)          # never aggregates => update stays BUFFERED
    c = cs[0]
    perturb(c); _ready(c)
    run(proto, env, 0, 40)
    first = srv.buffer[0]
    assert first.status is UpdateStatus.BUFFERED and c.outstanding is first
    _ready(c, 10 ** 6)                                               # plenty of fresh local progress ...
    run(proto, env, 40, 200)
    assert proto.n_created == 1 and c.update_counter == 1            # ... still no second update
    with pytest.raises(RuntimeError):
        c.make_update(0.0, {})                                       # and the client API itself refuses


# ----------------------------------------------------------------------------- 2
def test_transport_ack_does_not_advance_reference(cfg):
    cs, srv, proto, env = build(cfg, 1, agg_buffer_size=99)
    c = cs[0]
    ref0 = c.reference.clone()
    perturb(c); _ready(c)
    run(proto, env, 0, 40)
    pkt = c.outstanding
    assert pkt.receipt_acked and pkt.status is UpdateStatus.BUFFERED   # transport ACK was received ...
    assert torch.equal(c.reference, ref0)                              # ... reference untouched
    assert c.steps_since == 5 and c.outstanding is pkt                 # counters untouched, still unresolved


# ----------------------------------------------------------------------------- 3
def test_receipt_rejection_does_not_advance_reference(cfg):
    cs, srv, proto, env = build(cfg, 1, max_staleness=0)
    c = cs[0]
    perturb(c); _ready(c)
    ref0 = c.reference.clone()
    pkt = c.make_update(0.0, {"coll_mbit": 1.0}, 1)
    pkt.transition(UpdateStatus.UPLOADING); pkt.transition(UpdateStatus.IN_FLIGHT)
    srv.version = 3                                                    # base 0 -> staleness 3 > 0
    assert srv.receive(pkt, 1.0) is False and pkt.status is UpdateStatus.REJECTED
    for m in srv.pop_outbox(0):
        assert c.handle_control(m) == "reject"
    assert torch.equal(c.reference, ref0) and c.steps_since == 5 and c.outstanding is None
    assert pkt.client_resolved and pkt.reason == "stale_at_receipt"


# ----------------------------------------------------------------------------- 4
def test_aggregation_rejection_does_not_advance_reference(cfg):
    cs, srv, proto, env = build(cfg, 1, agg_buffer_size=1, max_staleness=2)
    c = cs[0]
    perturb(c); _ready(c)
    ref0 = c.reference.clone()
    pkt = c.make_update(0.0, {}, 1)
    pkt.transition(UpdateStatus.UPLOADING); pkt.transition(UpdateStatus.IN_FLIGHT)
    assert srv.receive(pkt, 1.0) and pkt.status is UpdateStatus.BUFFERED
    srv.version = 9                                                    # becomes stale while waiting in the buffer
    assert srv.maybe_aggregate(2.0) is None and pkt.status is UpdateStatus.REJECTED
    kinds = [c.handle_control(m) for m in srv.pop_outbox(0)]
    assert kinds == ["receipt", "reject"]
    assert torch.equal(c.reference, ref0) and c.steps_since == 5 and c.outstanding is None
    assert srv.version == 9 and srv.n_accepted == 0                    # nothing was aggregated


# ----------------------------------------------------------------------------- 5
def test_successful_aggregation_advances_reference_exactly_once(cfg):
    cs, srv, proto, env = build(cfg, 1, agg_buffer_size=1)
    c = cs[0]
    perturb(c); _ready(c, 7)
    ref0 = c.reference.clone()
    pkt = c.make_update(0.0, {}, 1)
    snap = pkt.snapshot.clone()
    _ready(c, 10)                                                      # 3 more steps happen after the snapshot
    pkt.transition(UpdateStatus.UPLOADING); pkt.transition(UpdateStatus.IN_FLIGHT)
    srv.receive(pkt, 1.0)
    assert srv.maybe_aggregate(1.0) is not None and pkt.status is UpdateStatus.AGGREGATED
    msgs = srv.pop_outbox(0)
    assert [m.kind for m in msgs] == ["receipt", "commit"] and msgs[1].server_version == 1
    assert c.handle_control(msgs[0]) == "receipt" and torch.equal(c.reference, ref0)   # receipt alone: no move
    assert c.handle_control(msgs[1]) == "commit"
    assert torch.equal(c.reference, snap) and c.steps_since == 3       # moved once; post-snapshot progress kept
    c.reference = c.reference + 1.0                                    # tamper: a second commit must not re-apply
    assert c.handle_control(msgs[1]) == "duplicate" and torch.equal(c.reference, snap + 1.0)
    assert c.stats["commits"] == 1


# ----------------------------------------------------------------------------- 6
def test_duplicate_terminal_responses_cannot_double_commit(cfg):
    cs, srv, proto, env = build(cfg, 1, agg_buffer_size=2)
    c = cs[0]
    perturb(c); _ready(c)
    pkt = c.make_update(0.0, {}, 1)
    pkt.transition(UpdateStatus.UPLOADING); pkt.transition(UpdateStatus.IN_FLIGHT)
    assert srv.receive(pkt, 1.0) is True
    assert srv.receive(pkt, 1.5) is False and srv.n_duplicates == 1    # duplicate delivery ignored
    assert len(srv.buffer) == 1 and srv.n_received == 1
    cfg.federated.agg_buffer_size = 1
    assert srv.maybe_aggregate(2.0) is not None and srv.version == 1
    with pytest.raises(IllegalTransition):                             # terminal state has no outgoing edge
        pkt.transition(UpdateStatus.AGGREGATED)
    assert srv.receive(pkt, 3.0) is False and srv.n_accepted == 1      # late duplicate cannot be aggregated again
    assert srv.maybe_aggregate(4.0) is None and srv.version == 1


# ----------------------------------------------------------------------------- 7
def test_control_messages_cannot_overwrite_each_other(cfg):
    cs, srv, proto, env = build(cfg, 1)
    c = cs[0]
    m = lambda k, u: ControlMessage(k, f"c0-u{u}", 0, u, 0)
    srv.outbox[0] = [m("receipt", 1), m("commit", 1)]
    env.links[0] = link(success=False)                                 # fading: nothing gets through
    proto.step(env, 0.0)
    first = c.ctrl
    assert first is not None and [x.kind for x in first.payload] == ["receipt", "commit"]
    srv.outbox[0].append(m("receipt", 2))                              # new message while the first transfer is active
    proto.step(env, 1.0)
    assert c.ctrl is first and len(first.payload) == 2                 # not replaced / not mutated
    assert [x.uid for x in srv.outbox[0]] == ["c0-u2"]                 # waits its turn, not lost
    env.links[0] = link(success=True)
    proto.step(env, 2.0)                                               # last bit sent at 2.002 s ...
    proto.step(env, 3.0)                                               # ... arrives (and is processed) on the next step
    assert c.ctrl is not first and c.stats["bytes_down"] == pytest.approx(2 * proto.ack_bytes)   # first list delivered whole
    assert c.stats["unknown_ctrl"] == 2                                # (no outstanding update here => both ignored, none applied)


# ----------------------------------------------------------------------------- 8
@pytest.mark.parametrize("mode", ["async", "sync"])
def test_sync_and_async_modes(cfg, mode):
    cs, srv, proto, env = build(cfg, 2, mode=mode, agg_buffer_size=1, sync_timeout_s=1e9)
    for c in cs:
        perturb(c); _ready(c)
    t = 0
    for rnd in range(3):
        t = run(proto, env, t, t + 60)
        for c in cs:
            perturb(c, 0.01); _ready(c)
    s = proto.summary()
    assert s["aggregation_rounds"] >= 1 and s["aggregated_updates"] >= 2
    assert srv.version == len(srv.history)
    ids = [u["uid"] for u in proto.update_log]
    assert len(ids) == len(set(ids))                                   # unique update ids, each settled once
    acc = proto.statistics_accounting()
    assert acc["balanced"] and acc["created"] == acc["settled"] + acc["unresolved"] + acc["terminal_unsettled"]
    if mode == "sync":
        first = srv.history[0]
        assert sorted(first["clients"]) == [0, 1]                      # barrier: first round waited for both clients


# ----------------------------------------------------------------------------- 9

def test_terminal_outcome_waits_through_control_outage_then_resolves_once(cfg):
    """A server commit is authoritative, but the client stays outstanding until its downlink arrives."""
    cs, srv, proto, env = build(cfg, 1, agg_buffer_size=1)
    c = cs[0]
    perturb(c)
    _ready(c)
    env.inject(0, coll_mbit=5.0, owned_loss_mbit=2.0, U_mature=3.0, V_mature=4.0)

    # Create and transmit one real update, then freeze creation of any subsequent update.
    proto.step(env, 0.0)
    proto.accept_new = False
    pkt = c.outstanding
    assert pkt is not None and pkt.status is UpdateStatus.IN_FLIGHT

    # The server receives and aggregates the update while the link is healthy.
    proto.step(env, 1.0)
    assert pkt.status is UpdateStatus.AGGREGATED
    assert pkt.stats_disposition == "consumed"
    assert not pkt.client_resolved and c.outstanding is pkt
    assert proto.statistics_accounting()["balanced"]
    assert proto.n_status["aggregated"] == 1
    assert len(proto.update_log) == 1

    # The terminal commit is now queued. An outage starts before the client can receive it.
    env.links[0] = link(success=False)
    proto.step(env, 2.0)
    pending = proto.pending_report()
    assert pending["awaiting_terminal_outcome_at_client"] == 1
    assert pending["control_transfers_active"] == 1
    assert c.ctrl is not None and [m.kind for m in c.ctrl.payload] == ["receipt", "commit"]
    assert pending["control_messages_queued"] == 0
    assert proto.drain_active()
    assert not pkt.client_resolved and c.outstanding is pkt
    assert proto.statistics_accounting()["balanced"]

    # Restore contact: the queued/in-progress terminal control message must resolve the same packet.
    env.links[0] = link(success=True)
    for t in range(3, 20):
        proto.step(env, float(t))
        if pkt.client_resolved:
            break

    assert pkt.client_resolved and c.outstanding is None
    assert proto.pending_report()["awaiting_terminal_outcome_at_client"] == 0
    assert not proto.drain_active()
    assert c.stats["commits"] == 1
    assert proto.n_status["aggregated"] == 1
    assert len(proto.update_log) == 1
    assert proto.statistics_accounting()["balanced"]
    assert sum(proto.stats_disposition.values()) == 1

    # Replaying the same terminal control is idempotent and cannot settle or commit twice.
    replay = ControlMessage("commit", pkt.uid, pkt.client_id, pkt.update_id, srv.version, "aggregated")
    assert c.handle_control(replay, 20.0) == "duplicate"
    assert c.stats["commits"] == 1
    assert c.stats["duplicate_ctrl"] == 1
    assert len(proto.update_log) == 1
