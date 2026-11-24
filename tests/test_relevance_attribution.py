"""Regression tests 13-16: batch-level semantic relevance attribution (audit item 4)."""
import math

import pytest

from fed_matd3_iout.federated.relevance import relevance, rho_sem
from fed_matd3_iout.semantic.attribution import BatchBook
from fed_matd3_iout.semantic.packet import Batch

MU = 0.01                                    # H = ln(1/0.05)/0.01 = 299.6 s
H = math.log(1 / 0.05) / MU


def batch(pid, t_collect, units=10.0, sigma=0.5, c=2.0, mu=MU, auv=0):
    return Batch(pid, 0, t_collect, units, sigma, "x", mu, c, "onboard", t_collect, auv)


def deliver(book, auv, b, take, t_deliver):
    """Deliver ``take`` units of b; utility follows U = d c sigma exp(-mu age) (age measured from collection here)."""
    head, rest = (b.split(take) if take < b.units else (b, None))
    u = head.units * head.c * head.sigma * math.exp(-head.mu * (t_deliver - head.t_collect))
    book.on_deliver(auv, head, u)
    return u, rest


# ----------------------------------------------------------------------------- 13
def test_delayed_delivery_attributed_to_original_batch_and_client():
    bk = BatchBook(2)
    b0, b1 = batch(1, 10.0, auv=0), batch(2, 12.0, auv=1)
    bk.on_collect(0, b0); bk.on_collect(1, b1)
    u, _ = deliver(bk, 0, b0, 10.0, t_deliver=70.0)                      # AUV0's batch delivered 60 s later
    c0, c1 = bk.claim(0, 100.0), bk.claim(1, 100.0)
    assert c0["U_mature"] == pytest.approx(u) and c0["V_mature"] == pytest.approx(10 * 2 * 0.5)
    assert c1["U_mature"] == 0.0 and c1["n_mature_batches"] == 0          # AUV1 gets nothing from AUV0's delivery
    assert c1["n_immature_batches"] == 1                                  # its own batch is still immature (not failed)


# ----------------------------------------------------------------------------- 14
def test_batch_collected_in_one_window_delivered_in_another():
    bk = BatchBook(1)
    b = batch(1, 10.0)
    bk.on_collect(0, b)
    w1 = bk.claim(0, 30.0)                                               # update 1 (collection window): not mature yet
    assert w1["V_mature"] == 0.0 and w1["n_immature_batches"] == 1
    head, rest = b.split(4.0)
    u1 = 4 * 2 * 0.5 * math.exp(-MU * 50)
    bk.on_deliver(0, head, u1)                                            # partial delivery in a later window ...
    u2, _ = deliver(bk, 0, rest, 6.0, t_deliver=120.0)                    # ... remainder even later
    w2 = bk.claim(0, 130.0)                                              # update 2 claims the WHOLE batch exactly once
    assert w2["V_mature"] == pytest.approx(10.0) and w2["U_mature"] == pytest.approx(u1 + u2)
    assert w2["units_del_mature"] == pytest.approx(10.0) and w2["n_mature_batches"] == 1
    w3 = bk.claim(0, 140.0)
    assert w3["V_mature"] == 0.0 and w3["n_mature_batches"] == 0          # never counted twice
    assert 0.0 <= rho_sem(w2) <= 1.0


def test_late_delivery_after_claim_by_age_is_not_reattributed():
    bk = BatchBook(1)
    b = batch(1, 0.0)
    bk.on_collect(0, b)
    c = bk.claim(0, H + 1.0)                                             # matured by age, undelivered
    assert c["V_mature"] > 0 and c["U_mature"] == 0.0
    assert bk.on_deliver(0, b, 5.0) == "late" and bk.late_units[0] == b.units
    assert bk.claim(0, H + 50.0)["V_mature"] == 0.0                      # claimed value is not re-counted
    with pytest.raises(KeyError):
        bk.on_deliver(0, batch(99, 1.0), 1.0)                            # never-collected data cannot be delivered


# ----------------------------------------------------------------------------- 15
def test_incomplete_batches_not_treated_as_failed_completed_batches():
    bk = BatchBook(1)
    b = batch(1, 10.0)
    bk.on_collect(0, b)
    c = bk.claim(0, 20.0)                                                # nothing delivered yet, but still young
    assert c["V_mature"] == 0.0 and c["n_immature_batches"] == 1
    assert rho_sem(c) is None                                            # undefined -- NOT zero relevance
    assert relevance(c | {"coll_mbit": 3.0, "owned_loss_mbit": 1.0}, "s3a", 0.5) == pytest.approx(0.75)   # service part only
    assert bk.pending(0)["n"] == 1                                       # stays open for a later update
    c2 = bk.claim(0, 10.0 + H + 1.0)                                     # only after the horizon is it a genuine failure
    assert c2["V_mature"] == pytest.approx(10.0) and rho_sem(c2) == 0.0


def test_episode_reset_censors_instead_of_failing():
    bk = BatchBook(1)
    bk.on_collect(0, batch(1, 1.0))
    bk.reset_episode()
    assert bk.pending(0)["n"] == 0 and bk.censored_V == pytest.approx(10.0)
    assert bk.claim(0, 1e6)["V_mature"] == 0.0                           # a censored batch is never attributed anywhere


# ----------------------------------------------------------------------------- 16
def test_relevance_bounded_and_meaningful():
    bk = BatchBook(1)
    t = 0.0
    batches = [batch(i, 5.0 * i, units=8.0, sigma=0.2 + 0.1 * i, c=1.0 + i % 3) for i in range(1, 7)]
    for b in batches:
        bk.on_collect(0, b)
    for k, b in enumerate(batches):                                      # deliveries: all / half / none, at various delays
        if k % 3 == 0:
            deliver(bk, 0, b, b.units, t_deliver=b.t_collect + 20.0 * (k + 1))
        elif k % 3 == 1:
            deliver(bk, 0, b, b.units / 2, t_deliver=b.t_collect + 40.0)
    c = bk.claim(0, 1e5)
    r = rho_sem(c)
    assert 0.0 < r < 1.0                                                 # partial delivery => strictly inside, no clipping
    assert c["U_mature"] <= c["V_mature"] + 1e-9
    full = BatchBook(1); b = batch(1, 0.0); full.on_collect(0, b); deliver(full, 0, b, b.units, 0.0)
    assert rho_sem(full.claim(0, 1.0)) == pytest.approx(1.0)             # perfect, instant delivery => exactly 1
    for kind in ("semantic", "service", "s3a", "fedavg"):
        for beta in (0.0, 0.3, 1.0):
            assert 0.0 <= relevance(c | {"coll_mbit": 4.0, "owned_loss_mbit": 1.0}, kind, beta) <= 1.0
    with pytest.raises(ValueError):                                      # inconsistent ledger is an error, never clipped
        rho_sem({"U_mature": 5.0, "V_mature": 1.0})
    assert relevance({"U_mature": 0.0, "V_mature": 4.0}, "semantic", 0.5) == 0.0   # a mature total failure is a real 0
    # semantics-off ablation uses the units of the same mature batches
    assert rho_sem(c, use_semantics=False) == pytest.approx(c["units_del_mature"] / c["units_coll_mature"])
