import numpy as np
from fed_matd3_iout.communication.contact_scheduler import ContactScheduler
from fed_matd3_iout.communication.model_transfer import ModelTransfer
from fed_matd3_iout.communication.packet_delivery import LinkState


def L(up=True, ok=True, rate=0.05, prop=0.5):
    return LinkState(up, not up, up, ok, 50.0, rate, 0.9, prop)


def test_no_start_when_link_down():
    s = ContactScheduler(2, 1.0, True)
    assert not s.may_start(0, 1000, L(False), 0.5)
    assert s.may_start(0, 1000, L(True), 0.5)


def test_defers_when_update_does_not_fit_window():
    s = ContactScheduler(1, 1.0, allow_fragmentation=False)
    for _ in range(10): s.observe([L(True)])
    s.observe([L(False)])                                      # window of 10 s closed
    assert s.expected_window[0] == 10.0
    for _ in range(2): s.observe([L(True)])
    big = 0.05 * 1e6 / 8 * 0.5 * 60                            # needs 60 s airtime > ~8 s remaining
    assert not s.may_start(0, big, L(True), 0.5) and s.deferred[0] == 1
    assert s.may_start(0, 0.05 * 1e6 / 8 * 0.5 * 3, L(True), 0.5)


def test_transfer_needs_multiple_steps_and_propagation():
    tr = ModelTransfer("upload", 3125.0 * 4)                   # 4 s of airtime at 0.05 Mbit/s * 0.5 share
    t = 0.0
    for k in range(3):
        tr.step(t, 1.0, L(), 0.5, True); t += 1
        assert not tr.done_tx                                  # NOT a successful transfer just because the link is open
    tr.step(t, 1.0, L(prop=2.0), 0.5, True)
    assert tr.done_tx and tr.t_arrival == tr.t_tx_done + 2.0


def test_fading_loss_and_window_loss():
    tr = ModelTransfer("upload", 100000.0)
    tr.step(0, 1, L(ok=False), 0.5, True)
    assert tr.progress_bytes == 0 and tr.failures == 1 and tr.attempts == 1
    tr.step(1, 1, L(), 0.5, True); p = tr.progress_bytes
    assert p > 0
    tr.step(2, 1, L(up=False), 0.5, True)
    assert tr.progress_bytes == p                              # fragmentation: progress kept
    tr2 = ModelTransfer("upload", 100000.0)
    tr2.step(0, 1, L(), 0.5, False); tr2.step(1, 1, L(up=False), 0.5, False)
    assert tr2.progress_bytes == 0 and tr2.resets == 1         # no fragmentation: restart
