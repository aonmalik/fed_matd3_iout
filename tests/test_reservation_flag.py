"""Sensor-uplink reservation flags: a transfer that has sent its last bit and only waits for propagation occupies no airtime and
must NOT reserve model airtime.  Half duplex reserves for any transfer still sending; full duplex only for an upload."""
import pytest

from fed_fixtures import Harness
from fed_matd3_iout.communication.model_transfer import ModelTransfer
from fed_matd3_iout.utils.config import load_config

PROP = 50.0


def _h(duplex):
    cfg = load_config(overrides={"acoustic.usv_link.duplex": duplex})
    h = Harness(cfg, 1, "async")
    h.hold()
    h.set_links(rate=1.0, prop=PROP)
    h.server.version = 1
    return h


@pytest.mark.parametrize("duplex", ["half", "full"])
@pytest.mark.parametrize("kind", ["ctrl", "download"])
def test_transfer_awaiting_propagation_does_not_reserve(kind, duplex):
    h = _h(duplex)
    c = h.clients[0]
    if kind == "ctrl":
        c.ctrl = ModelTransfer("ack", 10, [], t_start=0.0)
    else:
        c.download = ModelTransfer("download", 10, (1, h.server.theta.clone()), t_start=0.0)
    tr = lambda: c.ctrl if kind == "ctrl" else c.download
    h.step()
    assert tr() is not None and tr().done_tx and h.t < tr().t_arrival       # last bit sent, still in propagation
    assert not h.env.model_traffic_active[0] and not h.env.model_uplink_active[0]


@pytest.mark.parametrize("kind", ["ctrl", "download"])
def test_transfer_still_sending_reserves_in_half_duplex_but_not_the_uplink_in_full(kind):
    h = _h("half")
    c = h.clients[0]
    big = 1e9
    if kind == "ctrl":
        c.ctrl = ModelTransfer("ack", big, [], t_start=0.0)
    else:
        c.download = ModelTransfer("download", big, (1, h.server.theta.clone()), t_start=0.0)
    h.step()
    assert h.env.model_traffic_active[0]                                  # half duplex: any transfer still sending reserves
    assert not h.env.model_uplink_active[0]                               # a downlink transfer is never an uplink reservation


def test_upload_still_sending_reserves_the_uplink():
    h = _h("full")
    h.clients[0].upload = ModelTransfer("upload", 1e9, None, t_start=0.0)
    h.step()
    assert h.env.model_traffic_active[0] and h.env.model_uplink_active[0]


def test_reservation_is_released_after_arrival():
    h = _h("half")
    c = h.clients[0]
    c.ctrl = ModelTransfer("ack", 10, [], t_start=0.0)
    h.run(int(PROP) + 5)
    assert c.ctrl is None and not h.env.model_traffic_active[0]
