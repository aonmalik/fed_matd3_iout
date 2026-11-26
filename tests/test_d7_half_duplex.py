"""D7 (provisional decision): half-duplex is the Protocol 2.2 default.  Scope: ONE airtime budget per AUV (one AUV<->USV link and one
modem per AUV in the code: one p_elec_w, one rx_power_w, one LinkProcess per AUV).  Budget: ``model_share`` of a step for ALL model
traffic (upload + control + download); sensor-data uplink keeps the rest while model traffic is active.  Deterministic, no randomness."""
import numpy as np
import pytest

from fed_fixtures import Harness
from fed_matd3_iout.communication.capacity_arbiter import arbitrate
from fed_matd3_iout.communication.model_transfer import ModelTransfer

POOL = 0.5


def _h(cfg):
    h = Harness(cfg, 1, "async")
    h.hold()
    h.set_links(rate=0.05)
    h.server.version = 1
    return h


def _load(h, up=1e9, ctrl=1e9, down=1e9):
    c = h.clients[0]
    c.upload = ModelTransfer("upload", up, None, t_start=0.0) if up else None
    c.ctrl = ModelTransfer("ack", ctrl, [], t_start=0.0) if ctrl else None
    c.download = ModelTransfer("download", down, (1, h.server.theta.clone()), t_start=0.0) if down else None
    return c


def test_conflicting_simultaneous_tx_and_rx_requests_never_exceed_one_budget(cfg):
    h = _h(cfg)
    c = _load(h)
    for _ in range(12):
        tx0, rx0 = c.stats["tx_s"], c.stats["rx_s"]
        h.step()
        assert (c.stats["tx_s"] - tx0) + (c.stats["rx_s"] - rx0) <= POOL * h.proto.dt + 1e-9     # no simultaneous TX and RX
        assert sum(h.proto.last_alloc[0].values()) <= POOL + 1e-9
    assert h.proto.alloc_violations == 0


def test_control_has_strict_priority_up_to_its_demand():
    g = arbitrate({"upload": 9.0, "ctrl": 0.2, "download": 9.0}, POOL)
    assert g["ctrl"] == pytest.approx(0.2) and g["upload"] == pytest.approx(g["download"]) == pytest.approx(0.15)
    g = arbitrate({"upload": 9.0, "ctrl": 9.0, "download": 9.0}, POOL)
    assert g["ctrl"] > g["upload"]                                                   # still served first, but bounded (see below)


def test_remaining_capacity_is_split_fairly_and_unused_capacity_is_reassigned_within_the_pool():
    assert arbitrate({"upload": 9.0, "ctrl": 0.0, "download": 9.0}, POOL) == {"upload": 0.25, "ctrl": 0.0, "download": 0.25}
    g = arbitrate({"upload": 0.05, "ctrl": 0.0, "download": 9.0}, POOL)             # upload needs little: surplus goes to download
    assert g["upload"] == pytest.approx(0.05) and g["download"] == pytest.approx(0.45)
    assert sum(g.values()) == pytest.approx(POOL)


def test_budget_exhaustion_control_is_capped_at_half_the_pool_while_others_wait():
    g = arbitrate({"upload": 5.0, "ctrl": 5.0, "download": 5.0}, POOL)
    assert g["ctrl"] == pytest.approx(POOL / 2)                                      # priority cannot take more than half
    assert g["upload"] == pytest.approx(g["download"]) == pytest.approx(POOL / 4)    # nobody is starved
    assert sum(g.values()) == pytest.approx(POOL)                                    # pool exactly exhausted
    alone = arbitrate({"upload": 0.0, "ctrl": 5.0, "download": 0.0}, POOL)
    assert alone["ctrl"] == pytest.approx(POOL)                                      # with nobody waiting, ctrl may use the pool


def test_unused_capacity_with_no_demand_is_not_granted():
    assert arbitrate({"upload": 0.0, "ctrl": 0.0, "download": 0.0}, POOL) == {"upload": 0.0, "ctrl": 0.0, "download": 0.0}
    g = arbitrate({"upload": 0.1, "ctrl": 0.0, "download": 0.0}, POOL)
    assert g["upload"] == pytest.approx(0.1)                                         # grant <= demand, never the whole pool


def test_direction_change_releases_the_pool_to_the_other_direction(cfg):
    h = _h(cfg)
    c = _load(h, up=1e9, ctrl=0, down=1e9)
    h.step()
    first = dict(h.proto.last_alloc[0])
    assert first["upload"] > 0 and first["download"] > 0
    c.upload = None                                                                  # the uplink transfer ends
    h.step()
    assert h.proto.last_alloc[0]["download"] == pytest.approx(POOL)                 # downlink now gets the whole pool
    assert h.proto.last_alloc[0]["upload"] == 0.0


def test_airtime_and_energy_follow_granted_airtime(cfg):
    h = _h(cfg)
    c = _load(h)
    h.step()
    g = h.proto.last_alloc[0]
    assert c.stats["tx_s"] == pytest.approx(g["upload"] * h.proto.dt)
    assert c.stats["rx_s"] == pytest.approx((g["ctrl"] + g["download"]) * h.proto.dt)


def test_identical_inputs_give_identical_allocations_and_progress(cfg):
    out = []
    for _ in range(2):
        h = _h(cfg)
        c = _load(h)
        trace = []
        for _ in range(8):
            h.step()
            trace.append((dict(h.proto.last_alloc[0]), c.upload.progress_bytes, c.ctrl.progress_bytes, c.download.progress_bytes))
        out.append(trace)
    assert out[0] == out[1]


def test_model_grants_never_exceed_the_airtime_the_sensor_uplink_left_free(cfg):
    """Half duplex: sensor-data airtime used in the step + model airtime <= one step (the modem cannot do both at once)."""
    h = _h(cfg)
    c = _load(h)
    h.env.sensor_uplink_airtime_s = np.array([0.8 * h.proto.dt])           # the sensor uplink already used 80% of the step
    h.step()
    assert sum(h.proto.last_alloc[0].values()) <= 0.2 + 1e-9               # only the free 20% (< pool 0.5) is available
    total = 0.8 * h.proto.dt + c.stats["tx_s"] + c.stats["rx_s"]
    assert total <= h.proto.dt + 1e-9
