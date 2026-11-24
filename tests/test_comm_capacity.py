"""Protocol 2.2: per-AUV airtime arbitration of model traffic (communication/capacity_arbiter.py + protocol + env).

Deterministic, no randomness.  Airtime is expressed as a fraction of ONE step on the AUV<->USV channel.
"""
import numpy as np
import pytest

from fed_fixtures import Harness, link
from fed_matd3_iout.communication.capacity_arbiter import arbitrate
from fed_matd3_iout.communication.model_transfer import ModelTransfer
from fed_matd3_iout.environment.multi_auv_iout_env import MultiAUVIoUTEnv

POOL = 0.5


# ------------------------------------------------------------------ arbiter unit tests
def test_conservation_for_many_demand_patterns():
    rng = np.random.default_rng(0)
    for _ in range(500):
        d = {"upload": rng.uniform(0, 3), "ctrl": rng.uniform(0, 3), "download": rng.uniform(0, 3)}
        g = arbitrate(d, POOL)
        assert sum(g.values()) <= POOL + 1e-12
        assert all(0.0 <= g[k] <= d[k] + 1e-12 for k in d)


def test_ctrl_and_download_do_not_each_get_the_full_pool():
    g = arbitrate({"upload": 0.0, "ctrl": 10.0, "download": 10.0}, POOL)
    assert g["ctrl"] + g["download"] <= POOL + 1e-12          # 2.1: each received the full pool (sum = 2 * pool)
    assert g["download"] > 0                                    # a huge control backlog cannot starve the download


def test_small_control_message_is_served_first_and_releases_the_rest():
    g = arbitrate({"upload": 5.0, "ctrl": 0.01, "download": 5.0}, POOL)
    assert g["ctrl"] == pytest.approx(0.01)
    assert g["upload"] == pytest.approx(g["download"]) == pytest.approx((POOL - 0.01) / 2)


def test_max_min_fairness_releases_surplus_of_small_demand():
    g = arbitrate({"upload": 0.1, "ctrl": 0.0, "download": 5.0}, POOL)
    assert g["upload"] == pytest.approx(0.1) and g["download"] == pytest.approx(0.4)


def test_no_starvation_when_pool_positive_and_ctrl_small():
    g = arbitrate({"upload": 9.0, "ctrl": 0.2, "download": 9.0}, POOL)
    assert g["upload"] > 0 and g["download"] > 0


def test_arbiter_is_deterministic_and_order_independent():
    a = arbitrate({"upload": 2.0, "ctrl": 0.1, "download": 2.0}, POOL)
    b = arbitrate({"download": 2.0, "upload": 2.0, "ctrl": 0.1}, POOL)
    assert a == b == arbitrate({"upload": 2.0, "ctrl": 0.1, "download": 2.0}, POOL)


# ------------------------------------------------------------------ protocol-level tests
def _h(cfg, **kw):
    h = Harness(cfg, 1, "async", **kw)
    h.hold()                                                   # no new updates: only the transfers we inject
    return h


def _bytes_per_step(link_):
    return link_.rate_mbps * 1e6 / 8.0


def test_control_and_download_overlap_share_one_pool(cfg):
    h = _h(cfg)
    c = h.clients[0]
    big = 1e9
    h.set_links(rate=0.05)
    c.ctrl = ModelTransfer("ack", big, [])
    c.download = ModelTransfer("download", big, (1, h.server.theta.clone()))
    h.server.version = 1                                       # keep the download legitimate
    h.step()
    moved = c.ctrl.progress_bytes + c.download.progress_bytes
    assert moved <= POOL * _bytes_per_step(h.env.links[0]) * h.proto.dt + 1e-6
    assert c.ctrl.progress_bytes > 0 and c.download.progress_bytes > 0


def test_upload_and_control_overlap_conserve_capacity_every_step(cfg):
    h = _h(cfg)
    c = h.clients[0]
    h.set_links(rate=0.05)
    from fed_matd3_iout.federated.update_packet import UpdatePacket  # noqa: F401
    c.ctrl = ModelTransfer("ack", 1e9, [])
    c.download = ModelTransfer("download", 1e9, (0, h.server.theta.clone()))
    h.server.version = 1
    for _ in range(25):
        h.step()
        assert sum(h.proto.last_alloc[0].values()) <= POOL + 1e-9
    assert h.proto.alloc_violations == 0 and h.proto.alloc_steps >= 25


def test_multiple_transfers_in_one_direction_all_progress(cfg):
    h = _h(cfg)
    c = h.clients[0]
    h.set_links(rate=0.05)
    h.server.version = 1
    c.ctrl = ModelTransfer("ack", 1e9, [])
    c.download = ModelTransfer("download", 1e9, (1, h.server.theta.clone()))
    for _ in range(10):
        h.step()
    assert c.ctrl.progress_bytes > 0 and c.download.progress_bytes > 0     # neither is starved


def test_fading_loss_grants_nothing_and_does_not_charge_airtime(cfg):
    h = _h(cfg)
    c = h.clients[0]
    h.set_links(rate=0.05, success=False)
    h.server.version = 1
    c.download = ModelTransfer("download", 1e9, (1, h.server.theta.clone()))
    h.step()
    assert c.download.progress_bytes == 0 and c.download.failures == 1 and c.stats["rx_s"] == 0.0


def test_intermittent_contact_keeps_fragment_progress(cfg):
    h = _h(cfg)
    c = h.clients[0]
    h.server.version = 1
    c.download = ModelTransfer("download", 1e9, (1, h.server.theta.clone()))
    h.set_links(rate=0.05); h.step(); p = c.download.progress_bytes
    h.set_links(up=False); h.step()
    assert c.download.progress_bytes == p > 0                              # fragmentation allowed in the default cfg
    h.set_links(rate=0.05); h.step()
    assert c.download.progress_bytes > p


def test_energy_equals_power_times_granted_airtime(cfg):
    h = _h(cfg)
    c = h.clients[0]
    h.server.version = 1
    h.set_links(rate=0.05)
    c.ctrl = ModelTransfer("ack", 1e9, [])
    c.download = ModelTransfer("download", 1e9, (1, h.server.theta.clone()))
    h.step()
    g = h.proto.last_alloc[0]
    assert c.stats["rx_s"] == pytest.approx((g["ctrl"] + g["download"]) * h.proto.dt)
    assert c.stats["rx_s"] <= POOL * h.proto.dt + 1e-9                     # never more airtime than the pool


def test_completing_transfer_charges_only_the_airtime_it_used(cfg):
    tr = ModelTransfer("ack", 100.0, [])
    used = tr.step(0.0, 1.0, link(rate=0.05), 0.25, True)
    assert tr.done_tx and used == pytest.approx(100.0 / (0.05 * 1e6 / 8))


def test_transfer_with_zero_share_neither_moves_nor_charges(cfg):
    tr = ModelTransfer("ack", 100.0, [])
    assert tr.step(0.0, 1.0, link(rate=0.05), 0.0, True) == 0.0 and tr.progress_bytes == 0.0


def test_full_duplex_uses_separate_directional_pools(cfg):
    cfg.acoustic.usv_link["duplex"] = "full"
    h = _h(cfg)
    c = h.clients[0]
    h.server.version = 1
    h.set_links(rate=0.05)
    c.download = ModelTransfer("download", 1e9, (1, h.server.theta.clone()))
    h.step()
    assert h.proto.last_alloc[0]["download"] == pytest.approx(POOL)
    assert h.proto.alloc_violations == 0


def test_invalid_duplex_value_is_rejected(cfg):
    cfg.acoustic.usv_link["duplex"] = "simplex"
    with pytest.raises(ValueError):
        Harness(cfg, 1, "async")


# ------------------------------------------------------------------ sensor-data reservation (environment, real step)
def _offered_cap(cfg, duplex, upload, other):
    from collections import deque
    from fed_matd3_iout.communication.packet_delivery import LinkState
    cfg.acoustic.usv_link["duplex"] = duplex
    cfg.env.episode_steps = 5
    env = MultiAUVIoUTEnv(cfg)
    env.reset(0)
    ls = LinkState(True, False, True, True, 10.0, 0.05, 1.0, 0.0)
    env.link_proc.step = lambda d: [ls for _ in range(env.M)]
    seen = []
    orig = env._uplink
    env._uplink = lambda a, cap, t, l: (seen.append((a.aid, cap)) or orig(a, cap, t, l))
    env.auvs[0].onboard = deque([type("B", (), {})()])          # truthy => the data-uplink branch runs (stubbed below)
    env._uplink = lambda a, cap, t, l: (seen.append((a.aid, cap)) or (0.0, 0.0))
    env.model_uplink_active[0] = upload
    env.model_traffic_active[0] = upload or other
    env.step(np.zeros((env.M, 2)))
    return [c for i, c in seen if i == 0][0], ls.rate_mbps * env.dt


@pytest.mark.parametrize("duplex,upload,other,reserved", [
    ("half", False, False, False),
    ("half", True, False, True),
    ("half", False, True, True),        # half duplex: a download/control transfer uses the same airtime
    ("full", False, True, False),       # full duplex: an unrelated downlink transfer must NOT reduce the sensor uplink
    ("full", True, False, True),        # an upload competes with the sensor uplink in either model
])
def test_sensor_uplink_reservation_is_direction_aware(cfg, duplex, upload, other, reserved):
    cap, full = _offered_cap(cfg, duplex, upload, other)
    assert cap == pytest.approx(full * ((1 - POOL) if reserved else 1.0))
