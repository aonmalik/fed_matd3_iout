import numpy as np
import pytest
from fed_matd3_iout.communication.acoustic_channel import (AcousticLink, db_to_lin, lin_to_db, rayleigh_success_prob,
                                                            thorp_absorption_db_per_km, transmission_loss_db)
from fed_matd3_iout.communication.propagation_delay import propagation_delay_s


def test_thorp_known_value_and_monotone():
    assert float(thorp_absorption_db_per_km(10.0)) == pytest.approx(1.2, abs=0.05)       # ~1.2 dB/km at 10 kHz
    f = np.array([1, 5, 10, 25, 50])
    assert np.all(np.diff(thorp_absorption_db_per_km(f)) > 0)                         # frequency dependent


def test_zero_distance_limit_and_monotone_loss():
    # d < d0 = 1 m is clamped: spreading term = 0 dB, absorption evaluated at 1 m = alpha(f)*1e-3 dB
    expected = float(thorp_absorption_db_per_km(25.0)) * 1e-3
    assert float(transmission_loss_db(0.0, 25.0)) == pytest.approx(expected)
    assert float(transmission_loss_db(0.5, 25.0)) == pytest.approx(expected)
    assert float(transmission_loss_db(0.0, 25.0)) < 0.01                               # ~0 dB, finite (no -inf)
    d = np.array([1, 10, 100, 1000, 5000])
    assert np.all(np.diff(transmission_loss_db(d, 25.0)) > 0)


def test_spreading_exponent_exact():
    # no-absorption check: doubling distance adds k*10*log10(2) dB spreading + absorption term
    a0 = float(thorp_absorption_db_per_km(25.0))
    diff = float(transmission_loss_db(200.0, 25.0, k=1.5) - transmission_loss_db(100.0, 25.0, k=1.5))
    assert diff == pytest.approx(15 * np.log10(2) + a0 * 0.1, rel=1e-9)


def test_snr_unit_consistency(cfg):
    l = AcousticLink(cfg.acoustic.sensor_link, cfg.acoustic)
    d = 20.0
    manual = l.sl - float(transmission_loss_db(d, l.f, l.k)) - l.nl
    assert float(l.snr_db(d)) == pytest.approx(manual)
    assert float(lin_to_db(db_to_lin(manual))) == pytest.approx(manual)              # dB <-> linear round trip
    assert float(l.snr_db(d)) > float(l.snr_db(10 * d))


def test_packet_success_bounds_and_derivation(cfg):
    thr = 10.0
    assert float(rayleigh_success_prob(thr, thr)) == pytest.approx(np.exp(-1.0))        # P(gamma>=thr), gamma~Exp(mean)
    for d in (1, 50, 500, 5000, 20000):
        for k in ("sensor_link", "usv_link"):
            p = float(AcousticLink(cfg.acoustic[k], cfg.acoustic).p_success(d))
            assert 0.0 <= p <= 1.0
    l = AcousticLink(cfg.acoustic.usv_link, cfg.acoustic)
    ps = [float(l.p_success(d)) for d in (100, 1000, 5000, 20000)]
    assert all(x >= y for x, y in zip(ps, ps[1:]))
    # Monte-Carlo agreement with the closed form
    rng = np.random.default_rng(0)
    mean_snr = 15.0
    mc = np.mean(rng.exponential(mean_snr, 400000) >= 10.0)
    assert mc == pytest.approx(float(rayleigh_success_prob(mean_snr, 10.0)), abs=3e-3)


def test_propagation_delay():
    assert propagation_delay_s(1500.0, 1500.0) == pytest.approx(1.0)
    assert propagation_delay_s(0.0) == 0.0


def test_rate_capped(cfg):
    l = AcousticLink(cfg.acoustic.sensor_link, cfg.acoustic)
    assert float(l.rate_mbps(10.0)) <= cfg.acoustic.sensor_link.max_rate_mbps + 1e-12


def test_source_level_and_noise_model():
    from fed_matd3_iout.communication.acoustic_channel import noise_psd_db, source_level_db
    assert source_level_db(1.0, 1.0) == pytest.approx(170.8)                        # 1 W omni source
    assert source_level_db(2.0, 1.0) - source_level_db(1.0, 1.0) == pytest.approx(10 * np.log10(2))
    n12 = noise_psd_db(12.0, 0.5, 5.0)
    assert 20.0 < n12 < 70.0                                                         # plausible dB re uPa^2/Hz band
    assert noise_psd_db(12.0, 0.5, 15.0) > noise_psd_db(12.0, 0.5, 2.0)              # rougher sea -> more noise
    assert noise_psd_db(1.0) > noise_psd_db(50.0)                                    # noise falls with frequency here


def test_gilbert_elliott_stationary_fraction(cfg):
    from fed_matd3_iout.communication.acoustic_channel import AcousticLink
    from fed_matd3_iout.communication.packet_delivery import LinkProcess
    u = dict(cfg.acoustic.usv_link); u.update(outage_p_drop=0.1, outage_p_recover=0.3, range_m=1e9)
    lp = LinkProcess(1, AcousticLink(cfg.acoustic.usv_link, cfg.acoustic), u, np.random.default_rng(0), np.random.default_rng(1))
    down = [lp.step(np.array([50.0]))[0].outage for _ in range(40000)]
    assert np.mean(down) == pytest.approx(0.1 / (0.1 + 0.3), abs=0.02)                # stationary P(down) = p_d/(p_d+p_r)
