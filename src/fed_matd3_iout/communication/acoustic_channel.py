"""Underwater acoustic link budget (all log-domain quantities in dB re 1 uPa unless stated).

  alpha(f)    Thorp absorption [dB/km], f in kHz                                  (Eq. C1)  class A (empirical, f>~0.4 kHz)
  TL(d,f)     = k*10log10(max(d,d0)/d0) + alpha(f)*d/1000   [dB], d in m, d0=1 m  (Eq. C2)  class B
  N(f)        = turbulence + shipping + wind + thermal PSDs, summed LINEARLY      (Eq. C3)  class A (Stojanovic/Coates)
  SL          = 170.8 + 10log10(eta*P_elec) + DI   [dB re uPa @1m]                (Eq. C4)  class A
  SNR_dB      = SL - TL - (N(f)_dB + 10log10(B_Hz))                               (Eq. C5)
  rate        = min(R_max, B log2(1+SNR_lin))/1e6  [Mbit/s]  (Shannon, mean SNR)  (Eq. C6)  class D
  P_succ      = exp(-gamma_th / SNR_lin)  Rayleigh: gamma ~ Exp(mean SNR_lin)      (Eq. C8)  class B
  tau_prop    = d / v_s                                                            (Eq. C7)

Zero-distance behaviour: for d < d0 = 1 m the spreading term is clamped to 0 dB (the far-field model is not
valid inside the reference range); TL is therefore continuous and non-decreasing in d.
"""
from __future__ import annotations

import math

import numpy as np


def thorp_absorption_db_per_km(f_khz):
    f2 = np.asarray(f_khz, dtype=float) ** 2
    return 0.11 * f2 / (1 + f2) + 44.0 * f2 / (4100.0 + f2) + 2.75e-4 * f2 + 0.003


def transmission_loss_db(d_m, f_khz, k: float = 1.5, d0: float = 1.0):
    d = np.maximum(np.asarray(d_m, dtype=float), d0)
    return k * 10.0 * np.log10(d / d0) + thorp_absorption_db_per_km(f_khz) * d / 1000.0


def noise_psd_db(f_khz: float, shipping: float = 0.5, wind_mps: float = 5.0) -> float:
    """Ambient noise PSD [dB re uPa^2/Hz] (Stojanovic empirical, f in kHz). Components added in linear power."""
    lf = math.log10(f_khz)
    n_t = 17.0 - 30.0 * lf
    n_s = 40.0 + 20.0 * (shipping - 0.5) + 26.0 * lf - 60.0 * math.log10(f_khz + 0.03)
    n_w = 50.0 + 7.5 * math.sqrt(wind_mps) + 20.0 * lf - 40.0 * math.log10(f_khz + 0.4)
    n_th = -15.0 + 20.0 * lf
    lin = sum(10.0 ** (x / 10.0) for x in (n_t, n_s, n_w, n_th))
    return 10.0 * math.log10(lin)


def source_level_db(p_elec_w: float, efficiency: float, directivity_db: float = 0.0) -> float:
    return 170.8 + 10.0 * math.log10(p_elec_w * efficiency) + directivity_db


def db_to_lin(x_db):
    return 10.0 ** (np.asarray(x_db, dtype=float) / 10.0)


def lin_to_db(x):
    return 10.0 * np.log10(np.asarray(x, dtype=float))


def shannon_rate_mbps(snr_lin, bandwidth_hz: float, max_rate_mbps: float):
    r = bandwidth_hz * np.log2(1.0 + np.asarray(snr_lin, dtype=float)) / 1e6      # bit/s -> Mbit/s
    return np.minimum(r, max_rate_mbps)


def rayleigh_success_prob(mean_snr_lin, thr_lin):
    """P(gamma >= gamma_th) for gamma ~ Exp(mean=mean_snr_lin)  (Rayleigh amplitude => exponential power)."""
    return np.exp(-np.asarray(thr_lin, dtype=float) / np.maximum(np.asarray(mean_snr_lin, dtype=float), 1e-300))


class AcousticLink:
    """Link budget for one link type (sensor<->AUV or AUV<->USV), built from a config section."""

    def __init__(self, link_cfg, ac_cfg):
        self.f = float(link_cfg["freq_khz"])
        self.B = float(link_cfg["bandwidth_hz"])
        self.k = float(ac_cfg["spreading_k"])
        self.sl = source_level_db(link_cfg["p_elec_w"], link_cfg["efficiency"], link_cfg.get("directivity_db", 0.0))
        self.nl = noise_psd_db(self.f, ac_cfg["shipping_s"], ac_cfg["wind_mps"]) + 10.0 * math.log10(self.B)
        self.thr_db = float(link_cfg["snr_threshold_db"])
        self.max_rate = float(link_cfg["max_rate_mbps"])
        self.fading = bool(link_cfg.get("fading", True))
        self.v_s = float(ac_cfg["sound_speed_mps"])

    def snr_db(self, d_m):
        return self.sl - transmission_loss_db(d_m, self.f, self.k) - self.nl

    def rate_mbps(self, d_m):
        return shannon_rate_mbps(db_to_lin(self.snr_db(d_m)), self.B, self.max_rate)

    def p_success(self, d_m):
        if not self.fading:
            return np.where(self.snr_db(d_m) >= self.thr_db, 1.0, 0.0)
        return rayleigh_success_prob(db_to_lin(self.snr_db(d_m)), db_to_lin(self.thr_db))

    def prop_delay(self, d_m):
        return np.asarray(d_m, dtype=float) / self.v_s
