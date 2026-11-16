"""AUV<->USV link process: contact windows (range), Gilbert-Elliott outages, per-step Rayleigh fading draw.

All random draws are made EVERY step for EVERY AUV regardless of state, so that two methods evaluated on the
same scenario seed see identical channel realisations (paired comparison).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acoustic_channel import AcousticLink


@dataclass
class LinkState:
    in_range: bool
    outage: bool            # True => link down (outage)
    up: bool                # in_range and not outage  (contact window open)
    success: bool           # fading draw: instantaneous SNR >= threshold this step
    dist_m: float
    rate_mbps: float
    p_success: float
    prop_delay_s: float


class LinkProcess:
    def __init__(self, n_auv: int, link: AcousticLink, usv_cfg, rng_fading: np.random.Generator,
                 rng_comm: np.random.Generator):
        self.n = n_auv
        self.link = link
        self.range_m = float(usv_cfg["range_m"])
        self.p_drop = float(usv_cfg["outage_p_drop"])
        self.p_rec = float(usv_cfg["outage_p_recover"])
        self.rf = rng_fading
        self.rc = rng_comm
        self.down = np.zeros(n_auv, dtype=bool)

    def step(self, slant_dists: np.ndarray) -> list[LinkState]:
        u = self.rc.random(self.n)
        e = self.rf.exponential(1.0, self.n)
        # Gilbert-Elliott transition
        newly_down = (~self.down) & (u < self.p_drop)
        newly_up = self.down & (u < self.p_rec)
        self.down = (self.down | newly_down) & ~newly_up
        out = []
        for i in range(self.n):
            d = float(slant_dists[i])
            in_r = d <= self.range_m
            mean_snr = float(10 ** (self.link.snr_db(d) / 10.0))
            thr = float(10 ** (self.link.thr_db / 10.0))
            succ = (e[i] * mean_snr >= thr) if self.link.fading else (mean_snr >= thr)
            out.append(LinkState(in_r, bool(self.down[i]), bool(in_r and not self.down[i]), bool(succ), d,
                                 float(self.link.rate_mbps(d)), float(self.link.p_success(d)),
                                 float(self.link.prop_delay(d))))
        return out
