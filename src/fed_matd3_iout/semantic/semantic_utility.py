"""Packet-level semantic utility (Eq. S2-S4).

U_k = d_k * c_k * sigma_k * exp(-mu_k * tau_k)       (delivered packet; d_k in {0,1})
U   = sum_k U_k                                       (total semantic utility)
U_norm = U / sum_k c_k sigma_k                        (normalised vs. undecayed value of ALL generated packets)

Batches of ``units`` identical packets contribute ``units`` times the per-packet value.
"""
from __future__ import annotations

import numpy as np

from .packet import Batch
from .value_of_information import voi


def delivered_utility(batch: Batch, age_s: float, delivered: bool = True) -> float:
    """Utility of delivering ``batch`` with end-to-end age ``age_s``. Failed delivery => 0."""
    if not delivered:
        return 0.0
    return float(batch.units * batch.c * voi(batch.sigma, batch.mu, age_s))


def undecayed_value(batch: Batch) -> float:
    return float(batch.units * batch.c * batch.sigma)


class UtilityLedger:
    """Accumulates delivered-packet statistics. Each delivered Batch must be recorded exactly once."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.U = 0.0                 # total decayed, criticality-weighted utility
        self.V_undecayed = 0.0       # sum c*sigma*units over delivered packets
        self.units = 0.0
        self.age_units = 0.0         # sum units*age
        self.crit_units = 0.0
        self.by_auv: dict[int, float] = {}
        self._seen: set = set()

    def record(self, batch: Batch, age_s: float, auv: int = -1) -> float:
        key = (batch.pkt_id, batch.t_collect, batch.units, batch.sensor_id, auv, round(float(age_s), 9))
        if key in self._seen:        # guards against double counting of the very same chunk
            raise RuntimeError(f"duplicate delivery of chunk {key}")
        self._seen.add(key)
        u = delivered_utility(batch, age_s)
        self.U += u
        self.V_undecayed += undecayed_value(batch)
        self.units += batch.units
        self.age_units += batch.units * max(age_s, 0.0)
        if batch.category == "critical":
            self.crit_units += batch.units
        self.by_auv[auv] = self.by_auv.get(auv, 0.0) + u
        return u

    @property
    def mean_per_packet(self) -> float:
        return self.U / self.units if self.units > 0 else 0.0

    @property
    def voi_retention(self) -> float:
        return self.U / self.V_undecayed if self.V_undecayed > 0 else 0.0

    @property
    def mean_age(self) -> float:
        return self.age_units / self.units if self.units > 0 else 0.0
