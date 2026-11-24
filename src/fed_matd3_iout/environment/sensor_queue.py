"""Finite-buffer sensor queue with arrival-BEFORE-service event order (Paper 2, Eqs. (1)-(5)).

    B_pre   = B + A                       (Eq. Q1)
    L       = max(0, B_pre - B_max)       (Eq. Q2)  overflow: excess of the NEW arrivals is tail-dropped
    B_avail = min(B_max, B_pre)           (Eq. Q3)
    D       = min(B_avail, R*dt/q, free)  (Eq. Q4)  collected (rate- and onboard-capacity limited), FIFO
    B(t+1)  = B_avail - D                 (Eq. Q5)
Units: B, A, L, D in data units; q = Q/B_max Mbit per unit (conversion factor, Eq. Q6).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..semantic.packet import Batch, next_packet_id
from ..semantic.mission_criticality import Category, sigma_of


class SensorQueue:
    def __init__(self, sensor_id: int, b_max: float, q_mbit_per_unit: float, omega: float,
                 cats: list[Category]):
        self.sid = sensor_id
        self.b_max = float(b_max)
        self.q = float(q_mbit_per_unit)
        self.omega = float(omega)
        self.cats = cats
        self.q_batches: deque[Batch] = deque()
        self.occ = 0.0
        # accounting (data units)
        self.generated = 0.0      # incl. initial contents (G)
        self.collected = 0.0      # D (left the sensor buffer into an AUV)
        self.lost = 0.0           # L (overflow)
        self.gen_value = 0.0      # sum c*sigma over generated packets (undecayed)
        self.lost_value = 0.0
        self.gen_crit_units = 0.0

    # ---------------------------------------------------------------- helpers
    def _make_batches(self, t: float, n_units: float, rng: np.random.Generator) -> list[Batch]:
        if n_units <= 0:
            return []
        n = int(round(n_units))
        probs = np.array([c.prob for c in self.cats])
        counts = rng.multinomial(n, probs) if n > 0 else np.zeros(len(self.cats), int)
        out = []
        for c, k in zip(self.cats, counts):
            if k > 0:
                out.append(Batch(next_packet_id(), self.sid, float(t), float(k), sigma_of(self.omega, c),
                                 c.name, c.mu, c.c_priority))
        return out

    def initialise(self, t: float, units: float, rng: np.random.Generator) -> None:
        """Initial buffer content counts toward generated data G (Paper 2: G includes b_j(0))."""
        units = min(float(round(units)), self.b_max)
        for b in self._make_batches(t, units, rng):
            self._push(b)
            self._count_generated(b)

    def _push(self, b: Batch):
        self.q_batches.append(b)
        self.occ += b.units

    def _count_generated(self, b: Batch):
        self.generated += b.units
        v = b.units * b.c * b.sigma
        self.gen_value += v
        if b.category == "critical":
            self.gen_crit_units += b.units

    # ---------------------------------------------------------------- events
    def arrive(self, t: float, a_units: float, rng: np.random.Generator) -> tuple[float, float]:
        """Apply arrivals A(t). Returns (generated, lost_units). Overflow is tail-dropped from the new batches."""
        a_units = float(a_units)
        if a_units <= 0:
            return 0.0, 0.0
        b_pre = self.occ + a_units                       # Eq. Q1
        loss = max(0.0, b_pre - self.b_max)              # Eq. Q2
        new = self._make_batches(t, a_units, rng)
        for b in new:
            self._count_generated(b)
        # tail-drop `loss` units from the end of the new batches
        rem = loss
        kept: list[Batch] = []
        for b in reversed(new):
            if rem <= 0:
                kept.append(b)
                continue
            if b.units <= rem + 1e-12:
                rem -= b.units
                self.lost_value += b.units * b.c * b.sigma
                continue
            drop = rem
            self.lost_value += drop * b.c * b.sigma
            b.units -= drop
            rem = 0.0
            kept.append(b)
        for b in reversed(kept):
            self._push(b)
        self.lost += loss
        # Eq. Q3: occ is now min(b_max, b_pre) up to float tolerance
        self.occ = min(self.occ, self.b_max)
        return a_units, loss

    def serve(self, max_units: float) -> list[Batch]:
        """FIFO removal of up to ``max_units`` (Eq. Q4/Q5). Removed units are returned exactly once."""
        out: list[Batch] = []
        budget = min(float(max_units), self.occ)
        while budget > 1e-12 and self.q_batches:
            head = self.q_batches[0]
            if head.units <= budget + 1e-12:
                self.q_batches.popleft()
                budget -= head.units
                out.append(head)
            else:
                taken, rest = head.split(budget)
                self.q_batches[0] = rest
                budget = 0.0
                out.append(taken)
        got = sum(b.units for b in out)
        self.occ = max(0.0, self.occ - got)
        if not self.q_batches:
            self.occ = 0.0
        self.collected += got
        return out

    # ---------------------------------------------------------------- views
    @property
    def fill(self) -> float:
        return self.occ / self.b_max

    def residual_check(self) -> float:
        """G - (collected + lost + residual): must be ~0 (mass balance, Eq. Q7)."""
        return self.generated - (self.collected + self.lost + self.occ)
