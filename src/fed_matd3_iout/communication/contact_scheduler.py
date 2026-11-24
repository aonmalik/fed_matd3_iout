"""AUV<->USV contact scheduler for model transfers.

Decides whether a pending update may START transmission now.
 * fragmentation allowed  : start whenever the contact window is open (progress persists across windows).
 * fragmentation disabled : start only if the estimated airtime fits the expected contact window, where the
   expected window is an EWMA of previously observed window lengths (initially unbounded).
Never assumes the link is continuously available and never assumes simultaneous contact of all AUVs.
"""
from __future__ import annotations

import numpy as np

from .packet_delivery import LinkState


class ContactScheduler:
    def __init__(self, n_auv: int, dt: float, allow_fragmentation: bool, ewma: float = 0.3):
        self.n = n_auv
        self.dt = dt
        self.allow_frag = allow_fragmentation
        self.ewma = ewma
        self.run_len = np.zeros(n_auv)            # current open-window length [s]
        self.expected_window = np.full(n_auv, np.inf)
        self.windows_seen = np.zeros(n_auv, dtype=int)
        self.deferred = np.zeros(n_auv, dtype=int)

    def observe(self, links: list[LinkState]) -> None:
        for i, l in enumerate(links):
            if l.up:
                self.run_len[i] += self.dt
            elif self.run_len[i] > 0:               # window just closed
                w = self.run_len[i]
                self.expected_window[i] = w if not np.isfinite(self.expected_window[i]) else \
                    (1 - self.ewma) * self.expected_window[i] + self.ewma * w
                self.windows_seen[i] += 1
                self.run_len[i] = 0.0

    def estimated_airtime_s(self, total_bytes: float, link: LinkState, share: float) -> float:
        r = link.rate_mbps * 1e6 / 8.0 * share
        return float("inf") if r <= 0 else total_bytes / r

    def may_start(self, i: int, total_bytes: float, link: LinkState, share: float) -> bool:
        if not link.up:
            return False
        if self.allow_frag:
            return True
        remaining = self.expected_window[i] - self.run_len[i]
        ok = self.estimated_airtime_s(total_bytes, link, share) <= max(remaining, 0.0) or \
            not np.isfinite(self.expected_window[i])
        if not ok:
            self.deferred[i] += 1
        return bool(ok)
