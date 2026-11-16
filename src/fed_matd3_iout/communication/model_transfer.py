"""A single model transfer (upload of a local update or download of the global actor).

Progress accrues only in steps where the contact window is open AND the fading draw succeeds. A failed
(fading) step transfers nothing (packet loss) and is counted. With fragmentation disabled, losing the contact
window resets progress (whole-model retransmission). The payload is considered *received* only after the last
bit arrives plus the one-way propagation delay.
"""
from __future__ import annotations

from dataclasses import dataclass

from .packet_delivery import LinkState


@dataclass
class ModelTransfer:
    kind: str                 # "upload" | "download"
    total_bytes: float
    payload: object = None
    progress_bytes: float = 0.0
    t_start: float = 0.0
    attempts: int = 0         # steps on which a transmission was attempted
    failures: int = 0         # attempted steps lost to fading
    resets: int = 0
    tx_time_s: float = 0.0    # time on air (for energy)
    done_tx: bool = False     # last bit sent
    t_tx_done: float = -1.0
    t_arrival: float = -1.0   # last bit received (t_tx_done + propagation)

    def step(self, t: float, dt: float, link: LinkState, share: float, allow_frag: bool) -> float:
        """Advance one step; returns seconds spent transmitting (for energy)."""
        if self.done_tx:
            return 0.0
        if not link.up:
            if self.progress_bytes > 0 and not allow_frag:
                self.progress_bytes = 0.0
                self.resets += 1
            return 0.0
        self.attempts += 1
        if not link.success:
            self.failures += 1
            return 0.0
        rate_Bps = link.rate_mbps * 1e6 / 8.0 * share
        if rate_Bps <= 0:
            return 0.0
        need = self.total_bytes - self.progress_bytes
        can = rate_Bps * dt
        if can >= need:
            used = need / rate_Bps
            self.progress_bytes = self.total_bytes
            self.done_tx = True
            self.t_tx_done = t + used
            self.t_arrival = self.t_tx_done + link.prop_delay_s
            self.tx_time_s += used
            return used
        self.progress_bytes += can
        self.tx_time_s += dt
        return dt
