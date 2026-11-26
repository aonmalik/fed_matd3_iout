"""A single model transfer (upload of a local update, control messages, or download of the global actor).

``share`` is the fraction of the step's airtime granted to THIS transfer by the per-AUV arbiter
(communication/capacity_arbiter.py); bytes moved = full link rate * dt * share, airtime used = share * dt (time division),
so the airtime/energy charged equals what was actually granted and what was actually used.

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
        rate_full = link.rate_mbps * 1e6 / 8.0                 # bytes/s at the full link rate
        need = self.total_bytes - self.progress_bytes
        if need <= 1e-9 * max(self.total_bytes, 1.0):          # rounding left nothing to send: the last bit is already out
            self.progress_bytes = self.total_bytes
            self.done_tx = True
            self.t_tx_done = t
            self.t_arrival = t + link.prop_delay_s
            return 0.0
        if rate_full <= 0 or share <= 0:
            return 0.0
        can = rate_full * dt * share                           # bytes movable in the granted slot
        if can >= need:
            self.progress_bytes = self.total_bytes
            self.done_tx = True
            self.t_tx_done = t + need / (rate_full * share)    # completion time inside the step (same convention as 2.1)
            self.t_arrival = self.t_tx_done + link.prop_delay_s
            used = need / rate_full                            # airtime actually on the air
            self.tx_time_s += used
            return used
        self.progress_bytes += can
        used = dt * share
        self.tx_time_s += used
        return used
