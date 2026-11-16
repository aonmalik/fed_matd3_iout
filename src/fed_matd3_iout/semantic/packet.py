"""Packet-level data representation.

A *packet* is one data unit (size q_j Mbit, Paper 2: Q/B_j).  All units generated at the same sensor, in the
same step, with the same criticality category are identical (same sigma, mu, c, t_gen) and are stored as one
``Batch`` of ``units`` packets.  This is exactly equivalent to storing them one by one and lets FIFO service
split a batch fractionally (rate-limited partial collection) without ever duplicating units.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

_pkt_counter = 0


def next_packet_id() -> int:
    global _pkt_counter
    _pkt_counter += 1
    return _pkt_counter


@dataclass
class Batch:
    pkt_id: int          # identifier of the (first) packet in the batch
    sensor_id: int
    t_gen: float         # generation timestamp [s]
    units: float         # number of packets (data units) in this batch  (>0)
    sigma: float         # semantic importance in [0,1]
    category: str        # mission-criticality category
    mu: float            # information-decay coefficient [1/s]
    c: float             # mission-priority multiplier (criticality enters ONLY here)
    status: str = "queued"   # queued | onboard | delivered | dropped
    t_collect: float = -1.0
    collected_by: int = -1

    def split(self, take: float) -> tuple["Batch", "Batch"]:
        """Split off ``take`` units from the head; returns (head, remainder). Units are conserved."""
        assert 0 < take < self.units
        head = replace(self, units=take)
        rest = replace(self, units=self.units - take)
        return head, rest
