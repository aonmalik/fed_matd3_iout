"""Staleness  Delta_i = r - v_i  (Eq. F2),  w_stale = exp(-kappa*Delta)  (Eq. F3, class C), max-staleness gate."""
from __future__ import annotations

import math


def staleness(server_version: int, base_version: int) -> int:
    return int(server_version) - int(base_version)


def staleness_factor(delta: int, kappa: float) -> float:
    return math.exp(-float(kappa) * max(int(delta), 0))


def is_acceptable(delta: int, max_staleness: int) -> bool:
    return 0 <= int(delta) <= int(max_staleness)
