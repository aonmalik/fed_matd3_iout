"""Goal-seeking reference controller (NOT a learning baseline and NOT one of the 11 required methods).

Steers at full speed toward the current goal (sensor or USV) using only the local observation (goal vector).
Purpose: sanity ceiling for 'can this mobility task be solved at all with the given allocator?' and for debugging.
"""
from __future__ import annotations

import numpy as np


def goal_seeking_policies(n_auvs: int):
    i0 = 2 * (n_auvs - 1)                       # observation layout: [neighbour rel pos (2(M-1)), goal vector (2), ...]

    def pol(o):
        gx, gy = float(o[i0]), float(o[i0 + 1])
        if gx == 0.0 and gy == 0.0:
            return np.zeros(2)
        return np.array([1.0, np.arctan2(gy, gx) / np.pi])
    return [pol for _ in range(n_auvs)]
