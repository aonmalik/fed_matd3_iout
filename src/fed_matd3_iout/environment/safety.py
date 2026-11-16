"""Inter-AUV separation: unsafe-proximity indicators (NOT verified physical collision counts)."""
from __future__ import annotations

import numpy as np


def pair_distances(pos: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    return d


def proximity_penalty(pos: np.ndarray, d_safe: float) -> np.ndarray:
    """Per-AUV penalty sum_k (d_safe - d_ik)^+ (each pair counted for both members, symmetric)."""
    d = pair_distances(pos)
    np.fill_diagonal(d, np.inf)
    return np.maximum(d_safe - d, 0.0).sum(axis=1)


class ProximityTracker:
    """Counts pair-steps inside d_safe and distinct *entry events* (pair enters the region)."""

    def __init__(self, n: int):
        self.inside = np.zeros((n, n), dtype=bool)
        self.pair_steps = 0
        self.events = 0

    def update(self, pos: np.ndarray, d_safe: float) -> tuple[int, int]:
        d = pair_distances(pos)
        ins = (d < d_safe) & ~np.eye(len(pos), dtype=bool)
        upper = np.triu(ins, 1)
        prev = np.triu(self.inside, 1)
        steps = int(upper.sum())
        ev = int((upper & ~prev).sum())
        self.inside = ins
        self.pair_steps += steps
        self.events += ev
        return steps, ev
