"""Unmanned surface vehicle: surface relay + federated aggregation server host (exactly one)."""
from __future__ import annotations

import numpy as np


class USV:
    def __init__(self, pos, speed: float, policy: str = "centroid"):
        self.pos = np.asarray(pos, dtype=float).copy()
        self.speed = float(speed)
        self.policy = policy

    def step(self, auv_pos: np.ndarray, dt: float, region: float) -> float:
        """Move toward the fleet centroid (relay rule) at <= speed. Returns travelled distance."""
        if self.policy == "static" or len(auv_pos) == 0:
            return 0.0
        target = auv_pos.mean(axis=0)
        vec = target - self.pos
        n = float(np.linalg.norm(vec))
        if n < 1e-9:
            return 0.0
        move = min(self.speed * dt, n)
        self.pos = np.clip(self.pos + vec / n * move, 0.0, region)
        return move
