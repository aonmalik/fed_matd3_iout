"""Static underwater sensor node."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sensor_queue import SensorQueue


@dataclass
class SensorNode:
    sid: int
    pos: np.ndarray               # horizontal (x,y) [m]
    depth: float                  # [m]
    lam_base: float               # data units / s
    queue: SensorQueue
    owner: int = -1               # AUV currently assigned (lock), -1 if none
    last_served_step: int = -1
    visits: int = 0
    owned_overflow: dict = field(default_factory=dict)   # auv -> overflow units while owned

    def lam(self, step: int, rate_change_step: int, factor: float) -> float:
        """lambda_j(t): piecewise-constant rate with an optional abrupt change (robustness experiments)."""
        if rate_change_step >= 0 and step >= rate_change_step:
            return self.lam_base * factor
        return self.lam_base
