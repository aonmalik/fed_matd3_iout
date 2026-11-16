"""AUV state container (kinematics + onboard carry buffer + service state machine)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..semantic.packet import Batch


@dataclass
class AUV:
    aid: int
    pos: np.ndarray
    depth: float
    cap_mbit: float
    target: int = -1                    # assigned sensor id (-1 none)
    to_usv: bool = False                # goal mode: True => head to USV
    hover_left: int = 0                 # remaining planned hover steps
    hover_sensor: int = -1
    onboard: deque = field(default_factory=deque)   # FIFO of Batch
    onboard_mbit: float = 0.0
    v: float = 0.0
    theta: float = 0.0
    travelled: float = 0.0
    boundary_flag: int = 0
    # link & belief state (observation freshness)
    belief_pos: np.ndarray = None       # believed positions of all AUVs (stale unless link up)
    belief_age: float = 0.0
    link_up: bool = False
    # statistics
    collected_mbit: float = 0.0
    served_units: float = 0.0
    service_time_s: float = 0.0
    transit_steps: int = 0
    hover_steps: int = 0

    def fill(self) -> float:
        return self.onboard_mbit / self.cap_mbit if self.cap_mbit > 0 else 0.0
