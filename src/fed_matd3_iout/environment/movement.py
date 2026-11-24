"""Goal-conditioned kinematics (Paper 2 Eq. (13)): a in [-1,1]^2 -> speed and absolute heading.

  v = v_min + (a1+1)/2 (v_max - v_min),   theta = pi * a2,   p <- p + (v [cos th, sin th] + current) dt
A move that would leave the square region is rejected (AUV stays) and flagged as a boundary violation.
"""
from __future__ import annotations

import math

import numpy as np


def action_to_vel(a, v_min: float, v_max: float) -> tuple[float, float]:
    a = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
    v = v_min + 0.5 * (a[0] + 1.0) * (v_max - v_min)
    th = math.pi * a[1]
    return float(v), float(th)


def step_position(p, v: float, th: float, dt: float, region: float, current=(0.0, 0.0)):
    """Returns (new_pos, moved, boundary_violation, travelled_m)."""
    disp = np.array([v * math.cos(th), v * math.sin(th)]) * dt + np.asarray(current, dtype=float) * dt
    new = np.asarray(p, dtype=float) + disp
    if new[0] < 0 or new[0] > region or new[1] < 0 or new[1] > region:
        return np.asarray(p, dtype=float).copy(), False, True, 0.0
    return new, True, False, float(np.linalg.norm(disp))
