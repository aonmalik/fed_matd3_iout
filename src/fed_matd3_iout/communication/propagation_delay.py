"""tau_prop = d / v_s   (Eq. C7).  d [m], v_s [m/s] -> seconds."""
from __future__ import annotations


def propagation_delay_s(d_m, v_s: float = 1500.0):
    return d_m / v_s
