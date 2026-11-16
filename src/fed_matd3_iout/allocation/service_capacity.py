"""Service capacity and collectable volume (Eqs. SC1-SC3).

  C_service = R * T_contact                        constant rate        [Mbit]  (Eq. SC1)
  C_service = integral_{t_a}^{t_a+T} R(tau) dtau   time-varying rate    [Mbit]  (Eq. SC2, trapezoid rule)
  D_service = min(B_arr, C_service / q, free_onboard / q)   [units]     (Eq. SC3)
Units: R [Mbit/s], T [s], q [Mbit/unit].  Same data is never counted twice: D_service is a *prediction*;
actual collection removes units from the queue exactly once (SensorQueue.serve).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def capacity_constant(rate_mbps, t_contact_s):
    return np.asarray(rate_mbps, dtype=float) * np.asarray(t_contact_s, dtype=float)


def capacity_integral(rate_fn: Callable[[np.ndarray], np.ndarray], t_a: float, t_contact_s: float, n: int = 65) -> float:
    tau = np.linspace(t_a, t_a + t_contact_s, n)
    r = np.asarray(rate_fn(tau), dtype=float)
    return float(np.trapz(r, tau)) if hasattr(np, "trapz") else float(np.trapezoid(r, tau))


def collectable_units(b_arr, capacity_mbit, q: float, onboard_free_mbit=np.inf):
    return np.minimum(np.minimum(b_arr, np.asarray(capacity_mbit, dtype=float) / q),
                      np.asarray(onboard_free_mbit, dtype=float) / q)


def service_time_s(units, rate_mbps, q: float):
    r = np.maximum(np.asarray(rate_mbps, dtype=float), 1e-12)
    return np.asarray(units, dtype=float) * q / r
