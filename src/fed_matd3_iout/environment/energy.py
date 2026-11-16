"""AUV energy model (Eq. E1-E4).  Components are disjoint -> no double counting.

  P_prop = P0 + kappa * v^3            [W]   hotel/baseline + cubic hydrodynamic drag (simplified, UNCALIBRATED)
  E_prop = P_prop * dt                 [J]   (v = 0 while hovering -> only P0)
  E_tx   = P_tx_elec * t_tx            [J]   AUV transmissions (data + model uploads) -- time on air
  E_rx   = P_rx * t_rx                 [J]   AUV receptions (sensor data during service, model/ack downloads)
  E_comp = e_grad * n_grad_steps       [J]   local training computation

The action-based quadratic penalty of Paper 2 is NOT mixed in: the cubic law is the physical energy; the
reward penalises this physical energy (normalised), not a separate quadratic action cost.
"""
from __future__ import annotations


def propulsion_power(v: float, p0: float, kappa: float) -> float:
    return p0 + kappa * v ** 3


def propulsion_energy(v: float, dt: float, p0: float, kappa: float) -> float:
    return propulsion_power(v, p0, kappa) * dt


class EnergyAccount:
    """Per-AUV energy ledger [J]."""

    def __init__(self, n_auv: int):
        self.prop = [0.0] * n_auv
        self.tx = [0.0] * n_auv
        self.rx = [0.0] * n_auv
        self.comp = [0.0] * n_auv

    def total(self, i: int | None = None) -> float:
        if i is None:
            return sum(self.prop) + sum(self.tx) + sum(self.rx) + sum(self.comp)
        return self.prop[i] + self.tx[i] + self.rx[i] + self.comp[i]
