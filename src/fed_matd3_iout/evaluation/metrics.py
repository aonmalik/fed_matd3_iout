"""Episode metrics with explicit denominators (all Mbit unless stated; all per-episode).

 generated G            : sum of sensor generation incl. initial buffer content                      [Mbit]
 collected              : left sensor buffers into AUVs                                              [Mbit]
 delivered              : delivered end-to-end to the USV                                            [Mbit]
 residual               : sensor backlog at episode end (onboard-carried data reported separately)   [Mbit]
 overflow               : tail-dropped at sensors                                                    [Mbit]
 collection_ratio       = collected / G
 delivery_ratio (PDR)   = delivered / G                (end-to-end, generated-referenced)
 delivery_of_collected  = delivered / collected
 U_total                = sum_k d_k c_k sigma_k exp(-mu_k tau_k)        (decayed, criticality-weighted)
 U_norm                 = U_total / sum_{generated} c_k sigma_k          (vs undecayed value of ALL generated packets)
 U_per_packet           = U_total / delivered packets
 critical_pdr           = delivered critical packets / generated critical packets
 mean_age               = unit-weighted mean end-to-end age of delivered packets [s]
 voi_retention          = U_total / sum_{delivered} c sigma
 energy_*               [J];  E_per_Mbit = E_total / delivered;  E_per_U = E_total / U_total
 travel_m               = sum over AUVs of travelled distance [m]
 mean_travel_time_s     = mean over AUVs of time spent in transit [s]
 backlog_clearance_time_s = FIRST-PASSAGE time at which (sensor backlog + onboard) <= BACKLOG_CLEARANCE_FRACTION (10%) of the data G
                          generated SO FAR (sensors keep generating, so G grows; this is NOT the completion time of a fixed workload).
                          Censored rule: if the condition never holds (including G = 0, where the ratio is undefined) the value is the
                          episode length and backlog_clearance_censored = 1.  The first passage is final: later new generation does not
                          re-open it.  (Protocol 2.2 rename of the metric called completion_time_s in 2.0/2.1; same definition.)
 completion_time_s      = DEPRECATED legacy alias of backlog_clearance_time_s (identical value), kept so 2.0/2.1 tables stay readable;
                          completion_censored is the alias of backlog_clearance_censored.  A fixed-workload completion time is NOT implemented
                          (see docs/AUTHOR_DECISIONS.md D10).
 unsafe_pair_steps/events, boundary_violations
 workload_imbalance     = std(collected_i) / mean(collected_i) over AUVs  (coefficient of variation; 0 if mean=0)
"""
from __future__ import annotations

import numpy as np


BACKLOG_CLEARANCE_FRACTION = 0.10


class BacklogClearanceTracker:
    """First-passage detector for  (residual + onboard) <= fraction * G_generated_so_far  (G > 0)."""

    def __init__(self, fraction: float = BACKLOG_CLEARANCE_FRACTION):
        self.fraction = float(fraction)
        self.time_s: float | None = None

    def update(self, t_end_s: float, generated: float, backlog: float) -> None:
        if self.time_s is None and generated > 0 and backlog <= self.fraction * generated:
            self.time_s = float(t_end_s)

    def result(self, episode_len_s: float) -> tuple[float, bool]:
        """(value, censored)."""
        return (self.time_s, False) if self.time_s is not None else (float(episode_len_s), True)


def episode_metrics(env) -> dict:
    acc = env.accounting()
    L = env.ledger
    gen_val = sum(s.queue.gen_value for s in env.sensors)
    gen_crit = sum(s.queue.gen_crit_units for s in env.sensors)
    en = env.energy
    E_prop, E_tx, E_rx, E_comp = sum(en.prop), sum(en.tx), sum(en.rx), sum(en.comp)
    E_tot = E_prop + E_tx + E_rx + E_comp
    col = np.array(env.collected_units_by_auv, dtype=float)
    G, deliv = acc["G"], acc["delivered"]
    return {
        "generated_mbit": G, "collected_mbit": acc["collected"], "delivered_mbit": deliv,
        "residual_mbit": acc["residual"], "onboard_mbit": acc["onboard"], "overflow_mbit": acc["lost"],
        "collection_ratio": acc["collected"] / G if G > 0 else 0.0,
        "delivery_ratio": deliv / G if G > 0 else 0.0,
        "delivery_of_collected": deliv / acc["collected"] if acc["collected"] > 0 else 0.0,
        "U_total": L.U, "U_norm": L.U / gen_val if gen_val > 0 else 0.0,
        "U_per_packet": L.mean_per_packet, "critical_pdr": L.crit_units / gen_crit if gen_crit > 0 else 0.0,
        "mean_age_s": L.mean_age, "voi_retention": L.voi_retention,
        "E_prop_j": E_prop, "E_comm_j": E_tx + E_rx, "E_comp_j": E_comp, "E_total_j": E_tot,
        "E_per_mbit": E_tot / deliv if deliv > 0 else float("nan"),
        "E_per_U": E_tot / L.U if L.U > 0 else float("nan"),
        "U_per_E": L.U / E_tot if E_tot > 0 else 0.0,
        "travel_m": float(env.travel_by_auv.sum()),
        "mean_travel_time_s": float(np.mean([a.transit_steps for a in env.auvs]) * env.dt),
        "backlog_clearance_time_s": float(getattr(env, "backlog_clearance_time_s", env.T * env.dt)),
        "backlog_clearance_censored": float(getattr(env, "backlog_clearance_censored", 1.0)),
        "completion_time_s": float(getattr(env, "backlog_clearance_time_s", env.T * env.dt)),     # DEPRECATED legacy alias
        "unsafe_pair_steps": env.prox.pair_steps, "unsafe_events": env.prox.events,
        "boundary_violations": float(env.boundary_viol.sum()),
        "workload_imbalance": float(col.std() / col.mean()) if col.mean() > 0 else 0.0,
        "mass_balance_err": abs(acc["sensor_balance_err"]) + abs(acc["e2e_balance_err"]),
    }
