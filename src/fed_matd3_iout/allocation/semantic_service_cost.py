"""Proposed semantic-service assignment cost (Eq. A1) -- a PROPOSED design (class C), not established.

  C_ij = -w_b B~ - w_s U~ + w_d T~ + w_o L~ + w_c S~ + w_l W~ - w_a A~
  B~_ij = predicted backlog at arrival / B_max                    in [0,1]   (exact | ce | current)
  U~_ij = u~_j * D_ij / B_max   with u~_j = E[c*sigma]_j / max_k E[c*sigma]_k   in [0,1]  (AGGREGATE APPROXIMATION)
  T~_ij = T_travel / T_ref,  T_ref = d_max / v_nom                 in [0,1]
  L~_ij = E[L_arr] / B_max (clipped)                              in [0,1]   (loss that is unrecoverable by arrival)
  S~_ij = 1[j != current target of i]                              {0,1}      (assignment continuity)
  W~_ij = load_i * service_time_ij / T_contact                    in [0,1]   (busier AUVs prefer short tasks)
  A~_j  = (t - last_served_j) / T_episode                          in [0,1]   (optional, default weight 0)
All terms are dimensionless, so quantities with different units are never added directly.
Note: terms depending only on i (not on j) cannot change a one-to-one optimum, hence W~ is an i-j interaction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .service_arrival_predictor import (certainty_equivalent_backlog, expected_backlog_exact, travel_time_s)
from .service_capacity import capacity_constant, collectable_units, service_time_s


@dataclass
class AllocState:
    auv_pos: np.ndarray          # (M,2)
    auv_load: np.ndarray         # (M,) in [0,1]
    auv_free_mbit: np.ndarray    # (M,)
    cur_target: np.ndarray       # (M,) int (-1 none)
    sensor_pos: np.ndarray       # (N,2)
    b: np.ndarray                # (N,) current occupancy [units]
    lam: np.ndarray              # (N,) [units/s]
    b_max: float
    q: float                     # Mbit/unit
    density: np.ndarray          # (N,) normalised E[c*sigma] in [0,1]
    last_served: np.ndarray      # (N,) step index (-1 never)
    t_step: int
    episode_steps: int
    rate_mbps: float             # sensor-link rate at hover
    d_max: float


def cost_matrix(st: AllocState, aw: dict, backlog_model: str, v_nom: float, contact_time_s: float,
                tol: float, use_semantic: bool = True) -> tuple[np.ndarray, dict]:
    M, N = len(st.auv_pos), len(st.sensor_pos)
    dist = np.linalg.norm(st.auv_pos[:, None, :] - st.sensor_pos[None, :, :], axis=-1)       # (M,N), straight path
    T = travel_time_s(dist, v_nom)
    b = np.broadcast_to(st.b[None, :], (M, N))
    lam = np.broadcast_to(st.lam[None, :], (M, N))
    if backlog_model == "exact":
        eb, el = expected_backlog_exact(b, lam, T, st.b_max, tol)
    elif backlog_model == "ce":
        eb = certainty_equivalent_backlog(b, lam, T, st.b_max)
        el = np.maximum(b + lam * T - st.b_max, 0.0)
    elif backlog_model == "current":
        eb, el = b.copy(), np.zeros_like(b)
    else:
        raise ValueError(backlog_model)
    B_t = np.clip(eb / st.b_max, 0, 1)
    L_t = np.clip(el / st.b_max, 0, 1)
    cap = capacity_constant(st.rate_mbps, contact_time_s)
    D = collectable_units(eb, cap, st.q, st.auv_free_mbit[:, None])
    U_t = np.clip(st.density[None, :] * D / st.b_max, 0, 1)
    T_ref = st.d_max / v_nom
    T_t = np.clip(T / T_ref, 0, 1)
    sw = (np.arange(N)[None, :] != st.cur_target[:, None]).astype(float)
    sw[st.cur_target < 0, :] = 0.0                       # no current target => nothing to preserve
    svc = service_time_s(D, st.rate_mbps, st.q)
    W_t = np.clip(st.auv_load[:, None] * svc / contact_time_s, 0, 1)
    age = np.where(st.last_served >= 0, st.t_step - st.last_served, st.t_step)
    A_t = np.clip(age / max(st.episode_steps, 1), 0, 1)[None, :] * np.ones((M, 1))
    w_s = aw["semantic"] if use_semantic else 0.0
    C = (-aw["backlog"] * B_t - w_s * U_t + aw["travel"] * T_t + aw["overflow"] * L_t
         + aw["switch"] * sw + aw["workload"] * W_t - aw["age"] * A_t)
    return C, dict(B=B_t, U=U_t, T=T_t, L=L_t, S=sw, W=W_t, A=A_t, D=D, dist=dist)
