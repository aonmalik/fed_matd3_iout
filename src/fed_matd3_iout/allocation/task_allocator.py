"""Event-triggered task allocation: 'sta' (proposed), 'greedy' / 'hungarian_priority' (Paper 2 priority), 'nearest'.

One-to-one: each participating AUV gets at most one sensor and each sensor at most one AUV (Hungarian /
shortest-augmenting-path via scipy.optimize.linear_sum_assignment on the rectangular cost matrix). Sensors
locked by non-participants (hovering / committed) are excluded. If free sensors < participating AUVs the
surplus AUVs stay unassigned (idle, -1).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .semantic_service_cost import AllocState, cost_matrix


def priority_matrix(st: AllocState, alpha: float) -> np.ndarray:
    """Paper 2 / Xu et al. priority rho_ij = lam_j b_j / B_j - alpha d_ij (higher is better)."""
    dist = np.linalg.norm(st.auv_pos[:, None, :] - st.sensor_pos[None, :, :], axis=-1)
    return st.lam[None, :] * st.b[None, :] / st.b_max - alpha * dist


class TaskAllocator:
    def __init__(self, acfg, use_semantic: bool = True):
        self.method = acfg["method"]
        self.backlog_model = acfg["backlog_model"]
        self.w = dict(acfg["weights"])
        self.alpha = float(acfg["priority_alpha"])
        self.v_nom = float(acfg["v_nom"])
        self.contact_time = float(acfg["contact_time_s"])
        self.tol = float(acfg["poisson_tol"])
        self.big = float(acfg["big_cost"])
        self.use_semantic = use_semantic
        self.events = 0
        self.last_diag = None

    def allocate(self, st: AllocState, participants: list[int], free_sensors: list[int], freed: int | None = None,
                 rng: np.random.Generator | None = None) -> dict[int, int]:
        """Returns {auv: sensor or -1} for the participants (greedy/nearest: only ``freed`` is reassigned)."""
        self.events += 1
        P, S = list(participants), list(free_sensors)
        out = {i: -1 for i in P}
        if not P or not S:
            return out
        sub = AllocState(st.auv_pos[P], st.auv_load[P], st.auv_free_mbit[P], st.cur_target[P], st.sensor_pos[S],
                         st.b[S], st.lam[S], st.b_max, st.q, st.density[S], st.last_served[S], st.t_step,
                         st.episode_steps, st.rate_mbps, st.d_max)
        if self.method == "sta":
            C, diag = cost_matrix(sub, self.w, self.backlog_model, self.v_nom, self.contact_time, self.tol,
                                  self.use_semantic)
            self.last_diag = diag
            r, c = linear_sum_assignment(C)
            for a, s in zip(r, c):
                out[P[a]] = S[s]
        elif self.method == "hungarian_priority":
            rho = priority_matrix(sub, self.alpha)
            r, c = linear_sum_assignment(-rho)
            for a, s in zip(r, c):
                out[P[a]] = S[s]
        elif self.method in ("greedy", "nearest"):
            who = freed if freed in P else P[0]
            a = P.index(who)
            for i in P:                                   # sequential baselines keep other AUVs' targets
                out[i] = int(st.cur_target[i]) if i != who else -1
            taken = {t for i, t in out.items() if i != who and t >= 0}
            cand = [k for k, s in enumerate(S) if s not in taken]
            if cand:
                if self.method == "greedy":
                    rho = priority_matrix(sub, self.alpha)[a]
                    k = max(cand, key=lambda kk: rho[kk])
                else:
                    d = np.linalg.norm(sub.sensor_pos - sub.auv_pos[a], axis=1)
                    k = min(cand, key=lambda kk: d[kk])
                out[who] = S[k]
        else:
            raise ValueError(f"unknown allocation method {self.method}")
        return out
