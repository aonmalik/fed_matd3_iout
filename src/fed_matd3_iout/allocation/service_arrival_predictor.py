"""Service-arrival backlog prediction for a sensor with current occupancy b, rate lam [units/s], travel time T [s].

Arrivals during travel N ~ Poisson(m), m = lam*T, no intermediate service, finite buffer B_max.

 Exact expectation (Eq. P1, class B -- exact under the stated model):
     E[B_arr] = sum_n min(B_max, b+n) Pois(n; m)
 Truncation (documented, controlled): with K = B_max - b, terms n > floor(K) equal B_max EXACTLY, and
     E[B_arr] = sum_{n<=n_hi} (b+n) p_n + TAIL,   n_hi = min(floor(K), n_tail),  n_tail = (1-tol) Poisson quantile.
 * n_hi = floor(K): TAIL = B_max (1 - F(n_hi)) and the result is EXACT (no truncation error).
 * n_hi = n_tail < floor(K): the remaining n > n_hi are (up to mass <= tol) still below the cap, so
   TAIL = b (1-F(n_hi)) + m (1-F(n_hi-1))   [uses sum_{n>N} n p_n = m (1-F(N-1))],
   whose only error is the cap overshoot E[(b+N-B_max)^+ ; N > n_tail] <= tol * m (vanishingly small; tested).
 Expected overflow (Eq. P2): E[L_arr] = sum_n [b+n-B_max]^+ p_n = b + m - E[B_arr]  (conservation identity).
 Certainty-equivalent (Eq. P3, class D -- an UPPER BOUND on E[B_arr] by Jensen, NOT the exact expectation):
     B_hat = min(B_max, b + lam*T)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson


def travel_time_s(dist_m, v_nom: float, eps: float = 1e-9):
    """T = d / (v + eps)  (Eq. P0). Straight-line planned path at constant nominal speed (documented assumption)."""
    return np.asarray(dist_m, dtype=float) / (v_nom + eps)


def certainty_equivalent_backlog(b, lam, T, b_max):
    return np.minimum(b_max, np.asarray(b, dtype=float) + np.asarray(lam, dtype=float) * np.asarray(T, dtype=float))


def _one(b: float, m: float, b_max: float, tol: float) -> tuple[float, float]:
    b = min(max(b, 0.0), b_max)
    if m <= 0.0:
        return b, 0.0
    K = b_max - b
    n_floor = int(np.floor(K + 1e-12))
    n_tail = int(poisson.isf(tol, m)) + 1
    n_hi = min(n_floor, n_tail)
    n = np.arange(n_hi + 1)
    p = poisson.pmf(n, m)
    head = float(((b + n) * p).sum())
    if n_hi == n_floor:                                  # remaining terms are exactly B_max
        tail = b_max * float(poisson.sf(n_hi, m))
    else:                                                # remaining terms (mass <= tol) are below the cap
        tail = b * float(poisson.sf(n_hi, m)) + m * float(poisson.sf(n_hi - 1, m))
    e_b = min(head + tail, b_max)
    e_l = max(0.0, b + m - e_b)
    return e_b, e_l


def expected_backlog_exact(b, lam, T, b_max, tol: float = 1e-9):
    """Vectorised over broadcastable (b, lam, T). Returns (E[B_arr], E[L_arr])."""
    b, lam, T = np.broadcast_arrays(np.asarray(b, dtype=float), np.asarray(lam, dtype=float),
                                    np.asarray(T, dtype=float))
    eb = np.empty(b.shape)
    el = np.empty(b.shape)
    for idx in np.ndindex(b.shape):
        eb[idx], el[idx] = _one(float(b[idx]), float(lam[idx] * T[idx]), float(b_max), tol)
    return eb, el
