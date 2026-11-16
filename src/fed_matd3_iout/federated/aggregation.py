"""Server-side aggregation (Eqs. F5-F8).  NOT unbiased FedAvg unless weights reduce to n_i.

  Z_i   = n_i^alpha * exp(-kappa Delta_i) * [eps_M + (1-eps_M) M_i]^eta                       (F5)
  w_i   = Z_i / sum_{j in R_r} Z_j,   optional cap w_i <= w_max with iterative renormalisation (F6)
  Clip_C(d) = d * min(1, C/||d||_2)         global L2-norm clipping over the whole flat vector (F7)
  theta^{r+1} = theta^r + eta_g * sum_{i in R_r} w_i Clip_C(Delta_i)                           (F8)
R_r contains ONLY successfully received and accepted updates.  Link reliability is NOT multiplied into Z_i.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def l2_clip(delta: torch.Tensor, clip_norm: float | None) -> tuple[torch.Tensor, float]:
    n = float(delta.norm())
    if clip_norm is None or not math.isfinite(clip_norm) or n <= clip_norm or n == 0.0:
        return delta, 1.0
    s = clip_norm / n
    return delta * s, s


def raw_weights(n, stale, rel, alpha: float, eta: float, eps_m: float) -> np.ndarray:
    n, stale, rel = (np.asarray(x, dtype=float) for x in (n, stale, rel))
    return np.power(np.maximum(n, 1e-12), alpha) * stale * np.power(eps_m + (1.0 - eps_m) * np.clip(rel, 0, 1), eta)


def normalise_with_cap(z: np.ndarray, cap: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if len(z) == 0:
        return z
    if z.sum() <= 0:
        z = np.ones_like(z)
    w = z / z.sum()
    n = len(w)
    cap = max(float(cap), 1.0 / n)               # a cap below 1/n is infeasible
    if cap >= 1.0:
        return w
    fixed = np.zeros(n, dtype=bool)
    for _ in range(n + 1):
        over = (w > cap + 1e-12) & ~fixed
        if not over.any():
            break
        fixed |= over
        w[fixed] = cap
        rest = ~fixed
        if rest.any():
            w[rest] = z[rest] / z[rest].sum() * (1.0 - cap * fixed.sum())
    return w / w.sum()


def aggregate(theta: torch.Tensor, deltas: list[torch.Tensor], weights: np.ndarray, server_lr: float,
              clip_norm: float | None) -> tuple[torch.Tensor, list[float]]:
    if len(deltas) == 0:
        return theta, []
    for d in deltas:
        if d.numel() != theta.numel():
            raise ValueError("model-shape mismatch in aggregation")
    acc = torch.zeros_like(theta)
    scales = []
    for d, w in zip(deltas, weights):
        c, s = l2_clip(d, clip_norm)
        acc += float(w) * c
        scales.append(s)
    return theta + server_lr * acc, scales
