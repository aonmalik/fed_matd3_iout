"""Normalisation helpers (all dimensionless outputs)."""
from __future__ import annotations

import numpy as np


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def safe_ratio(num, den, default=0.0):
    den = np.asarray(den, dtype=float)
    num = np.asarray(num, dtype=float)
    out = np.full(np.broadcast(num, den).shape, default, dtype=float)
    m = den > 0
    np.divide(num, den, out=out, where=m)
    return out if out.ndim else float(out)


def minmax(x, lo, hi):
    return clip01((np.asarray(x, dtype=float) - lo) / max(hi - lo, 1e-12))
