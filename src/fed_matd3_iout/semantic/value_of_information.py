"""VoI_k(tau) = sigma_k * exp(-mu_k * tau)   (Eq. S1)."""
from __future__ import annotations

import numpy as np


def voi(sigma, mu, age_s):
    age = np.maximum(np.asarray(age_s, dtype=float), 0.0)
    return np.asarray(sigma, dtype=float) * np.exp(-np.asarray(mu, dtype=float) * age)
