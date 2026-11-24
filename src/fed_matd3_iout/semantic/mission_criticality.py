"""Mission-criticality categories and packet attribute sampling.

sigma = omega_sensor * sigma_factor(category) in [0,1]   (importance; NO criticality factor)
c     = c_priority(category)                              (criticality multiplier; applied once, in utility)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Category:
    name: str
    prob: float
    sigma_factor: float
    c_priority: float
    mu: float


def build_categories(sem_cfg) -> list[Category]:
    cats = [Category(n, float(v["prob"]), float(v["sigma_factor"]), float(v["c_priority"]), float(v["mu"]))
            for n, v in sem_cfg["categories"].items()]
    p = np.array([c.prob for c in cats])
    if abs(p.sum() - 1.0) > 1e-9:
        raise ValueError(f"category probabilities must sum to 1, got {p.sum()}")
    for c in cats:
        if not (0 <= c.sigma_factor <= 1) or c.c_priority < 1 or c.mu < 0:
            raise ValueError(f"invalid category {c}")
    return cats


def sigma_of(omega: float, cat: Category) -> float:
    """Normalised semantic importance in [0,1]."""
    return float(min(1.0, max(0.0, omega * cat.sigma_factor)))


def mean_semantic_value_density(omega: float, cats: list[Category]) -> float:
    """E[c*sigma] per packet at a sensor with weight omega.  AGGREGATE APPROXIMATION used only for
    task allocation / relevance normalisation -- never in the principal utility objective."""
    return float(sum(c.prob * c.c_priority * sigma_of(omega, c) for c in cats))
