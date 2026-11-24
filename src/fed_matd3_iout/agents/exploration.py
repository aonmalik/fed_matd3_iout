"""Exploration noise schedule: linear decay of the Gaussian std over episodes."""
from __future__ import annotations

import numpy as np


class NoiseSchedule:
    def __init__(self, start: float, end: float, decay_episodes: int):
        self.start, self.end, self.n = float(start), float(end), max(int(decay_episodes), 1)

    def std(self, episode: int) -> float:
        f = min(max(episode / self.n, 0.0), 1.0)
        return self.start + f * (self.end - self.start)


def gaussian_action_noise(rng: np.random.Generator, shape, std: float) -> np.ndarray:
    return rng.normal(0.0, std, size=shape)
