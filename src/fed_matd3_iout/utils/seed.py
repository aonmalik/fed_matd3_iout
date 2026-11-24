"""Independent, named random streams (environment, arrivals, fading, comm scheduling, training, eval)."""
from __future__ import annotations

import random

import numpy as np

STREAMS = ("layout", "arrivals", "fading", "comm", "policy_noise", "eval")
_STREAM_ID = {n: i for i, n in enumerate(STREAMS)}


def make_rng(base_seed: int, stream: str, offset: int = 0) -> np.random.Generator:
    """Deterministic generator for a named stream; distinct streams never share state."""
    if stream not in _STREAM_ID:
        raise KeyError(f"unknown stream {stream}")
    ss = np.random.SeedSequence([int(base_seed), _STREAM_ID[stream], int(offset)])
    return np.random.default_rng(ss)


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
