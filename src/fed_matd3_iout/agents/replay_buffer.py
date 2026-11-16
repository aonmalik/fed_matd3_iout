"""Replay memory.

JointStore  : one ring of joint transitions (state, all local obs, all actions, all rewards, next ..., done) kept
              once in memory (the simulator provides the joint information needed by the centralised critics).
ClientReplay: a *logical* per-AUV replay buffer = ring of transition ids (tids) admitted by that AUV.  Admission
              is client-specific (Paper 2: transitions are stored only for steps in which the AUV is in transit),
              which yields different (non-IID) local experience distributions without duplicating memory.
"""
from __future__ import annotations

import numpy as np
import torch


class JointStore:
    def __init__(self, capacity: int, state_dim: int, n_agents: int, obs_dim: int, act_dim: int):
        self.cap = int(capacity)
        self.M = n_agents
        self.s = np.zeros((self.cap, state_dim), np.float32)
        self.s2 = np.zeros((self.cap, state_dim), np.float32)
        self.o = np.zeros((self.cap, n_agents, obs_dim), np.float32)
        self.o2 = np.zeros((self.cap, n_agents, obs_dim), np.float32)
        self.a = np.zeros((self.cap, n_agents, act_dim), np.float32)
        self.r = np.zeros((self.cap, n_agents), np.float32)
        self.d = np.zeros((self.cap,), np.float32)
        self.n_added = 0                      # monotonically increasing transition id counter

    def add(self, s, o, a, r, s2, o2, d) -> int:
        k = self.n_added % self.cap
        self.s[k], self.o[k], self.a[k], self.r[k], self.s2[k], self.o2[k], self.d[k] = s, o, a, r, s2, o2, float(d)
        tid = self.n_added
        self.n_added += 1
        return tid

    def valid(self, tid: np.ndarray) -> np.ndarray:
        return tid >= self.n_added - self.cap

    def gather(self, tids: np.ndarray, device) -> dict:
        k = tids % self.cap
        f = lambda x: torch.as_tensor(x[k], device=device)
        return dict(s=f(self.s), o=f(self.o), a=f(self.a), r=f(self.r), s2=f(self.s2), o2=f(self.o2), d=f(self.d))


class ClientReplay:
    def __init__(self, store: JointStore, capacity: int, rng: np.random.Generator):
        self.store = store
        self.cap = int(capacity)
        self.ring = np.full(self.cap, -1, dtype=np.int64)
        self.n = 0
        self.rng = rng

    def add(self, tid: int) -> None:
        self.ring[self.n % self.cap] = tid
        self.n += 1

    def __len__(self) -> int:
        m = self.ring[: min(self.n, self.cap)]
        return int(self.store.valid(m).sum()) if len(m) else 0

    def sample(self, batch_size: int, device) -> dict:
        m = self.ring[: min(self.n, self.cap)]
        m = m[self.store.valid(m)]
        if len(m) == 0:
            raise RuntimeError("empty replay buffer")
        idx = self.rng.integers(0, len(m), size=batch_size)
        return self.store.gather(m[idx], device)
