"""Twin (or single) Q-networks.

centralized (CTDE): Q_{i,k}(s, a_1..a_M)  input = [state, joint action]   (needs simulator state at TRAINING time)
independent        : Q_{i,k}(o_i, a_i)     input = [local obs, own action]
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(inp: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))


class TwinCritic(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, twin: bool = True):
        super().__init__()
        self.twin = twin
        self.q1 = _mlp(in_dim, hidden)
        self.q2 = _mlp(in_dim, hidden) if twin else None

    def forward(self, x: torch.Tensor):
        q1 = self.q1(x).squeeze(-1)
        q2 = self.q2(x).squeeze(-1) if self.twin else q1
        return q1, q2

    def q1_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.q1(x).squeeze(-1)
