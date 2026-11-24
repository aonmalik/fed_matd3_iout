"""Deterministic actor mu_theta(o) with tanh output in [-1,1]^act_dim, plus flat-parameter utilities."""
from __future__ import annotations

import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, act_dim), nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def flatten_params(m: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()]).clone().cpu()


def load_flat_params(m: nn.Module, flat: torch.Tensor) -> None:
    n = sum(p.numel() for p in m.parameters())
    if flat.numel() != n:
        raise ValueError(f"parameter vector size mismatch: got {flat.numel()}, model has {n}")
    i = 0
    with torch.no_grad():
        for p in m.parameters():
            k = p.numel()
            p.copy_(flat[i:i + k].view_as(p).to(p.device, p.dtype))
            i += k


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
