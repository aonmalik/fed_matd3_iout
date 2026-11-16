"""Local actor update  Delta_i = theta_i^local - theta_i^base  (Eq. F1) with full provenance."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class UpdatePacket:
    client_id: int
    update_id: int
    base_version: int                 # global version the client was last synchronised to
    delta: torch.Tensor               # flat parameter difference
    snapshot: torch.Tensor            # local parameters at creation (becomes the new reference after ACK)
    n_samples: float                  # n_i = gradient steps * batch size since the reference
    local_steps: int
    n_transitions: int
    norm: float
    stats: dict = field(default_factory=dict)   # measured rollout statistics (relevance inputs)
    t_created: float = 0.0
    t_upload_start: float = -1.0
    t_received: float = -1.0
    size_bytes: float = 0.0

    @property
    def n_params(self) -> int:
        return int(self.delta.numel())
