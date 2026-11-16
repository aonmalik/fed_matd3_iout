"""AUV-side federated client: local actor/critics/replay, versions, update construction, global-model installation."""
from __future__ import annotations

import torch

from ..agents.actor import flatten_params, load_flat_params
from ..agents.matd3_agent import MATD3Agent
from ..agents.replay_buffer import ClientReplay
from ..communication.model_transfer import ModelTransfer
from .update_packet import UpdatePacket


class FederatedClient:
    def __init__(self, cid: int, agent: MATD3Agent, replay: ClientReplay, fcfg, batch_size: int, speed: float):
        self.id = cid
        self.agent = agent
        self.replay = replay
        self.fcfg = fcfg
        self.batch_size = int(batch_size)
        self.speed = float(speed)
        self.credit = 0.0
        self.base_version = 0                    # last received global version
        self.local_version = 0                   # number of local actor-update packets created
        self.reference = flatten_params(agent.actor)   # theta^base (what Delta is measured against)
        self.steps_since = 0
        self.trans_since = 0
        self.total_steps = 0
        self.update_counter = 0
        self.upload: ModelTransfer | None = None
        self.ack: ModelTransfer | None = None
        self.download: ModelTransfer | None = None
        self.awaiting_global = False             # sync mode barrier
        self.stats = dict(attempted=0, succeeded=0, failed=0, deferred=0, timeouts=0, bytes_up=0.0, bytes_down=0.0,
                          tx_s=0.0, rx_s=0.0, latencies=[], sync_latencies=[])

    @property
    def model_params(self) -> int:
        return int(self.reference.numel())

    def update_ready(self) -> bool:
        return (self.steps_since >= int(self.fcfg["local_steps_per_update"])
                and self.trans_since >= int(self.fcfg["min_local_transitions"]))

    def make_update(self, t: float, window_stats: dict) -> UpdatePacket:
        """Delta = theta_local - theta_base  (Eq. F1)."""
        snap = flatten_params(self.agent.actor)
        delta = snap - self.reference
        self.update_counter += 1
        self.local_version += 1
        pkt = UpdatePacket(self.id, self.update_counter, self.base_version, delta, snap,
                           float(self.steps_since * self.batch_size), self.steps_since, self.trans_since,
                           float(delta.norm()), dict(window_stats), t_created=t,
                           size_bytes=float(delta.numel() * float(self.fcfg["model_bytes_per_param"])))
        return pkt

    def install_global(self, theta: torch.Tensor, version: int, blend: float) -> None:
        cur = flatten_params(self.agent.actor)
        new = blend * theta + (1.0 - blend) * cur
        load_flat_params(self.agent.actor, new)
        load_flat_params(self.agent.actor_t, new)          # keep the target actor consistent with the synced policy
        self.reference = new.clone()
        self.base_version = int(version)
        self.steps_since = 0
        self.trans_since = 0
