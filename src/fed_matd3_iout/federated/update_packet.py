"""Local actor update  Delta_i = theta_i^local - theta_i^base  (Eq. F1), its lifecycle, and the downlink control messages.

Lifecycle (one state machine per update; terminal states have NO outgoing transition => outcomes are processed exactly once)

    CREATED -> UPLOADING -> IN_FLIGHT -> BUFFERED -> AGGREGATED
                    |            |           |-----> REJECTED     (stale at aggregation, ...)
                    |            |           `-----> EXPIRED      (waited in the USV buffer longer than max_buffer_age_s)
                    |            `-----------------> REJECTED     (refused at receipt: shape / stale / non-finite)
                    `------------------------------> EXPIRED      (upload did not finish within upload_timeout_s)

Who decides what (authority):
  * CREATED/UPLOADING          : the client (it may abandon an upload that does not finish -> EXPIRED).
  * IN_FLIGHT .. terminal      : the USV server.  Once the last bit has left the AUV the client can no longer withdraw the
                                  update, so it can never declare "expired" while the server might still aggregate it.
The client LEARNS the terminal outcome only through a downlink ``ControlMessage`` (it needs an acoustic contact), so
``status`` (authoritative) may be terminal while ``client_resolved`` is still False.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch


class UpdateStatus(str, Enum):
    CREATED = "created"
    UPLOADING = "uploading"
    IN_FLIGHT = "in_flight"
    BUFFERED = "buffered"
    AGGREGATED = "aggregated"
    REJECTED = "rejected"
    EXPIRED = "expired"


TERMINAL = frozenset({UpdateStatus.AGGREGATED, UpdateStatus.REJECTED, UpdateStatus.EXPIRED})
_ALLOWED = {
    UpdateStatus.CREATED: {UpdateStatus.UPLOADING},
    UpdateStatus.UPLOADING: {UpdateStatus.IN_FLIGHT, UpdateStatus.EXPIRED},
    UpdateStatus.IN_FLIGHT: {UpdateStatus.BUFFERED, UpdateStatus.REJECTED},
    UpdateStatus.BUFFERED: {UpdateStatus.AGGREGATED, UpdateStatus.REJECTED, UpdateStatus.EXPIRED},
}


class IllegalTransition(RuntimeError):
    pass


@dataclass
class UpdatePacket:
    client_id: int
    update_id: int                    # strictly increasing per client => uid is globally unique
    base_version: int                 # global version the client was last synchronised to
    delta: torch.Tensor               # flat parameter difference
    snapshot: torch.Tensor            # local parameters at creation (becomes the new reference ONLY after aggregation commit)
    n_samples: float                  # n_i = gradient steps * batch size since the reference
    local_steps: int
    n_transitions: int
    norm: float
    stats: dict = field(default_factory=dict)   # window statistics FROZEN at creation; owned by this update (see disposition)
    t_created: float = 0.0
    t_upload_start: float = -1.0
    t_received: float = -1.0
    size_bytes: float = 0.0
    episode_id: int = -1              # environment episode that produced ``stats``
    sync_epoch: int = 0               # client's global-install counter at creation (reference still valid iff unchanged)
    status: UpdateStatus = UpdateStatus.CREATED
    reason: str = ""
    stats_disposition: str = "held"   # held | consumed (aggregated) | returned (to window) | discarded_episode
    receipt_acked: bool = False       # client has seen the transport receipt
    client_resolved: bool = False     # client has processed the terminal outcome
    bytes_sent: float = 0.0           # uplink bytes actually transmitted for this update (full size once IN_FLIGHT)
    history: list = field(default_factory=list)

    @property
    def n_params(self) -> int:
        return int(self.delta.numel())

    @property
    def uid(self) -> str:
        return f"c{self.client_id}-u{self.update_id}"

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    def transition(self, new: UpdateStatus, t: float = 0.0, reason: str = "") -> None:
        if new not in _ALLOWED.get(self.status, set()):
            raise IllegalTransition(f"{self.uid}: {self.status.value} -> {new.value} is not allowed")
        self.status = new
        if reason:
            self.reason = reason
        self.history.append((new.value, float(t)))


@dataclass
class ControlMessage:
    """Server -> client downlink message about ONE update.

    kind 'receipt' : TRANSPORT ACK. The last bit arrived and the update was accepted into the buffer. NOT a commit:
                     it never advances the client's reference model.
    kind 'commit'  : final outcome AGGREGATED (the update's progress is part of global version ``server_version``).
    kind 'reject'  : final outcome REJECTED (reason in ``reason``).
    kind 'expire'  : final outcome EXPIRED in the USV buffer.
    """
    kind: str
    uid: str
    client_id: int
    update_id: int
    server_version: int
    reason: str = ""
    t_sent: float = 0.0

    def __post_init__(self):
        if self.kind not in ("receipt", "commit", "reject", "expire"):
            raise ValueError(f"unknown control message kind {self.kind}")

    @property
    def final(self) -> bool:
        return self.kind != "receipt"
