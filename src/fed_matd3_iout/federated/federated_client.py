"""AUV-side federated client: local actor/critics/replay, versions, update construction, global-model installation.

Update lifecycle on the client (see update_packet.py for the full state machine):

  * ONE outstanding update per client for its WHOLE life (``self.outstanding``): from creation until the client has
    processed a terminal outcome.  ``update_ready`` is False while an update is outstanding, so a second update can never
    start while the first one is uploading, in flight, buffered at the USV, or waiting for its outcome.
  * The reference model theta^base (what Delta is measured against) advances ONLY in ``handle_control`` on a ``commit``
    message (definitive successful aggregation).  A transport ``receipt``, a ``reject`` or an ``expire`` never moves it.
  * Counters: on commit the steps/transitions that were folded into the committed snapshot are subtracted (progress made
    after the snapshot is kept, because it is not in the new reference's delta yet).  On reject/expire nothing is reset:
    the progress is simply part of the next update's delta against the unchanged reference.
  * Bounded retry/back-off (protocol 2.1, ``federated.retry``; absent/None/enabled=false => protocol-2.0 behaviour):
    every consecutive FAILED update (rejected / expired) delays the next upload by
    ``min(max_s, base_s * factor**(k-1))`` seconds (k = consecutive failures, counted from the moment the client learns of the
    failure; deterministic, no jitter).  A commit or the installation of a newer global model resets k (a new reference is
    a new situation).  Back-off only DEFERS the start of the next upload: the update lifecycle, the reference model and the
    progress counters are untouched, and the unsent progress is simply folded into the next update's delta.
  * ``install_global`` (download of a newer global model) replaces the reference and bumps ``sync_epoch``; a commit that
    belongs to an older epoch therefore cannot overwrite the freshly installed reference.
"""
from __future__ import annotations

import torch

from ..agents.actor import flatten_params, load_flat_params
from ..agents.matd3_agent import MATD3Agent
from ..agents.replay_buffer import ClientReplay
from ..communication.model_transfer import ModelTransfer
from .update_packet import ControlMessage, UpdatePacket, UpdateStatus


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
        self.sync_epoch = 0                      # incremented by every install_global
        self.outstanding: UpdatePacket | None = None   # the single unresolved update (created -> client-resolved)
        self.upload: ModelTransfer | None = None
        self.ctrl: ModelTransfer | None = None         # downlink carrying a LIST of control messages (never overwritten)
        self.download: ModelTransfer | None = None
        self.awaiting_global = False             # sync mode barrier
        r = fcfg.get("retry") if hasattr(fcfg, "get") else None
        self.retry = None if (r is None or not bool(r.get("enabled", True))) else {k: float(v) for k, v in r.items() if k != "enabled"}
        self.consecutive_failures = 0
        self.retry_not_before = float("-inf")    # earliest time the next upload may start
        self._defer_open = False                 # a deferral episode (counted once) is in progress
        self.resolved_uids: set[str] = set()     # terminal outcomes already processed (exactly-once guard)
        self.stats = dict(attempted=0, succeeded=0, failed=0, deferred=0, timeouts=0, bytes_up=0.0, bytes_down=0.0,
                          tx_s=0.0, rx_s=0.0, latencies=[], sync_latencies=[], commits=0, receipts=0,
                          duplicate_ctrl=0, unknown_ctrl=0,
                          retries=0, retry_deferred=0, retry_deferred_steps=0, wasted_bytes_up=0.0, backoff_s=0.0)

    @property
    def model_params(self) -> int:
        return int(self.reference.numel())

    def update_ready(self) -> bool:
        return (self.outstanding is None
                and self.steps_since >= int(self.fcfg["local_steps_per_update"])
                and self.trans_since >= int(self.fcfg["min_local_transitions"]))

    # ------------------------------------------------------------------ retry / back-off
    def backoff_delay(self, k: int) -> float:
        """Delay after the k-th consecutive failure (k >= 1): base * factor^(k-1), capped at max_s."""
        r = self.retry
        return 0.0 if r is None or k <= 0 else min(r["max_s"], r["base_s"] * r["factor"] ** (k - 1))

    def note_failure(self, t: float) -> None:
        """A rejected / expired update has been processed by the client at time t."""
        self.consecutive_failures += 1
        if self.retry is not None:
            d = self.backoff_delay(self.consecutive_failures)
            self.retry_not_before = float(t) + d
            self.stats["backoff_s"] += d
        self._defer_open = False

    def retry_allows(self, t: float) -> bool:
        return self.retry is None or t >= self.retry_not_before

    def _reset_backoff(self) -> None:
        self.consecutive_failures = 0
        self.retry_not_before = float("-inf")
        self._defer_open = False

    # ------------------------------------------------------------------ creation
    def make_update(self, t: float, window_stats: dict, episode_id: int = -1) -> UpdatePacket:
        """Delta = theta_local - theta_base  (Eq. F1).  Refuses while another update is unresolved."""
        if self.outstanding is not None:
            raise RuntimeError(f"client {self.id}: update {self.outstanding.uid} is still unresolved "
                               f"({self.outstanding.status.value}); only one outstanding update is allowed")
        snap = flatten_params(self.agent.actor)
        delta = snap - self.reference
        self.update_counter += 1
        self.local_version += 1
        pkt = UpdatePacket(self.id, self.update_counter, self.base_version, delta, snap,
                           float(self.steps_since * self.batch_size), self.steps_since, self.trans_since,
                           float(delta.norm()), dict(window_stats), t_created=t,
                           size_bytes=float(delta.numel() * float(self.fcfg["model_bytes_per_param"])),
                           episode_id=int(episode_id), sync_epoch=self.sync_epoch)
        pkt.history.append((UpdateStatus.CREATED.value, float(t)))
        self.outstanding = pkt
        return pkt

    # ------------------------------------------------------------------ outcomes
    def expire_upload(self, t: float, reason: str = "upload_timeout") -> UpdatePacket:
        """Client-side abandon of an upload that did not finish (the only outcome a client may decide itself)."""
        pkt = self.outstanding
        if pkt is None or pkt.status is not UpdateStatus.UPLOADING:
            raise RuntimeError("only an UPLOADING update can be abandoned by the client")
        pkt.transition(UpdateStatus.EXPIRED, t, reason)
        self._resolve(pkt)
        self.note_failure(t)
        return pkt

    def handle_control(self, msg: ControlMessage, t: float | None = None) -> str:
        """Process one downlink control message. Idempotent: a repeated or unknown message changes nothing."""
        pkt = self.outstanding
        if msg.uid in self.resolved_uids:
            self.stats["duplicate_ctrl"] += 1
            return "duplicate"
        if pkt is None or pkt.uid != msg.uid:
            self.stats["unknown_ctrl"] += 1
            return "unknown"
        if msg.kind == "receipt":
            if not pkt.receipt_acked:
                pkt.receipt_acked = True
                self.stats["receipts"] += 1
            return "receipt"                      # transport ACK: NOTHING else changes (reference, counters, outstanding)
        if msg.kind == "commit":
            if self.sync_epoch == pkt.sync_epoch:          # no newer global model installed in between
                self.reference = pkt.snapshot.clone()
                self.steps_since = max(0, self.steps_since - pkt.local_steps)
                self.trans_since = max(0, self.trans_since - pkt.n_transitions)
            self.stats["commits"] += 1
            self.stats["succeeded"] += 1
            self._reset_backoff()
            self._resolve(pkt)
            return "commit"
        # reject / expire: reference and counters untouched
        self.stats["failed"] += 1
        if self.fcfg["mode"] == "sync":
            self.awaiting_global = False
        self._resolve(pkt)
        self.note_failure(msg.t_sent if t is None else t)
        return msg.kind

    def _resolve(self, pkt: UpdatePacket) -> None:
        pkt.client_resolved = True
        self.resolved_uids.add(pkt.uid)
        self.outstanding = None

    # ------------------------------------------------------------------ global model
    def install_global(self, theta: torch.Tensor, version: int, blend: float) -> None:
        cur = flatten_params(self.agent.actor)
        # blend == 1 must be an exact replacement: 1*theta + 0*cur would turn a non-finite LOCAL parameter into NaN
        # in the freshly installed model (0*nan = nan), silently corrupting the actor that is supposed to be reset.
        new = theta.clone() if blend >= 1.0 else blend * theta + (1.0 - blend) * cur
        load_flat_params(self.agent.actor, new)
        load_flat_params(self.agent.actor_t, new)          # keep the target actor consistent with the synced policy
        self.reference = new.clone()
        self.base_version = int(version)
        self.sync_epoch += 1
        self._reset_backoff()
        self.steps_since = 0
        self.trans_since = 0
