"""Batch-level attribution of delivered semantic utility to the AUV (client) that collected the data.

Why this exists (audit item 4).  The old relevance score divided ``U_del`` (utility delivered inside a time window) by
``V_coll`` (value collected inside the same window).  Data collected in window w is usually delivered in window w+1 or
later, so the two sums describe different packets: delayed delivery was credited to the wrong window.

Mechanism (consistent with the environment).  In ``MultiAUVIoUTEnv`` an AUV only ever uploads data it collected itself
(there is no AUV-to-AUV hand-off), so ``Batch.collected_by`` identifies the originating client for the whole life of a
batch.  Every collected chunk is recorded here under the key ``(pkt_id, t_collect)``; ``Batch.split`` preserves both,
so partial uploads of one chunk map back to the same entry.  Delivery utility is added to the SAME entry.

Maturity (how incomplete batches are treated).  An entry is claimed for relevance only once it is *mature*:
  * fully delivered (units_del >= units_coll), or
  * older than  H_k = ln(1/eps) / mu_k  seconds since collection  (its remaining VoI is <= eps of the initial value, so
    waiting longer cannot change its realised utility by more than a fraction eps).  eps = ``federated.maturity_eps``.
Immature entries are *not* counted as failures: they stay open and are claimed by a later update, when they mature.
A batch with mu = 0 never decays and therefore matures only through full delivery (or is censored at episode end).

Exactly-once.  A claimed entry is removed from the open set, so its collected value enters exactly one update's
statistics.  If a claimed-by-age entry is delivered later, the late remainder is *not* re-attributed (it is recorded in
``late_units``); by construction its value is <= eps of the batch.  A tombstone keeps the late delivery from being
mistaken for an unknown key.

Boundedness.  For a mature set S:  U_mature = sum_S delivered  c sigma exp(-mu age)  <=  sum_S units_del c sigma
<= V_mature = sum_S units_coll c sigma, because age >= 0 and units_del <= units_coll.  Hence
rho_sem = U_mature / V_mature lies in [0, 1] *without clipping*.

Episode boundary.  ``reset_episode`` censors every unclaimed entry (it is counted in ``censored_*`` diagnostics and never
attributed): the environment's onboard buffers are discarded at reset, so those batches can no longer be delivered.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .packet import Batch

_TOL = 1e-6   # units


@dataclass
class _Entry:
    t_collect: float
    mu: float
    V: float = 0.0            # undecayed value  sum units * c * sigma  of the collected chunk
    units: float = 0.0        # units collected
    U: float = 0.0            # decayed utility delivered so far
    units_del: float = 0.0    # units delivered so far


class BatchBook:
    def __init__(self, n_auv: int, maturity_eps: float = 0.05):
        if not (0.0 < maturity_eps < 1.0):
            raise ValueError("maturity_eps must be in (0, 1)")
        self.n = n_auv
        self.eps = float(maturity_eps)
        self.open: list[dict] = [dict() for _ in range(n_auv)]       # key -> _Entry (collected, unclaimed)
        self.tomb: list[dict] = [dict() for _ in range(n_auv)]       # key -> undelivered units after claim-by-age
        self.late_units = [0.0] * n_auv
        self.late_U = [0.0] * n_auv                  # utility delivered after its entry was claimed by age (mass balance only)
        self.claimed_U = [0.0] * n_auv               # running totals of everything ever claimed (mass balance)
        self.claimed_V = [0.0] * n_auv
        self.claimed_units_del = [0.0] * n_auv
        self.censored_V = 0.0
        self.censored_units = 0.0
        self.censored_U = 0.0
        self.claimed_entries = [0] * n_auv

    # ---------------------------------------------------------------- events
    def on_collect(self, auv: int, b: Batch) -> None:
        key = (b.pkt_id, b.t_collect)
        e = self.open[auv].get(key)
        if e is None:
            e = self.open[auv][key] = _Entry(float(b.t_collect), float(b.mu))
        e.V += b.units * b.c * b.sigma
        e.units += b.units

    def on_deliver(self, auv: int, chunk: Batch, utility: float) -> str:
        """Attribute the utility of one delivered chunk to the entry it was collected into."""
        key = (chunk.pkt_id, chunk.t_collect)
        e = self.open[auv].get(key)
        if e is not None:
            e.U += utility
            e.units_del += chunk.units
            return "attributed"
        rem = self.tomb[auv].get(key)
        if rem is not None:                                   # delivered after its entry was claimed by age
            self.late_units[auv] += chunk.units
            self.late_U[auv] += utility
            rem -= chunk.units
            if rem <= _TOL:
                del self.tomb[auv][key]
            else:
                self.tomb[auv][key] = rem
            return "late"
        raise KeyError(f"delivered chunk {key} was never recorded as collected by AUV {auv}")

    # ---------------------------------------------------------------- claiming
    def _mature(self, e: _Entry, now_s: float) -> bool:
        if e.units_del >= e.units - _TOL:
            return True
        if e.mu > 0.0 and (now_s - e.t_collect) >= math.log(1.0 / self.eps) / e.mu:
            return True
        return False

    def claim(self, auv: int, now_s: float) -> dict:
        """Return (and remove) the mature entries of ``auv``.  Each entry is claimed at most once."""
        U = V = ud = uc = 0.0
        n_m = n_i = 0
        for key, e in list(self.open[auv].items()):
            if self._mature(e, now_s):
                U += e.U
                V += e.V
                ud += e.units_del
                uc += e.units
                n_m += 1
                del self.open[auv][key]
                rem = e.units - e.units_del
                if rem > _TOL:
                    self.tomb[auv][key] = rem
            else:
                n_i += 1
        self.claimed_entries[auv] += n_m
        self.claimed_U[auv] += U
        self.claimed_V[auv] += V
        self.claimed_units_del[auv] += ud
        return {"U_mature": U, "V_mature": V, "units_del_mature": ud, "units_coll_mature": uc,
                "n_mature_batches": float(n_m), "n_immature_batches": float(n_i)}

    # ---------------------------------------------------------------- bookkeeping
    def pending(self, auv: int) -> dict:
        es = self.open[auv].values()
        return {"V": sum(e.V for e in es), "units": sum(e.units for e in es), "n": len(self.open[auv]),
                "U": sum(e.U for e in es), "units_del": sum(e.units_del for e in es)}

    def utility_accounted(self, auv: int) -> float:
        """Delivered utility of ``auv`` that the book knows about: claimed + still open + delivered after claim-by-age.
        Mass balance: equals ``UtilityLedger.by_auv[auv]`` (modulo entries dropped by reset_episode, see censored_U)."""
        return self.claimed_U[auv] + self.pending(auv)["U"] + self.late_U[auv]

    def reset_episode(self) -> None:
        for a in range(self.n):
            for e in self.open[a].values():
                self.censored_V += e.V
                self.censored_units += e.units
                self.censored_U += e.U
            self.open[a].clear()
            self.tomb[a].clear()
