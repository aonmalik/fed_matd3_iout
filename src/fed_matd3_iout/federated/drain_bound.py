"""Deterministic SERVER-SIDE upper bound for resolving an update that is pending at the training cutoff (D8 derivation).

For an update created at or before the cutoff, every transition to a SERVER-DECIDED terminal state is bounded by configured timers:
  UPLOADING : expires ``upload_timeout_s`` after its start (communication_protocol.step), or completes earlier;
  IN_FLIGHT : arrives after at most max propagation delay = usv_link.range_m / sound_speed_mps (slant distance <= contact range);
  BUFFERED  : aggregated/rejected once the oldest buffered update is ``agg_timeout_s`` old (USVServer.ready), otherwise it
              expires after ``max_buffer_age_s``.  During a drain no new update arrives, so the timeout path decides.
Bound (from the cutoff):  upload_timeout_s + max_propagation + agg_timeout_s + one step.

NOT bounded: the final control message ('commit'/'reject'/'expire') travelling back over the AUV's downlink.  It needs a contact
window; contact depends on AUV motion/range and a Gilbert-Elliott outage with geometric duration, so no deterministic bound exists.
A drain can therefore end with updates that the server has decided but the client has not heard about
(``awaiting_terminal_outcome_at_client``); they are reported, never forced.
"""
from __future__ import annotations


def server_side_drain_bound_s(cfg) -> float:
    f, u, a = cfg["federated"], cfg["acoustic"]["usv_link"], cfg["acoustic"]
    prop = float(u["range_m"]) / float(a["sound_speed_mps"])
    dt = float(cfg["env"]["dt_s"])
    return float(f["upload_timeout_s"]) + prop + float(f["agg_timeout_s"]) + dt
