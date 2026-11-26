"""Deterministic SERVER-SIDE upper bound for resolving an update that is pending at the training cutoff (D8 derivation).

For an update created at or before the cutoff, every transition to a SERVER-DECIDED terminal state is bounded by configured timers:
  UPLOADING : expires ``upload_timeout_s`` after its start (communication_protocol.step), or completes earlier;
  IN_FLIGHT : arrives after at most max propagation delay = usv_link.range_m / sound_speed_mps (slant distance <= contact range);
  BUFFERED  : aggregated/rejected once the oldest buffered update is ``agg_timeout_s`` old (USVServer.ready), otherwise it
              expires after ``max_buffer_age_s``.  During a drain no new update arrives, so the timeout path decides.
Bound (from the cutoff), mode dependent:
  async :  upload_timeout_s + max_propagation + min(agg_timeout_s,  max_buffer_age_s) + one step      (USVServer.ready: oldest buffered age >= agg_timeout_s)
  sync  :  upload_timeout_s + max_propagation + min(sync_timeout_s, max_buffer_age_s) + one step      (USVServer.ready: all clients present, or round older than sync_timeout_s)
``max_buffer_age_s`` caps the buffered dwell (``expire_buffer`` runs before ``ready``), so the buffered term is the smaller of the two; a
valid config keeps max_buffer_age_s above both timeouts, hence the timeout term normally decides.  The sync bound is a CONSERVATIVE
upper bound (the round clock starts before the cutoff, so the real wait is often shorter).  The 'one step' term is the final simulation
step in which the server acts.

NOT bounded: the final control message ('commit'/'reject'/'expire') travelling back over the AUV's downlink.  It needs a contact
window; contact depends on AUV motion/range and a Gilbert-Elliott outage with geometric duration, so no deterministic bound exists.
A drain can therefore end with updates that the server has decided but the client has not heard about
(``awaiting_terminal_outcome_at_client``); they are reported, never forced.
"""
from __future__ import annotations


def drain_bound_components(cfg, mode: str | None = None) -> dict:
    """Terms of the server-side bound for ``mode`` ('async' | 'sync'; default = ``federated.mode``)."""
    f, u, a = cfg["federated"], cfg["acoustic"]["usv_link"], cfg["acoustic"]
    mode = str(f["mode"] if mode is None else mode)
    if mode not in ("async", "sync"):
        raise ValueError(f"drain bound: unknown federated mode {mode!r}")
    timeout_key = "agg_timeout_s" if mode == "async" else "sync_timeout_s"
    max_age = float(f.get("max_buffer_age_s", float("inf")))
    comp = {"mode": mode, "upload_timeout_s": float(f["upload_timeout_s"]),
            "max_propagation_s": float(u["range_m"]) / float(a["sound_speed_mps"]),
            "buffered_dwell_s": min(float(f[timeout_key]), max_age), "timeout_key": timeout_key,
            "final_step_s": float(cfg["env"]["dt_s"])}
    comp["bound_s"] = comp["upload_timeout_s"] + comp["max_propagation_s"] + comp["buffered_dwell_s"] + comp["final_step_s"]
    return comp


def server_side_drain_bound_s(cfg, mode: str | None = None) -> float:
    """Server-side bound in seconds (see module docstring); does NOT cover the final downlink notification to the client."""
    return float(drain_bound_components(cfg, mode)["bound_s"])
