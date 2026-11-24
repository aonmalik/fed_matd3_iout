# Author decisions (protocol 2.1 preparation)

**Status of every item below: OPEN. No option has been chosen on the author's behalf.** Where the code currently behaves in one way, that is
the *default in effect* inherited from earlier phases, listed so it is visible, not an endorsement. Any change to a methodological item changes
results and therefore requires a new protocol version (2.2), a changelog entry, a re-frozen `config_hash` and a re-run of
`scripts/final_validation.py`. Evidence quoted comes from `docs/final_validation_report.json` / `docs/aggregation_diagnostics.json`; none of it is
a performance result.

## D1. Maturity threshold `federated.maturity_eps` (currently 0.05, horizon H = ln(1/eps)/mu)
A collected batch counts toward rho_sem once it is fully delivered or older than H (then waiting cannot change its value by more than a fraction eps).
| Option | Consequence |
|---|---|
| A. keep 0.05 (a-priori value, not tuned) | Measured insensitivity: pooled rho_sem 0.1095-0.1121 for eps 0.01-0.5; Spearman rank vs eps=0.05 >= 0.96; censored value share 2.9% (eps 0.01) -> 1.1%; late-delivered unit share 0.17% -> 3.0%. |
| B. smaller eps (0.01) | Longer horizon: more batches stay immature inside an update window, fewer defined rho_sem, more censoring at episode end; fewer late deliveries. |
| C. larger eps (0.2-0.5) | Shorter horizon: relevance available sooner, but more batches are claimed before delivery (late-delivery share grows to ~3%) and counted as failures. |
Not to be chosen by downstream performance (that would be tuning on the test objective). The measurement shows the metric hardly depends on it.

## D2. Undefined relevance components (currently: drop the undefined component, use the other; both undefined -> M = 0)
| Option | Consequence |
|---|---|
| A. drop (current) | A client with no matured batch is not punished for data that simply has not matured; but M then reflects rho_svc alone, a different scale. |
| B. score undefined as 0 | M = (1-beta)*rho_svc when rho_sem undefined: halves M for such updates (beta = 0.5), conflating "no evidence" with "failure". |
| C. carry the last defined value forward | Smooths M, but uses stale evidence and couples windows. |
Measured: with untrained policies rho_sem is undefined in ~17 of 18 sampled updates and rho_svc is mostly exactly 0 (nothing collected), so M = 0 either way; with the goal-seeking reference policy both are defined in 100% of windows and the options coincide. The choice therefore matters mainly in the transition regime, which has not been measured.

## D3. Rejected/expired statistics: return vs discard (currently: return to the window if same episode; discard if the episode changed)
| Option | Consequence |
|---|---|
| A. return (current) | Evidence is conserved and counted once in a later update; a persistently failing client's windows keep growing and its next update describes a longer window. |
| B. discard | Simpler, but the batches were claimed once (claim-once ledger) so their evidence is lost permanently and rho_sem is biased toward whatever happened in non-failed windows; failures become invisible to the relevance of the same client. |
| C. return but cap window length | Bounds the growth in A; needs a cap parameter. |
Retry back-off (protocol 2.1) makes failure streaks longer in time, which makes A's growing windows more relevant than before.

## D4. Client-side expiry after transmission (currently: none; after the last bit leaves, only the USV decides)
| Option | Consequence |
|---|---|
| A. none (current) | No ambiguity about the update's fate; but the client stays blocked until it hears a terminal outcome over the acoustic downlink, so a client out of contact can be blocked for a long time. |
| B. client timeout T_ack after the last bit | Frees the client, but the USV may still aggregate the update: the client would then hold a reference that does not include progress the global model already contains (double application of its progress on the next update). Requires an idempotent protocol (the update uid is already unique) and a rule for a late commit. |
| C. B + late-commit reconciliation (client applies a late commit to the reference if the epoch is unchanged) | Correct but more protocol surface; needs new lifecycle tests. |

## D5. Actor Adam state after installing the global actor (currently: moments NOT reset, target actor hard-copied)
| Option | Consequence |
|---|---|
| A. keep moments (current) | Moments were accumulated for the old local weights; the first steps after install are scaled by statistics that no longer match the weights (can be a kick or a damping). No extra code. |
| B. reset the actor optimiser on install | Clean restart; with Adam the first steps are ~lr-sized sign steps, i.e. a transient; the effect is larger the more often installs occur (every download). |
| C. average/transfer moments | Not standard; adds server/communication state (not modelled). |
All options keep critics and critic optimisers local. Impact unmeasured: needs a pilot ablation.

## D6. Items raised by this phase
* **Retry/back-off parameters** (base 120 s = agg_timeout_s, factor 2, cap 480 s; reset by commit or by a newer global install; no jitter; no hard suspension). Anchored to protocol time scales, not tuned. Alternatives: other base/cap; hard suspension after N failures until a new global arrives (stops all waste but can exclude a client permanently if no aggregation happens); jitter (irrelevant here: the scheduler already arbitrates the shared link). Measured effect: see section 2 of `docs/FINAL_SCIENTIFIC_AUDIT.md`. **Approve or change.**
* **Relevance dynamic range (risk R2).** Measured: untrained policies -> M = 0 for ~all updates (factor stuck at the floor eps_M, relevance inert); goal-seeking reference policy -> M ~ 0.54 with between-AUV spread ~0.05, i.e. weight TV ~ 1.5% vs uniform. With symmetric AUVs the formula, as designed, cannot separate clients much. Options (NOT implemented, each is a protocol change): normalise M within a round (e.g. divide by the round mean), raise eta, lower eps_M, or study heterogeneous clients (client_speed < 1, uneven sensor density) where relevance can differ. Do not change before the pilot.
* **Pilot scope** (episodes, evaluation scenarios, optional longer single-method run for relevance dynamics): `scripts/run_pilot_2seeds.py`.
* **Seed count 10** was requested and is implemented. It makes a Holm-corrected exact Wilcoxon *attainable* (minimum p = 2^-9 = 0.00195; x10 comparisons = 0.0195 < 0.05); it guarantees nothing about significance or power.

---
# Added in the Protocol 2.2 candidate (D1-D6 above are untouched and remain OPEN)

**Status vocabulary.** D7-D10 below were given as *provisional* decisions in the Protocol 2.2 finalisation instructions. They are implemented as
provisional defaults. They are NOT recorded as formally approved by the author; the full records (selected option, rationale, locations, tests)
are in `docs/PROTOCOL_2.2_DECISION_LOG.md`.

## D7. Channel duplex / airtime model  -- PROVISIONAL: half duplex (`acoustic.usv_link.duplex: half`)
Scope found in the code: ONE airtime pool PER AUV, covering that AUV's USV-link modem (sensor-data uplink + model upload = TX, control + download = RX). The sensor<->AUV link (25 kHz) is separate and not pooled; the USV side is not modelled as a shared resource (gap). `full` remains selectable. Airtime convention: time division (granted/used airtime) instead of 2.1's "full step at reduced rate" (about 2x energy per byte at share 0.5); this changes federated-method results versus 2.1. Needs approval if the real modem is full duplex.

## D8. Final communication drain  -- PROVISIONAL: bounded drain, `federated.final_drain_s: 600.0`
A deterministic SERVER-side bound exists (521.1 s: upload timeout 400 + propagation 0.073 + aggregation timeout 120 + 1 step; `federated/drain_bound.py`); the client-notification leg over the downlink has none (contact/outage unbounded), so 600 s (= `max_buffer_age_s`) is an explicit PROVISIONAL pilot value needing sensitivity analysis, not a universal horizon. 2.1 behaviour (0 s) is frozen in `configs/protocols/v2.1/`. No learning, storage or new updates during the drain; unresolved updates are reported, never forced.

## D9. MAPPO / training compute energy  -- PROVISIONAL: joules when measured, otherwise `not_measured`
No power instrument exists in the build environment (no RAPL powercap, no GPU interface) => compute energy is `not_measured` (None, never 0) for every method; measured compute TIME is reported separately (`training_compute`). The MATD3 family's modelled `e_grad_j` (an assumption) is not a measurement. Cross-method training total-energy comparisons are INCOMPLETE; no energy-efficiency claim is permitted from them.

## D10. Fixed-workload completion metric  -- PROVISIONAL: added as a separate metric
`fixed_workload_completion_time_s` (+ censored flag, delivered fraction) with arrivals disabled and the scenario's initial buffers as the identical workload; completion = delivery to the USV. `backlog_clearance_time_s` is unchanged. Details: `evaluation/fixed_workload.py`.
