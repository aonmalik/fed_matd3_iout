# Protocol 2.2 decision log (CANDIDATE)

**Nothing below has been approved by the author.** D7–D10 are *proposed/provisional* decisions that were implemented so that a small pilot can run;
each can be reversed by config where noted. D1–D6 are referenced, not restated or altered: their full text is in `docs/AUTHOR_DECISIONS.md`
(section headings D1–D6), they are all **OPEN**, and every default stays as in Protocol 2.1. Protocol 2.1 is frozen in `configs/protocols/v2.1/`
(config_hash `c76394bf7eaa9005`). Status vocabulary: `implemented` (code exists), `tested` (automated tests pass), `provisional`, `requires approval`.

## D1–D6 (reference only)
| ID | Subject | Status | Where |
|---|---|---|---|
| D1 | semantic maturity threshold `federated.maturity_eps` (0.05) | requires approval (open, unchanged) | AUTHOR_DECISIONS.md |
| D2 | undefined relevance components (drop the undefined one) | requires approval (open, unchanged) | AUTHOR_DECISIONS.md |
| D3 | rejected/expired statistics return policy | requires approval (open, unchanged) | AUTHOR_DECISIONS.md |
| D4 | client-side expiry after transmission (none) | requires approval (open, unchanged) | AUTHOR_DECISIONS.md |
| D5 | actor Adam state after global install (not reset) | requires approval (open, unchanged) | AUTHOR_DECISIONS.md |
| D6 | retry/back-off parameters (+ relevance dynamic-range concern) | requires approval (open, unchanged) | AUTHOR_DECISIONS.md |

## D7 — Duplex and airtime convention
* **Statement / selected option (provisional):** half-duplex is the default (`acoustic.usv_link.duplex: half`); `full` stays selectable.
* **Scope found in the code (inspected, not assumed):** the half-duplex restriction is *per AUV USV-link modem*: each AUV has one `LinkProcess` link to the USV, one transmit power (`p_elec_w`) and one receive power (`rx_power_w`), hence ONE airtime budget per AUV that covers sensor-data uplink + model upload (TX) and control + download (RX). It is **not** per acoustic link pair in general: the sensor↔AUV link (25 kHz, `acoustic.sensor_link`) is a separate modem/frequency and is not pooled. The USV side is not modelled as a shared resource (gap: no USV-side contention, no USV modem airtime).
* **Convention:** time division. Quantities are fractions of one step `dt`. `model_share` is a *pool* for ALL model traffic of an AUV (2.1 gave each transfer the full share). In a step, grants satisfy: Σ grants ≤ min(pool, 1 − airtime already used by the sensor-data uplink in that step); each grant ≤ the transfer's demand. Hence TX and RX never overlap and sensor + model airtime ≤ dt.
* **Priority / fairness (`communication/capacity_arbiter.py`):** `ctrl` is served first up to its demand but never more than HALF the pool while another transfer waits (so no starvation); the rest is split by max-min fairness (equal split, surplus of small demands re-assigned). Capacity a transfer does not need is re-assigned to the others *within the same step and pool*; it is never granted without demand. Unused pool is not transferred to the sensor-data uplink inside the step (the sensor uplink is held to `1 − model_share` while model traffic is active; one-step lag because the environment steps before the protocol).
* **Direction effect:** half: any active model transfer reserves `model_share` of the sensor uplink. full: only an upload reduces the sensor uplink; control+download share a separate downlink pool (the same `model_share` number — no new parameter was invented).
* **Airtime/energy:** granted airtime = grant × dt; used airtime = bytes moved / full link rate (a completing transfer is charged only what it used). Energy = `p_tx_elec_w` × tx airtime, `rx_power_w` × rx airtime (`env.add_comm_energy`). Same accounting for every federated method; non-federated methods have no model traffic.
* **Included / excluded:** propagation delay is included in arrival time, NOT in airtime. Fragments lost to fading/outage: no progress and no airtime charged for that step; fragments already received are kept (`allow_fragmentation`); the byte-level retransmission of a corrupted fragment is not modelled. Protocol overhead = `ack_bytes` for control messages only; no headers/preambles/guard times are modelled (**gap**: no modem turnaround time, no guard interval — placeholders required if the real modem needs them).
* **Reservation rule (review correction R4):** a transfer that has sent its last bit and only waits for propagation occupies no airtime and does not reserve it; reservations count only transfers still sending (`tests/test_reservation_flag.py`). `model_share` must be finite and within [0, 1] (`tests/test_model_share_validation.py`).
* **Implementation:** `communication/capacity_arbiter.py`, `communication/model_transfer.py`, `federated/communication_protocol.py` (`_grants`), `environment/multi_auv_iout_env.py` (reservation, `sensor_uplink_airtime_s`). **Config:** `acoustic.usv_link.model_share`, `acoustic.usv_link.duplex`.
* **Tests:** `tests/test_comm_capacity.py`, `tests/test_d7_half_duplex.py` (conflicting TX/RX, control priority, fairness, exhaustion, unused capacity, direction change, airtime/energy, determinism, sensor-airtime exclusivity).
* **Consequences vs 2.1:** federated-method results are not comparable (time-division energy ≈ half of 2.1's charge per byte at share 0.5; lower combined airtime).
* **Unresolved:** true modem duplex capability unknown. If the intended modem is full duplex, this must be approved (set `duplex: full`) — not changed silently. **Status: implemented, tested, provisional, requires approval.**

## D8 — Final drain policy
* **Stages (recorded by `Trainer.finalize()` → `termination` in `evaluation.json`):** (1) training cutoff (`t_training_cutoff_s`), (2) pending-update census (`pending_at_cutoff`), (3) drain start, (4) bounded drain (`drain_s`, ≤ `final_drain_s`), (5) drain termination (`drain_exhausted_budget`, resolved count `aggregated_during_drain`), (6) evaluation start (`t_evaluation_start_s`, `pending_at_evaluation_start`). Unresolved updates at evaluation start are reported, never discarded or forced.
* **Drain content:** the environment keeps stepping with the current deterministic actors (links evolve); NO gradient step, no replay storage, no new federated update (`proto.accept_new=False`), no extra episode. The training budget is unchanged.
* **Bound analysis (`federated/drain_bound.py`, mode-aware):** a deterministic SERVER-side resolution bound exists: `upload_timeout_s + max propagation + min(timeout, max_buffer_age_s) + one step`, where `timeout` is `agg_timeout_s` in **asynchronous** mode and `sync_timeout_s` in **synchronous** mode (`max_buffer_age_s` caps the buffered dwell). Current values: **async = 400 + 0.073 + 120 + 1 = 521.1 s** (the configured mode); **sync = 400 + 0.073 + 300 + 1 = 701.1 s** (a conservative bound; the 600 s drain does NOT cover it, so a synchronous pilot needs its own explicit `final_drain_s`). The bound concerns server decisions only. The final control message (commit/reject/expire) to the client needs a downlink contact; contact and Gilbert–Elliott outages have no deterministic bound, so **the bound does not guarantee delivery of the final downlink notification** and no total bound exists.
* **Selected (provisional):** `federated.final_drain_s: 600.0` (= `max_buffer_age_s`; ≥ the async bound 521.1 s, < the sync bound 701.1 s) — an explicit *pilot setting*, not a universal horizon and not taken from the smoke experiment; **needs sensitivity analysis**. 2.1 behaviour (0 s) is frozen in `configs/protocols/v2.1`; setting 0 reproduces cutoff-only evaluation.
* **Tests:** `tests/test_drain_bound_modes.py` (async/sync, cap, terms), `tests/test_d8_drain_policy.py` (zero/one/multiple pending, completion within horizon, unresolved at expiry, zero-length drain, no training during drain, deterministic evaluation start), `tests/test_final_drain.py`. **Status: implemented, tested, provisional, requires approval.**
* **Consequence:** 2.2 federated global models can include updates that 2.1 would have left pending; not comparable with 2.1.

## D9 — Training compute energy (MAPPO and others)
* **Selected (provisional):** compute energy is reported in joules **only from a real measurement**; otherwise `status: not_measured`, `energy_j: null` (never 0), with measured compute wall/CPU time reported separately (`training_compute` in `evaluation.json`).
* **Inspection:** no hardware instrument exists in the build environment (no RAPL powercap counters, no GPU power interface). Hence energy is `not_measured` for every method here. Measurement boundary when an instrument exists: learner gradient-update computation only, same for all methods; idle power NOT subtracted (stated in the report); package counters include other host processes; counters read at update boundaries (no sampling frequency); trapezoid integration for power samples; invalid readings / gaps are *missing intervals* and downgrade the status to `partial`.
* **Cross-method:** the MATD3 family charges the ASSUMED `energy.e_grad_j` per gradient step (a model parameter, not a measurement); MAPPO charged nothing. These are not comparable and not mixed with measurements ⇒ **cross-method training total-energy comparisons are INCOMPLETE**; no energy-efficiency claim is allowed. Mission and evaluation energy are unaffected.
* **Implementation:** `utils/compute_energy.py`, `training/trainer.py`, `baselines/mappo.py`, `validation/fairness.py` (N11). **Tests:** `tests/test_d9_compute_energy.py`. **Status: implemented, tested, provisional, requires approval.**

## D10 — Fixed-workload completion metric
* **Selected (provisional):** added `fixed_workload_completion_time_s` (+ `fixed_workload_censored`, `fixed_workload_delivered_fraction`) as a separate measurement; `backlog_clearance_time_s` and its censoring are untouched.
* **Definition (`evaluation/fixed_workload.py`):** workload = initial sensor buffers of the evaluation scenario (same seed for every method); arrivals disabled (`traffic.scale = 0`); authoritative completion event = unit **delivered to the USV** (collected / assigned / onboard / transmitted / acknowledged do not count); clock starts at reset, stops at the end of the first step with delivered ≥ W (resolution = one step; includes collection, travel, queueing and uplink incl. airtime shared with model traffic); timeout = episode horizon ⇒ right-censored (kept, value = timeout); lost units ⇒ censored `failed_units`; W = 0 ⇒ completes at 0, flagged `zero_workload`; arrivals detected ⇒ `fixed_workload_violated`. Aggregation: scenarios → mean per training seed → seeds as units; the mean of censored values is a lower bound, so always report the censoring rate.
* **Config:** `experiment.fixed_workload_eval: true`. **Tests:** `tests/test_d10_fixed_workload.py`. **Comparability:** valid only under identical workload and measurement conditions; methods still differ by the declared non-equivalences.
* **Status: implemented, tested, provisional, requires approval.** Choice of workload (initial buffers) and of the threshold-free completion rule is a proposal.

### D8 — Drain Semantics and Continued Mission Dynamics

*Status: documents the existing implementation (inspected, not changed). Provisional like the rest of D8; not approved by the author.*

A drain step is a **full `env.step()`** on the training environment (`Trainer.finalize()`), with learning frozen (`proto.accept_new=False`). It is **not** a communication-only freeze. Retained because pending uploads/aggregation/downloads need simulated time and the model-share pool shares the half-duplex modem with the sensor uplink (D7); freezing the environment would remove that coupling. The three activity classes are kept distinct:

| # | Question | Answer during drain | Class |
|---|---|---|---|
| 1 | Sensor arrivals continue? | **Yes** (Poisson arrivals, same RNG stream) | mission-state evolution |
| 2 | AUV movement continues? | **Yes** (current deterministic local actors, no exploration noise; episode truncation is ignored) | mission-state evolution |
| 3 | Task allocation continues? | **Yes** | mission-state evolution |
| 4 | Task execution / service continues? | **Yes** (collection, service, hover) | mission-state evolution |
| 5 | Sensor-data uplink continues? | **Yes** (and still takes airtime from the model pool, D7) | mission-state evolution |
| 6 | Communication transfers continue? | **Yes**: uploads, control, aggregation, downlink progress, retries/expiry apply | communication progress |
| 7 | New federated updates generated? | **No** | training/learning |
| 8 | Gradient computation / policy optimisation? | **No** (also no replay-buffer writes, no new episodes) | training/learning |
| 9 | Drain energy in reported energy? | **No**: not in logged episode energy nor evaluation energy; recorded separately in `termination["drain_mission"]` (deltas of generated/delivered/lost Mbit and propulsion/tx/rx/compute J) | energy |
| 10 | Evaluation starts when? | **After the drain ends** (all updates resolved, or the bounded budget expired), on freshly reset scenario episodes (the evaluator builds a new environment; no drain state carries over) | ordering |

Unresolved updates at drain expiry are never discarded silently: they are counted in `unresolved_at_drain_expiry`, `pending_at_evaluation_start`, and `evaluation_started_with_unresolved`. The record is machine-readable in `termination["drain_semantics"]` (constant `DRAIN_SEMANTICS`). Verified by `tests/test_d8_drain_semantics.py` and `tests/test_d8_drain_policy.py`, `tests/test_final_drain.py`. Open question for the author: whether a communication-only freeze (environment not stepped) is preferred; this would be a protocol change.
