# Assumptions and Limitations

## Honest status
* **No scientific result is claimed.** Unit/integration tests and a smoke test pass; a 25-episode pilot (single seed) showed the learned policies did **not** yet solve the task (<1 Mbit
  delivered vs ~85 Mbit for the goal-seeking reference controller) - far below Paper 2's 500-episode budget. Whether S3A-FMATD3 helps is **untested**.
* `~/dt_smarl_step8/` was not available in the build workspace, so Phase 1 inspection and code reuse were not possible; models were re-implemented from the equations in the two papers.
  The frozen Paper-2 manifest was not read (`configs/experiment_manifest.json` is new).

## Parameters taken from the papers (verified)
Region 200 m, Q=2 Mbit / 5000 units, rates {3,5,8,12}x3, AUV and sensor depth 50 m, speeds [1.2,2.2] (v_nom 1.7), R_dc 6 m, d_safe 5 m, 1000 steps, STA weights (1,1,0.05), gamma 0.97,
hidden/batch 128, lr 1e-3, tau 1e-3, replay 20000, policy delay 2, noise 0.8, 500 episodes (P2 Table I); p_c .3, p_h .4, gamma_crit 2.5, gamma_low .5, mu .05/.01, omega .3/.6/1,
f 12 kHz, B 4 kHz, k 1.5, Thorp, Rayleigh, v_s 1500, USV 5 m/s (SAC-TD3).

## Assumptions (not in the papers; all exposed in `configs/`)
Sensor-link frequency 25 kHz; all transmit powers, efficiencies, SNR thresholds, modem rate caps (0.05 Mbit/s); AUV-USV range 110 m; Gilbert-Elliott outage; onboard buffer 20 Mbit and
offload thresholds; initial buffer law U(0,0.5); mu of the 'high' category; propulsion coefficients (P0=20 W, kappa=8); tx/rx/compute powers; max_hover_steps=60; USV centroid relay rule
(P2 uses FIM-based relocation); static semantic hotspot weights (P1 uses a Markov chain); model payload = parameters x 4 bytes.

## Known limitations
* Propulsion/communication/computation constants are placeholders: **calibrate** before interpreting energy results. Evaluation episodes contain no learning, so E_comp = 0 there and model traffic is absent.
* Straight-line, constant-speed travel-time prediction; exact backlog expectation conditional on that T.
* Centralised critics use simulator state (CTDE); critics are not federated.
* Relevance statistics are windowed; rho_sem may saturate; own-sensor overflow depends on the allocator. The weighting changes the effective objective - **not unbiased FedAvg**; no convergence proof.
* No privacy mechanism or threat model; clipping without noise gives no privacy guarantee.
* Potential-based shaping is not strictly policy-invariant under target switching.
* Single link process per AUV, no inter-AUV acoustic interference; unsafe-proximity counts are indicators, not verified collisions.
* The federation clock continues across episodes while layouts reset each episode.
* Reward differs from Paper 2 Eq.(14); only the collision term has a dedicated reward unit test.


## Federated protocol semantics after the audit repair

**Lifecycle.** `created -> uploading -> in_flight -> buffered -> aggregated | rejected | expired`. Terminal states have no outgoing transition, so each outcome is produced once; update ids are `(client, counter)` and duplicates are ignored by the USV.
Client decides only `created/uploading -> expired` (upload timeout). From `in_flight` the USV decides, because once the last bit has left the AUV the client can no longer withdraw it.
A BUFFERED update older than `federated.max_buffer_age_s` expires at the USV.

**ACK versus commit.** The USV sends `receipt` (transport ACK) when the update is buffered, and one final message: `commit` (carries the new global version), `reject` (reason) or `expire`. Messages wait in a per-client outbox and travel as one list per downlink transfer, so none overwrites another.
The client's reference model moves **only** on `commit`, and only if no global model was installed since the update was created (`sync_epoch`). On commit the steps/transitions folded into the snapshot are subtracted (progress after the snapshot is kept). Reject/expire leave reference and counters untouched.
Consequence: `n_i`, delta and staleness are measured against a reference that is guaranteed to be the model the server has actually seen.

**Statistics accounting.** Window statistics belong to the update that popped them. At the authoritative terminal state they are settled exactly once: *aggregated -> consumed*; *rejected/expired and same episode -> returned to the window* (their model progress is still inside the next delta); *other episode -> discarded* and counted (`stats_discarded_episode`).
Distinguished counts: created (`attempted_updates`), `transmitted_updates`, `received_updates`, aggregated (`accepted_updates`), `rejected_updates`, `expired_updates`, `unresolved_updates`, duplicates ignored. Conservation is checked by `FederatedProtocol.statistics_accounting()`.
*Changed definitions:* `failed_updates` is now rejected + expired (previously rejected + upload timeouts, summed on the client); the old `win_restore` (which restored statistics without regard to episode) is replaced by `return_window_stats(..., episode_id)`; per-window keys `U_del`, `V_coll`, `units_del`, `units_coll` no longer exist.
The environment now also resets its energy-delta baseline at episode reset (previously the first step of a new episode could see a wrong delta).

**Semantic relevance attribution.** Every AUV uploads only data it collected itself, so each collected chunk is recorded in a ledger under `(pkt_id, t_collect)` for its collecting AUV; delivery utility is added to the same entry (also across windows and partial uploads).
A batch is *mature* when fully delivered or older than `H = ln(1/eps)/mu` (remaining value <= eps = `federated.maturity_eps`); it is claimed by exactly one update. With `S_i` the mature batches claimed:
`rho_sem,i = sum_S U_delivered / sum_S (units c sigma)`, in [0,1] without clipping because decay <= 1 and delivered <= collected; an inconsistent ledger raises. `rho_svc = C/(C+L_own)` and `M_i = beta rho_sem + (1-beta) rho_svc` as before.
Immature batches are not failures; undefined components (no mature batch, no sensor events) are dropped and the other component is used (M_i = 0 if none). Batches unclaimed at episode end are censored (counted in `censored_V`, never attributed).
Limitations: the maturity horizon is a modelling choice; delivery after claim-by-age is recorded (`late_units`) but not re-attributed; `mu = 0` batches mature only by full delivery; `rho_svc` still depends on the allocator.

**Smoke configuration.** `offload_hi: 0.04 > offload_lo: 0.01` (previously 0.02 against an inherited 0.05, which made the mode switch on and off every step) and `contact_time_s: 5.0 = max_hover_steps * dt_s`, the same relation as the main config (60 = 60 x 1.0). Main-config values are unchanged. `validate_config` rejects inverted hysteresis, `contact_time_s > max_hover_steps*dt`, and `max_buffer_age_s <=` aggregation timeouts.

## Scientific review notes (code reading; no new experiments)
Implemented and unit-tested: twin critics with clipped double-Q target, target-policy smoothing with noise clipping, delayed actor updates with Polyak targets after every `policy_delay`-th critic step, centralised critics with simulator state (CTDE), actor-only federation (critics, replay and optimisers stay local), staleness `r - v_i` with `exp(-kappa Delta)` and a hard gate, capped normalised weights, L2 clipping, named seed streams with matched evaluation scenarios, t-based 95% CI over training seeds.
Not demonstrated: that semantic/service weighting improves anything; convergence; fairness of the baselines in a full run. Caveats found: (i) the centralised critic training reads a shared joint store and peers' target actors, i.e. training is not communication-free between AUVs and is outside the acoustic model; (ii) weighting changes the effective objective (not unbiased FedAvg); (iii) clipping adds no noise, so **there is no differential privacy, no secure aggregation and no formal privacy guarantee**.
**Protocol 2.1:** 10 training seeds; bounded retry/back-off after rejection (parameters are a-priori design values, see `docs/AUTHOR_DECISIONS.md` D6); `install_global` with blend = 1 is an exact replacement. Open author decisions D1-D5 are in `docs/AUTHOR_DECISIONS.md`; none is resolved. Measured: the relevance factor has little dynamic range (inert with untrained policies, ~1.5% weight TV with a competent symmetric reference policy), so a measurable contribution of semantic weighting is not expected without client heterogeneity.
**Resolved:** the former 200-vs-500 `train_episodes` discrepancy is resolved to 500 (protocol 2.0, evidence: Paper 2 Table I); the validator now fails (exit 1) on any manifest/YAML mismatch.
Other remaining limitations: the client cannot expire an update after its last bit has left (it waits for the USV outcome, which needs a downlink contact, so a client can stay blocked while out of range); downlink control messages are modelled as small transfers with the same loss model; no scalability or ablation run has been executed after these changes.
