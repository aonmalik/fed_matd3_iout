# Protocol 2.2 candidate — unresolved issues (none is resolved merely because it is documented)

## A. Author decisions (all OPEN or PROVISIONAL, nothing approved)
D1 maturity threshold · D2 undefined relevance components · D3 rejected/expired statistics policy · D4 client-side expiry after transmission · D5 Adam state after global install · D6 retry/back-off parameters — all untouched, defaults as in 2.1, see `AUTHOR_DECISIONS.md`.
D7 duplex/airtime (provisional half) · D8 drain length (provisional 600 s) · D9 compute-energy reporting (provisional `not_measured`) · D10 fixed-workload metric (provisional) — see `PROTOCOL_2.2_DECISION_LOG.md`.

## B. Provisional configuration choices
`acoustic.usv_link.duplex: half`; `federated.final_drain_s: 600.0` (= max_buffer_age_s; NOT a derived total bound; needs a sensitivity analysis, e.g. 0 / 300 / 600 / 1200 s); `experiment.fixed_workload_eval: true`; fixed-workload definition (initial buffers, delivery to USV, one-step resolution).

## C. Missing physical-layer / system assumptions
Real modem duplex capability unknown; no modem turnaround, guard time, preamble or header overhead; byte-level retransmission of corrupted fragments not modelled; USV-side modem/airtime contention not modelled (one USV serves all AUVs without a capacity limit); sensor↔AUV link not pooled with the USV link (separate modem assumed); all energy constants (`p_elec_w`, `rx_power_w`, `e_grad_j`) are uncalibrated assumptions; one-step lag between the environment's sensor-uplink reservation and the protocol's grants.

## D. Compute energy
No power instrument in the build environment: compute energy is `not_measured` for every method (compute time is measured). MATD3's modelled `e_grad_j` charge is an assumption; MAPPO charges nothing. Cross-method TRAINING total-energy comparisons are incomplete and must not support energy-efficiency claims. Hardware measurement (RAPL/GPU counters, idle subtraction, per-process attribution) is future work.

## E. Drain horizon
A deterministic server-side bound (521.1 s) exists; the client-notification and download legs have none (contact/outage). A drain can end with unresolved updates (reported). The effect of the drain on the final global model is unmeasured at scale.

## F. Method / evaluation concerns kept visible
* Relevance dynamic range: relevance has little influence under symmetric clients in the tested reference policies (TV vs uniform ≈ 1.5% in the 2.1 diagnostics); **no tuning of `eta`, `eps_m` or normalisation was done or is proposed without approval**; heterogeneous-client diagnostics are descriptive only. A null result for the semantic/service ablation is possible.
* MAPPO is not like-for-like (N2/N3/N4/N11). Critics are CTDE and not communication-constrained. `n_samples` = replay-sample presentations (steps × batch), not unique transitions; aggregation is not unbiased FedAvg. Backlog clearance is not fixed-workload completion; fixed-workload times with censoring are lower bounds on the mean.
* No differential privacy / secure aggregation. 10 seeds do not guarantee significance; attainability of Holm-corrected tests is not power.

## G. Evidence limits
Smoke-scale evidence only (e.g. a 3-episode × 3-seed census: 0/0/2 updates pending at cutoff). No comparative experiment, no pilot (`run_pilot_2seeds.py` was not run), no 2.1-vs-2.2 numerical comparison, no measured effect of C1–C3.

## H. Required before any publication claim
Approved D1–D10; hardware-based or otherwise justified energy accounting; calibration of acoustic/energy constants; a pilot then the full 500-episode × 10-seed protocol with paired statistics; drain sensitivity; ablations incl. relevance; 2.1-vs-2.2 impact quantification; independent review of the inherited changes; correctness argument for time-division airtime against the real modem.
