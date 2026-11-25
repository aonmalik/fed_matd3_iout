# Mathematical Model Audit

Generated from `scripts/equation_registry.py` (do not edit by hand). Classes: A established; B valid under stated assumptions; C proposed modelling choice requiring experimental justification; D approximation requiring validation; E incorrect/inconsistent as printed and corrected. **Nothing here is claimed to be experimentally verified**; class C/D items are unvalidated. P1 = SAC-TD3 paper, P2 = STA-TA paper (both read for this audit). Original `dt_smarl_step8` code could not be inspected (see DEVELOPMENT_STATUS.md).

- Class A  established: 22 entries
- Class B  valid under stated assumptions: 19 entries
- Class C  proposed (needs experimental justification): 19 entries
- Class D  approximation (needs validation): 6 entries
- Class E  inconsistent as printed -> corrected: 0 entries

## Discrepancies and decisions

1. **P1 Eq.(8)/(11) (path loss, SNR).** Printed with l = d/1000 km and no reference range, so TL < 0 dB for d < 1 km; P_t (W) is divided by a dB-domain noise PSD. Implemented instead: 1 m reference, SL re 1 uPa @1 m, SNR in dB (C2, C4, C5). Class E as printed.
2. **Energy.** P1/P2 use a quadratic action penalty; this project uses cubic propulsion power (E1) plus separate tx/rx/compute terms. They are not mixed; coefficients uncalibrated.
3. **P1 sigma (Eq.18)** is a category-mixture mean per node; here category is drawn per packet with the same mean (S5). c enters once (S2).
4. **P1 eta_sem = U/N_del** is reported as `U_per_packet`; `U_norm` (generated-referenced) is additional (S4).
5. **P2 reward (Eq.14)** is not reproduced; R1 is the proposed reward of the task specification.
6. **mu for the 'high' category and initial buffer law** are not in the papers; assumed and exposed in config.
7. **Own implementation bug found and fixed:** truncated Poisson tail (P4).
8. **Reward shaping** invariance is not strict under target switching (R2).
9. **Link range** is a 3-D slant distance; AUVs sit 50 m below the USV, so range < 50 m can never connect.

## Equation registry

### Q0 - Poisson arrivals
**Class:** B  valid under stated assumptions

- **Expression:** `A_j(t) ~ Poisson(lambda_j(t) dt)`
- **Meaning:** new data units generated at sensor j in one step
- **Symbols / units:** lambda [units/s], dt [s], A [units]
- **Assumptions:** lambda constant within a step; integer unit arrivals
- **Relation to Paper 1/2:** P2 Sec. III-B; P1 Eq.(2)
- **Implementation:** `src/fed_matd3_iout/environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.step`
- **Test:** `tests/test_environment.py::test_env_arrivals_match_lambda`
- **Config:** traffic.base_rates, traffic.scale, traffic.rate_change_*
- **Known limitations:** Rate change is piecewise-constant; no burstiness/correlation.

### Q1 - Pre-service occupancy
**Class:** B  valid under stated assumptions

- **Expression:** `B_pre = B + A`
- **Meaning:** occupancy after arrivals, before service
- **Symbols / units:** units
- **Assumptions:** arrival-before-service
- **Relation to Paper 1/2:** P2 Eq.(1)
- **Implementation:** `src/fed_matd3_iout/environment/sensor_queue.py:SensorQueue.arrive`
- **Test:** `tests/test_queue.py::test_arrival_before_service_order`
- **Config:** env.buffer_units
- **Known limitations:** Event order is never changed.

### Q2 - Overflow
**Class:** B  valid under stated assumptions

- **Expression:** `L = max(0, B_pre - B_max)`
- **Meaning:** tail-dropped (lost) units
- **Symbols / units:** units
- **Assumptions:** excess of NEW arrivals is dropped
- **Relation to Paper 1/2:** P2 Eq.(2)
- **Implementation:** `src/fed_matd3_iout/environment/sensor_queue.py:SensorQueue.arrive`
- **Test:** `tests/test_queue.py::test_buffer_capacity_and_overflow`
- **Config:** env.buffer_units
- **Known limitations:** Drop policy (tail of new arrivals) is a modelling choice; value of lost packets tracked for diagnostics.

### Q3 - Available backlog
**Class:** B  valid under stated assumptions

- **Expression:** `B_avail = min(B_max, B_pre)`
- **Meaning:** occupancy actually stored
- **Symbols / units:** units
- **Assumptions:** as Q1-Q2
- **Relation to Paper 1/2:** P2 Eq.(3)
- **Implementation:** `src/fed_matd3_iout/environment/sensor_queue.py:SensorQueue.arrive`
- **Test:** `tests/test_queue.py::test_buffer_capacity_and_overflow`
- **Config:** env.buffer_units
- **Known limitations:** -

### Q4 - Collected amount
**Class:** B  valid under stated assumptions

- **Expression:** `D = min{B_avail, R_ij dt / q, free_onboard/q}`
- **Meaning:** units removed from the sensor in one service step (FIFO)
- **Symbols / units:** R [Mbit/s], q = Q/B_max [Mbit/unit]
- **Assumptions:** rate-limited; extra onboard-capacity cap is an addition to P2 (default 20 Mbit assumed)
- **Relation to Paper 1/2:** P2 Eq.(4) (+onboard cap)
- **Implementation:** `src/fed_matd3_iout/environment/sensor_queue.py:SensorQueue.serve`
- **Test:** `tests/test_queue.py::test_no_negative_and_no_double_collection`
- **Config:** env.onboard_capacity_mbit, acoustic.sensor_link.max_rate_mbps
- **Known limitations:** Single sensor-link rate at hover (worst case r_dc); no per-step rate variation.

### Q5 - Buffer update
**Class:** B  valid under stated assumptions

- **Expression:** `B(t+1) = B_avail - D`
- **Meaning:** residual backlog
- **Symbols / units:** units
- **Assumptions:** -
- **Relation to Paper 1/2:** P2 Eq.(5)
- **Implementation:** `src/fed_matd3_iout/environment/sensor_queue.py:SensorQueue.serve`
- **Test:** `tests/test_queue.py::test_partial_batch_split_conserves_units`
- **Config:** -
- **Known limitations:** -

### Q6 - Unit conversion
**Class:** A  established

- **Expression:** `q = Q / B_max  [Mbit per unit]`
- **Meaning:** Mbit <-> units conversion
- **Symbols / units:** Q [Mbit], B_max [units]
- **Assumptions:** -
- **Relation to Paper 1/2:** P2 Sec. III-B
- **Implementation:** `src/fed_matd3_iout/environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.__init__`
- **Test:** `tests/test_mass_balance.py::test_env_end_to_end_mass_balance`
- **Config:** env.buffer_mbit, env.buffer_units
- **Known limitations:** -

### Q7 - Sensor mass balance
**Class:** A  established

- **Expression:** `G = D + L + B_res`
- **Meaning:** accounting identity per sensor and fleet (G includes initial contents)
- **Symbols / units:** units or Mbit
- **Assumptions:** -
- **Relation to Paper 1/2:** P2 Eq.(7)
- **Implementation:** `src/fed_matd3_iout/environment/sensor_queue.py:SensorQueue.residual_check`
- **Test:** `tests/test_mass_balance.py::test_queue_mass_balance_random_ops`
- **Config:** -
- **Known limitations:** -

### Q8 - End-to-end mass balance
**Class:** A  established

- **Expression:** `G = Delivered + Onboard + L + B_res`
- **Meaning:** extends Q7 to AUV-carried and USV-delivered data
- **Symbols / units:** Mbit
- **Assumptions:** -
- **Relation to Paper 1/2:** extension
- **Implementation:** `src/fed_matd3_iout/environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.accounting`
- **Test:** `tests/test_mass_balance.py::test_env_end_to_end_mass_balance`
- **Config:** -
- **Known limitations:** Verified to <1e-9 Mbit; dropped data after delivery not modelled.

### Q9 - Hover duration
**Class:** B  valid under stated assumptions

- **Expression:** `h = max(1, ceil(Q b / (B R dt)))  [steps]`
- **Meaning:** hover steps computed once at contact
- **Symbols / units:** b units, R Mbit/s
- **Assumptions:** computed once; capped
- **Relation to Paper 1/2:** P2 Eq.(6)
- **Implementation:** `src/fed_matd3_iout/environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.step`
- **Test:** `tests/test_environment.py::test_hover_duration_rule`
- **Config:** env.max_hover_steps
- **Known limitations:** max_hover_steps cap is an ADDITION (default 60).

### P0 - Travel time
**Class:** D  approximation (needs validation)

- **Expression:** `T_ij = d_ij / (v_nom + eps)`
- **Meaning:** planned travel time
- **Symbols / units:** d [m], v_nom [m/s]
- **Assumptions:** straight-line path at constant nominal speed (P2)
- **Relation to Paper 1/2:** P2 Eq.(15)
- **Implementation:** `src/fed_matd3_iout/allocation/service_arrival_predictor.py:travel_time_s`
- **Test:** `tests/test_service_arrival.py::test_zero_arrival_and_zero_travel`
- **Config:** allocation.v_nom
- **Known limitations:** Ignores heading changes, boundary rejections, drift and tracking error.

### P1 - Exact expected backlog at arrival
**Class:** B  valid under stated assumptions

- **Expression:** `E[B_arr] = sum_n min(B_max, b+n) Pois(n; lambda T)`
- **Meaning:** expectation of capped backlog, no intermediate service
- **Symbols / units:** b, B_max [units], lambda T [units]
- **Assumptions:** Poisson arrivals, constant lambda, known T, no service in between
- **Relation to Paper 1/2:** extends P2 Eq.(16)
- **Implementation:** `src/fed_matd3_iout/allocation/service_arrival_predictor.py:expected_backlog_exact`
- **Test:** `tests/test_service_arrival.py::test_exact_matches_brute_force`
- **Config:** allocation.backlog_model=exact, allocation.poisson_tol
- **Known limitations:** Exact only given T; travel-time uncertainty not modelled. Tail truncation documented (P1-trunc).

### P2 - Expected overflow
**Class:** B  valid under stated assumptions

- **Expression:** `E[L_arr] = sum_n [b+n-B_max]^+ Pois(n; lambda T) = b + lambda T - E[B_arr]`
- **Meaning:** expected loss before arrival
- **Symbols / units:** units
- **Assumptions:** as P1
- **Relation to Paper 1/2:** new
- **Implementation:** `src/fed_matd3_iout/allocation/service_arrival_predictor.py:expected_backlog_exact`
- **Test:** `tests/test_service_arrival.py::test_overflow_conservation_identity`
- **Config:** allocation.poisson_tol
- **Known limitations:** -

### P3 - Certainty-equivalent backlog
**Class:** D  approximation (needs validation)

- **Expression:** `B_hat = min(B_max, b + lambda T)`
- **Meaning:** fast approximation; UPPER BOUND on E[B_arr] (Jensen), NOT the exact expectation
- **Symbols / units:** units
- **Assumptions:** replaces random arrivals by their mean
- **Relation to Paper 1/2:** P2 Eqs.(16)-(17)
- **Implementation:** `src/fed_matd3_iout/allocation/service_arrival_predictor.py:certainty_equivalent_backlog`
- **Test:** `tests/test_service_arrival.py::test_certainty_equivalent_is_upper_bound_not_exact`
- **Config:** allocation.backlog_model=ce
- **Known limitations:** Gap largest when b+lambda T ~ B_max.

### P4 - Truncated Poisson sum
**Class:** B  valid under stated assumptions

- **Expression:** `n_hi = min(floor(B_max-b), Q_{1-tol}); tail by sum_{n>N} n p_n = m(1-F(N-1))`
- **Meaning:** controlled truncation of P1
- **Symbols / units:** tol [-]
- **Assumptions:** tol small
- **Relation to Paper 1/2:** new
- **Implementation:** `src/fed_matd3_iout/allocation/service_arrival_predictor.py:expected_backlog_exact`
- **Test:** `tests/test_service_arrival.py::test_truncation_accuracy_bound`
- **Config:** allocation.poisson_tol (1e-9)
- **Known limitations:** BUG FOUND+FIXED: first version valued the tail at B_max (error ~tol*B_max).

### SC1 - Service capacity (constant rate)
**Class:** B  valid under stated assumptions

- **Expression:** `C = R T_contact`
- **Meaning:** Mbit transferable in a contact
- **Symbols / units:** R [Mbit/s], T [s]
- **Assumptions:** constant rate
- **Relation to Paper 1/2:** task spec
- **Implementation:** `src/fed_matd3_iout/allocation/service_capacity.py:capacity_constant`
- **Test:** `tests/test_service_arrival.py::test_service_capacity_constant_and_integral`
- **Config:** allocation.contact_time_s
- **Known limitations:** -

### SC2 - Service capacity (time-varying)
**Class:** D  approximation (needs validation)

- **Expression:** `C = integral R(tau) dtau`
- **Meaning:** trapezoid quadrature
- **Symbols / units:** Mbit
- **Assumptions:** R integrable
- **Relation to Paper 1/2:** task spec
- **Implementation:** `src/fed_matd3_iout/allocation/service_capacity.py:capacity_integral`
- **Test:** `tests/test_service_arrival.py::test_service_capacity_constant_and_integral`
- **Config:** -
- **Known limitations:** Utility function only; the environment uses constant hover rate (SC1).

### SC3 - Collectable volume
**Class:** B  valid under stated assumptions

- **Expression:** `D_service = min{B_arr, C/q, free/q}`
- **Meaning:** predicted collectable units
- **Symbols / units:** units
- **Assumptions:** -
- **Relation to Paper 1/2:** task spec
- **Implementation:** `src/fed_matd3_iout/allocation/service_capacity.py:collectable_units`
- **Test:** `tests/test_service_arrival.py::test_service_capacity_constant_and_integral`
- **Config:** -
- **Known limitations:** Prediction only; collection removes units once (Q4).

### S1 - Value of information
**Class:** C  proposed (needs experimental justification)

- **Expression:** `VoI(tau) = sigma exp(-mu tau)`
- **Meaning:** exponentially decaying value
- **Symbols / units:** sigma [0,1], mu [1/s], tau [s]
- **Assumptions:** exponential decay
- **Relation to Paper 1/2:** P1 Eq.(17)
- **Implementation:** `src/fed_matd3_iout/semantic/value_of_information.py:voi`
- **Test:** `tests/test_semantic_utility.py::test_voi_decay`
- **Config:** semantic.categories.*.mu
- **Known limitations:** Decay law is a modelling choice. mu(high) NOT given in P1 (assumed = routine).

### S2 - Packet semantic utility
**Class:** C  proposed (needs experimental justification)

- **Expression:** `U_k = d_k c_k sigma_k exp(-mu_k tau_k^e2e)`
- **Meaning:** utility of a delivered packet; criticality c enters ONCE
- **Symbols / units:** d in {0,1}, c >= 1
- **Assumptions:** tau = delivery time - generation time (incl. queueing, propagation)
- **Relation to Paper 1/2:** P1 Eq.(19)
- **Implementation:** `src/fed_matd3_iout/semantic/semantic_utility.py:delivered_utility`
- **Test:** `tests/test_semantic_utility.py::test_mission_criticality_applied_once`
- **Config:** semantic.categories.*.c_priority
- **Known limitations:** tau omits processing and USV->shore RF delay (P1 Eq.14).

### S3 - Total utility
**Class:** C  proposed (needs experimental justification)

- **Expression:** `U = sum_k U_k`
- **Meaning:** total semantic utility; each delivered chunk recorded once
- **Symbols / units:** -
- **Assumptions:** -
- **Relation to Paper 1/2:** P1 Eq.(19)
- **Implementation:** `src/fed_matd3_iout/semantic/semantic_utility.py:UtilityLedger.record`
- **Test:** `tests/test_semantic_utility.py::test_no_duplicate_utility`
- **Config:** -
- **Known limitations:** -

### S4 - Normalised utilities
**Class:** C  proposed (needs experimental justification)

- **Expression:** `U_norm = U / sum_generated c sigma ;  U_per_packet = U / N_del`
- **Meaning:** U_per_packet is P1's eta_sem; U_norm is generated-referenced
- **Symbols / units:** -
- **Assumptions:** -
- **Relation to Paper 1/2:** P1 Eq.(20) (U_per_packet)
- **Implementation:** `src/fed_matd3_iout/evaluation/metrics.py:episode_metrics`
- **Test:** `tests/test_integration.py::test_metric_definitions_and_denominators`
- **Config:** -
- **Known limitations:** U_norm can exceed 1 (c>1).

### S5 - Packet importance
**Class:** C  proposed (needs experimental justification)

- **Expression:** `sigma = clip(omega_sensor * sigma_factor(category), 0, 1)`
- **Meaning:** normalised importance; NO criticality factor
- **Symbols / units:** omega in {0.3,0.6,1.0}
- **Assumptions:** static omega per sensor; per-packet category
- **Relation to Paper 1/2:** P1 Eq.(18) (E[sigma] identical: omega(p_c+p_h+(1-p_c-p_h)gamma_low))
- **Implementation:** `src/fed_matd3_iout/semantic/mission_criticality.py:sigma_of`
- **Test:** `tests/test_semantic_utility.py::test_importance_normalised`
- **Config:** semantic.categories, semantic.hotspot_weights
- **Known limitations:** SIMPLIFICATION: P1 evolves omega_m(t) by a Markov chain and uses clusters; not reproduced.

### S6 - Aggregate semantic density
**Class:** D  approximation (needs validation)

- **Expression:** `u_j = E[c sigma]_j / max_k E[c sigma]_k`
- **Meaning:** AGGREGATE APPROXIMATION used only in allocation cost (A1) - never in the objective
- **Symbols / units:** [0,1]
- **Assumptions:** category mix known
- **Relation to Paper 1/2:** new
- **Implementation:** `src/fed_matd3_iout/semantic/mission_criticality.py:mean_semantic_value_density`
- **Test:** `tests/test_semantic_utility.py::test_density_aggregate_approximation`
- **Config:** -
- **Known limitations:** -

### A1 - Semantic-service assignment cost
**Class:** C  proposed (needs experimental justification)

- **Expression:** `C_ij = -w_b B~ - w_s U~ + w_d T~ + w_o L~ + w_c S~ + w_l W~ - w_a A~`
- **Meaning:** normalised (all in [0,1]) assignment cost
- **Symbols / units:** see semantic_service_cost.py docstring for each term
- **Assumptions:** terms commensurate after normalisation
- **Relation to Paper 1/2:** extends P2 Eq.(20) (b,d,sw terms identical, weights 1,1,0.05)
- **Implementation:** `src/fed_matd3_iout/allocation/semantic_service_cost.py:cost_matrix`
- **Test:** `tests/test_task_allocator.py::test_cost_terms_normalised`
- **Config:** allocation.weights.*
- **Known limitations:** PROPOSED. semantic/overflow/workload/age terms are new; each ablated.

### A2 - One-to-one assignment
**Class:** A  established

- **Expression:** `min sum C_ij x_ij s.t. sum_j x_ij = 1, sum_i x_ij <= 1`
- **Meaning:** linear assignment (Hungarian / shortest augmenting path)
- **Symbols / units:** x binary
- **Assumptions:** rectangular allowed; surplus AUVs idle
- **Relation to Paper 1/2:** P2 Eqs.(8)-(9),(11)
- **Implementation:** `src/fed_matd3_iout/allocation/task_allocator.py:TaskAllocator.allocate`
- **Test:** `tests/test_task_allocator.py::test_one_to_one_and_optimal`
- **Config:** allocation.method
- **Known limitations:** -

### A3 - Priority score
**Class:** A  established

- **Expression:** `rho_ij = lambda_j b_j / B_j - alpha d_ij`
- **Meaning:** Xu/P2 greedy & Hungarian-priority baselines
- **Symbols / units:** alpha [1/m]
- **Assumptions:** -
- **Relation to Paper 1/2:** P2 Sec. V-B
- **Implementation:** `src/fed_matd3_iout/allocation/task_allocator.py:priority_matrix`
- **Test:** `tests/test_task_allocator.py::test_greedy_reassigns_only_freed`
- **Config:** allocation.priority_alpha
- **Known limitations:** -

### E1 - Propulsion power
**Class:** C  proposed (needs experimental justification)

- **Expression:** `P_prop = P0 + kappa v^3`
- **Meaning:** hotel/baseline + cubic drag
- **Symbols / units:** W, kappa [W/(m/s)^3]
- **Assumptions:** simplified, UNCALIBRATED
- **Relation to Paper 1/2:** replaces quadratic action penalty of P1 Eq.(15)/P2
- **Implementation:** `src/fed_matd3_iout/environment/energy.py:propulsion_power`
- **Test:** `tests/test_environment.py::test_energy_model`
- **Config:** energy.p0_w, energy.kappa
- **Known limitations:** Coefficients are placeholders; calibrate before publication.

### E2 - Propulsion energy
**Class:** A  established

- **Expression:** `E_prop = P_prop dt`
- **Meaning:** energy per step
- **Symbols / units:** J
- **Assumptions:** -
- **Relation to Paper 1/2:** -
- **Implementation:** `src/fed_matd3_iout/environment/energy.py:propulsion_energy`
- **Test:** `tests/test_environment.py::test_energy_model`
- **Config:** env.dt_s
- **Known limitations:** -

### E3 - Communication energy
**Class:** B  valid under stated assumptions

- **Expression:** `E_tx = P_tx t_air ; E_rx = P_rx t_rx`
- **Meaning:** tx for data/model uploads, rx for sensor reception and downloads
- **Symbols / units:** J
- **Assumptions:** electrical power constant
- **Relation to Paper 1/2:** P1 Eq.(15)
- **Implementation:** `src/fed_matd3_iout/environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.add_comm_energy`
- **Test:** `tests/test_environment.py::test_energy_components_disjoint`
- **Config:** energy.p_tx_elec_w, acoustic.rx_power_w
- **Known limitations:** Disjoint from E1/E4; processing power of P1 not modelled separately.

### E4 - Computation energy
**Class:** C  proposed (needs experimental justification)

- **Expression:** `E_comp = e_grad * (#gradient steps)`
- **Meaning:** local training energy
- **Symbols / units:** J
- **Assumptions:** constant energy per step (assumed)
- **Relation to Paper 1/2:** P1 Eq.(15) P_proc
- **Implementation:** `src/fed_matd3_iout/training/local_training.py:local_update`
- **Test:** `tests/test_environment.py::test_compute_energy_accounting`
- **Config:** energy.e_grad_j
- **Known limitations:** Zero in evaluation episodes (no learning); inference energy not modelled.

### C1 - Thorp absorption
**Class:** A  established

- **Expression:** `a(f)=0.11f^2/(1+f^2)+44f^2/(4100+f^2)+2.75e-4 f^2+0.003  [dB/km, f kHz]`
- **Meaning:** frequency-dependent absorption
- **Symbols / units:** f [kHz]
- **Assumptions:** valid ~0.4-50 kHz
- **Relation to Paper 1/2:** P1 Eq.(9)
- **Implementation:** `src/fed_matd3_iout/communication/acoustic_channel.py:thorp_absorption_db_per_km`
- **Test:** `tests/test_acoustic_channel.py::test_thorp_known_value_and_monotone`
- **Config:** acoustic.*.freq_khz
- **Known limitations:** -

### C2 - Transmission loss
**Class:** B  valid under stated assumptions

- **Expression:** `TL = k 10log10(max(d,1 m)/1 m) + a(f) d/1000  [dB]`
- **Meaning:** spreading + absorption, 1 m reference
- **Symbols / units:** d [m], k=1.5
- **Assumptions:** far-field, isovelocity, 1 m reference range
- **Relation to Paper 1/2:** P1 Eq.(8) used l=d/1000 km with l^k and no reference range -> class E as printed
- **Implementation:** `src/fed_matd3_iout/communication/acoustic_channel.py:transmission_loss_db`
- **Test:** `tests/test_acoustic_channel.py::test_spreading_exponent_exact`
- **Config:** acoustic.spreading_k
- **Known limitations:** DISCREPANCY: P1 Eq.(8) as printed gives TL<0 dB (gain) for d<1 km; this code uses the 1 m reference consistent with SL re 1 m. Clarify in the paper.

### C3 - Ambient noise
**Class:** B  valid under stated assumptions

- **Expression:** `N(f) = 10log10(sum of turbulence+shipping+wind+thermal in linear power)  [dB re uPa^2/Hz]`
- **Meaning:** Stojanovic/Coates empirical PSDs
- **Symbols / units:** f [kHz], s, w [m/s]
- **Assumptions:** components added in linear domain
- **Relation to Paper 1/2:** P1 Eq.(10)
- **Implementation:** `src/fed_matd3_iout/communication/acoustic_channel.py:noise_psd_db`
- **Test:** `tests/test_acoustic_channel.py::test_source_level_and_noise_model`
- **Config:** acoustic.shipping_s, acoustic.wind_mps
- **Known limitations:** Empirical, valid for stated f range only.

### C4 - Source level
**Class:** A  established

- **Expression:** `SL = 170.8 + 10log10(eta P_elec) + DI  [dB re uPa @1m]`
- **Meaning:** sonar-equation source level
- **Symbols / units:** P [W]
- **Assumptions:** -
- **Relation to Paper 1/2:** replaces P1 Eq.(11) P_t/(PL N df) (unit mix)
- **Implementation:** `src/fed_matd3_iout/communication/acoustic_channel.py:source_level_db`
- **Test:** `tests/test_acoustic_channel.py::test_source_level_and_noise_model`
- **Config:** acoustic.*.p_elec_w, efficiency
- **Known limitations:** Powers/efficiencies are ASSUMPTIONS.

### C5 - Received SNR
**Class:** B  valid under stated assumptions

- **Expression:** `SNR_dB = SL - TL - (N(f)_dB + 10log10 B_Hz)`
- **Meaning:** mean SNR; dB domain throughout
- **Symbols / units:** dB, Hz
- **Assumptions:** narrowband
- **Relation to Paper 1/2:** P1 Eq.(11) (corrected units)
- **Implementation:** `src/fed_matd3_iout/communication/acoustic_channel.py:AcousticLink.snr_db`
- **Test:** `tests/test_acoustic_channel.py::test_snr_unit_consistency`
- **Config:** acoustic.*.bandwidth_hz
- **Known limitations:** -

### C6 - Achievable rate
**Class:** D  approximation (needs validation)

- **Expression:** `R = min(R_max, B log2(1+SNR_lin)) / 1e6  [Mbit/s]`
- **Meaning:** Shannon on mean SNR with modem cap
- **Symbols / units:** B [Hz]
- **Assumptions:** ignores fading outage and coding gap
- **Relation to Paper 1/2:** P2 uses the rate model of Xu et al. (details not reproduced)
- **Implementation:** `src/fed_matd3_iout/communication/acoustic_channel.py:shannon_rate_mbps`
- **Test:** `tests/test_acoustic_channel.py::test_rate_capped`
- **Config:** acoustic.*.max_rate_mbps
- **Known limitations:** Rate caps are ASSUMPTIONS and strongly shape results.

### C7 - Propagation delay
**Class:** B  valid under stated assumptions

- **Expression:** `tau = d / v_s`
- **Meaning:** one-way delay
- **Symbols / units:** d [m], v_s=1500 m/s
- **Assumptions:** constant sound speed
- **Relation to Paper 1/2:** P1 Eq.(13)
- **Implementation:** `src/fed_matd3_iout/communication/propagation_delay.py:propagation_delay_s`
- **Test:** `tests/test_acoustic_channel.py::test_propagation_delay`
- **Config:** acoustic.sound_speed_mps
- **Known limitations:** No refraction/multipath.

### C8 - Rayleigh packet success
**Class:** B  valid under stated assumptions

- **Expression:** `P = exp(-gamma_th / SNR_mean)`
- **Meaning:** P(instantaneous SNR >= gamma_th), SNR ~ Exp(mean)
- **Symbols / units:** -
- **Assumptions:** Rayleigh amplitude => exponential power; capture threshold
- **Relation to Paper 1/2:** P1 Eq.(12) (consistent: beta = gamma_th)
- **Implementation:** `src/fed_matd3_iout/communication/acoustic_channel.py:rayleigh_success_prob`
- **Test:** `tests/test_acoustic_channel.py::test_packet_success_bounds_and_derivation`
- **Config:** acoustic.*.snr_threshold_db
- **Known limitations:** Verified by Monte-Carlo. Per-step draw; no temporal fading correlation.

### C9 - Contact/outage process
**Class:** C  proposed (needs experimental justification)

- **Expression:** `range gate d_slant <= R_usv; Gilbert-Elliott outage P(up->down)=p_d, P(down->up)=p_r`
- **Meaning:** intermittent AUV-USV windows
- **Symbols / units:** p [1/step]
- **Assumptions:** two-state Markov outage independent per AUV
- **Relation to Paper 1/2:** new
- **Implementation:** `src/fed_matd3_iout/communication/packet_delivery.py:LinkProcess.step`
- **Test:** `tests/test_acoustic_channel.py::test_gilbert_elliott_stationary_fraction`
- **Config:** acoustic.usv_link.*
- **Known limitations:** Outage law is an assumption (no data).

### M1 - Actor
**Class:** A  established

- **Expression:** `a_i = mu_i(o_i), tanh output`
- **Meaning:** deterministic bounded policy
- **Symbols / units:** -
- **Assumptions:** -
- **Relation to Paper 1/2:** P2 Sec. IV-F
- **Implementation:** `src/fed_matd3_iout/agents/actor.py:Actor.forward`
- **Test:** `tests/test_matd3.py::test_actor_output_bounds`
- **Config:** matd3.hidden
- **Known limitations:** -

### M2 - Smoothed joint target action
**Class:** A  established

- **Expression:** `a'_k = clip(mu'_k(o'_k) + clip(eps,-c,c), -1, 1) for ALL k`
- **Meaning:** TD3 target-policy smoothing with all target actors
- **Symbols / units:** eps~N(0,sigma^2)
- **Assumptions:** -
- **Relation to Paper 1/2:** TD3/MATD3
- **Implementation:** `src/fed_matd3_iout/agents/matd3_agent.py:MATD3Agent.target_joint_action`
- **Test:** `tests/test_matd3.py::test_joint_action_uses_all_target_actors`
- **Config:** matd3.target_noise, matd3.noise_clip
- **Known limitations:** -

### M3 - Clipped double-Q target
**Class:** A  established

- **Expression:** `y_i = r_i + gamma (1-d) min_k Q'_{i,k}(s', a')`
- **Meaning:** bootstrapped target; d=0 at time-limit truncation
- **Symbols / units:** -
- **Assumptions:** -
- **Relation to Paper 1/2:** TD3/MATD3
- **Implementation:** `src/fed_matd3_iout/agents/matd3_agent.py:MATD3Agent.compute_target`
- **Test:** `tests/test_matd3.py::test_twin_target_minimum_and_terminal`
- **Config:** matd3.gamma
- **Known limitations:** -

### M4 - Critic loss
**Class:** A  established

- **Expression:** `L = E[(Q_{i,k}(s,a_1..a_M) - y_i)^2]`
- **Meaning:** centralised twin critics (state + joint action)
- **Symbols / units:** -
- **Assumptions:** simulator state available at TRAINING only (CTDE)
- **Relation to Paper 1/2:** MATD3
- **Implementation:** `src/fed_matd3_iout/agents/matd3_agent.py:MATD3Agent.update`
- **Test:** `tests/test_target_updates.py::test_critic_loss_decreases_on_fixed_batch`
- **Config:** matd3.critic_lr, matd3.critic_mode
- **Known limitations:** Training is NOT fully decentralised.

### M5 - Deterministic policy gradient
**Class:** A  established

- **Expression:** `grad J = E[grad_a Q_{i,1}(s, a_{-i}, mu_i(o_i)) grad mu_i]`
- **Meaning:** other agents' actions from replay
- **Symbols / units:** -
- **Assumptions:** -
- **Relation to Paper 1/2:** DPG
- **Implementation:** `src/fed_matd3_iout/agents/matd3_agent.py:MATD3Agent.update`
- **Test:** `tests/test_target_updates.py::test_actor_gradient_flows_only_through_own_action`
- **Config:** matd3.actor_lr
- **Known limitations:** -

### M6 - Delayed update + Polyak
**Class:** A  established

- **Expression:** `every d-th critic step: actor step; theta' <- tau theta + (1-tau) theta'`
- **Meaning:** delayed actor and target updates
- **Symbols / units:** tau, d
- **Assumptions:** -
- **Relation to Paper 1/2:** TD3
- **Implementation:** `src/fed_matd3_iout/agents/matd3_agent.py:polyak_update`
- **Test:** `tests/test_target_updates.py::test_delayed_actor_update_and_target_ordering`
- **Config:** matd3.tau, matd3.policy_delay
- **Known limitations:** -

### F1 - Local update
**Class:** A  established

- **Expression:** `Delta_i = theta_i^local - theta_i^base`
- **Meaning:** actor parameter difference vs last synchronised global model
- **Symbols / units:** flat params
- **Assumptions:** one outstanding update per AUV
- **Relation to Paper 1/2:** new
- **Implementation:** `src/fed_matd3_iout/federated/federated_client.py:FederatedClient.make_update`
- **Test:** `tests/test_federated_aggregation.py::test_client_update_construction_and_versions`
- **Config:** federated.local_steps_per_update
- **Known limitations:** Delta is snapshotted when transmission STARTS.

### F2 - Staleness
**Class:** A  established

- **Expression:** `Delta_i(stale) = r - v_i`
- **Meaning:** server version minus base version
- **Symbols / units:** rounds
- **Assumptions:** -
- **Relation to Paper 1/2:** new
- **Implementation:** `src/fed_matd3_iout/federated/staleness.py:staleness`
- **Test:** `tests/test_staleness.py::test_staleness_definition_and_factor`
- **Config:** -
- **Known limitations:** -

### F3 - Staleness factor
**Class:** C  proposed (needs experimental justification)

- **Expression:** `w_stale = exp(-kappa Delta)`
- **Meaning:** down-weighting of stale updates
- **Symbols / units:** kappa
- **Assumptions:** -
- **Relation to Paper 1/2:** new
- **Implementation:** `src/fed_matd3_iout/federated/staleness.py:staleness_factor`
- **Test:** `tests/test_staleness.py::test_staleness_definition_and_factor`
- **Config:** federated.kappa
- **Known limitations:** Functional form unvalidated.

### F3b - Max-staleness gate
**Class:** C  proposed (needs experimental justification)

- **Expression:** `accept iff 0 <= Delta <= Delta_max`
- **Meaning:** reject over-stale updates
- **Symbols / units:** rounds
- **Assumptions:** -
- **Relation to Paper 1/2:** new
- **Implementation:** `src/fed_matd3_iout/federated/staleness.py:is_acceptable`
- **Test:** `tests/test_staleness.py::test_max_staleness_gate`
- **Config:** federated.max_staleness
- **Known limitations:** -

### F4 - Semantic-service relevance
**Class:** C  proposed (needs experimental justification)

- **Expression:** `M = beta rho_sem + (1-beta) rho_svc;  rho_sem = U_mature/V_mature over the MATURE batches S_i of AUV i;  rho_svc = C/(C+L_own)`
- **Meaning:** relevance in [0,1] from MEASURED, batch-attributed statistics
- **Symbols / units:** see relevance.py / semantic/attribution.py
- **Assumptions:** mature = fully delivered or older than ln(1/eps)/mu
- **Relation to Paper 1/2:** new
- **Implementation:** `src/fed_matd3_iout/federated/relevance.py:relevance`
- **Test:** `tests/test_relevance_attribution.py::test_relevance_bounded_and_meaningful`
- **Config:** federated.beta_sem, federated.maturity_eps
- **Known limitations:** CHANGED (audit): numerator and denominator now describe the SAME batches, so no clipping is needed (bounded by construction; inconsistency raises). Undefined components (no mature batch / no sensor events) are dropped, not scored 0. Maturity horizon is a modelling choice. Own-sensor overflow depends on allocation, not only on the AUV's policy.

### F13 - Batch-level attribution ledger
**Class:** C  proposed (needs experimental justification)

- **Expression:** `key (pkt_id, t_collect) per AUV; V += units c sigma at collection; U += delivered utility at delivery; claim once when mature; censor at episode end`
- **Meaning:** delivered utility is credited to the batch and the AUV that collected it, across windows
- **Symbols / units:** H = ln(1/eps)/mu
- **Assumptions:** AUVs upload only their own data (no hand-off)
- **Relation to Paper 1/2:** new
- **Implementation:** `src/fed_matd3_iout/semantic/attribution.py:BatchBook`
- **Test:** `tests/test_relevance_attribution.py::test_batch_collected_in_one_window_delivered_in_another`
- **Config:** federated.maturity_eps
- **Known limitations:** Late delivery after claim-by-age is counted in late_units, not re-attributed (<= eps of the batch value). Batches of mu=0 mature only by full delivery.

### F14 - Update lifecycle
**Class:** A  established

- **Expression:** `created -> uploading -> in_flight -> buffered -> aggregated | rejected | expired; one outstanding update per client; only a 'commit' control message advances the reference`
- **Meaning:** transport receipt != aggregation commit
- **Symbols / units:** -
- **Assumptions:** server is authoritative from in_flight on; terminal states have no outgoing transition
- **Relation to Paper 1/2:** audit repair
- **Implementation:** `src/fed_matd3_iout/federated/update_packet.py:UpdatePacket.transition`
- **Test:** `tests/test_update_lifecycle.py::test_successful_aggregation_advances_reference_exactly_once`
- **Config:** federated.max_buffer_age_s, federated.upload_timeout_s
- **Known limitations:** Client cannot withdraw an update once its last bit has left (no client-side expiry after in_flight).

### F15 - Window-statistics settlement
**Class:** A  established

- **Expression:** `AGGREGATED -> consumed; REJECTED/EXPIRED -> returned to the window if same episode else discarded; settled exactly once per update`
- **Meaning:** no loss and no double counting of update-owned statistics
- **Symbols / units:** -
- **Assumptions:** episode_id stamped on every update
- **Relation to Paper 1/2:** audit repair
- **Implementation:** `src/fed_matd3_iout/federated/communication_protocol.py:FederatedProtocol._settle`
- **Test:** `tests/test_stats_accounting.py::test_old_episode_stats_cannot_contaminate_next_episode`
- **Config:** -
- **Known limitations:** Statistics returned after a rejection are re-reported by the NEXT update (their model progress is also still in its delta).

### F5 - Aggregation weight
**Class:** C  proposed (needs experimental justification)

- **Expression:** `Z_i = n_i^alpha exp(-kappa Delta_i) [eps_M + (1-eps_M) M_i]^eta`
- **Meaning:** NOT unbiased FedAvg; link reliability NOT multiplied in
- **Symbols / units:** n_i = grad steps x batch size
- **Assumptions:** weights change the effective objective
- **Relation to Paper 1/2:** new
- **Implementation:** `src/fed_matd3_iout/federated/aggregation.py:raw_weights`
- **Test:** `tests/test_federated_aggregation.py::test_weight_formula_and_normalisation`
- **Config:** federated.alpha, eta, eps_m
- **Known limitations:** No convergence guarantee; no privacy guarantee.

### F6 - Normalisation and cap
**Class:** C  proposed (needs experimental justification)

- **Expression:** `w_i = Z_i / sum_R Z_j; optional w_i <= w_max with iterative renormalisation`
- **Meaning:** convex weights over RECEIVED+ACCEPTED updates only
- **Symbols / units:** -
- **Assumptions:** -
- **Relation to Paper 1/2:** new
- **Implementation:** `src/fed_matd3_iout/federated/aggregation.py:normalise_with_cap`
- **Test:** `tests/test_federated_aggregation.py::test_weight_cap_and_renormalisation`
- **Config:** federated.max_client_weight
- **Known limitations:** Cap below 1/|R| infeasible -> uniform.

### F7 - L2 clipping
**Class:** A  established

- **Expression:** `Clip_C(d) = d min(1, C/||d||_2)`
- **Meaning:** global L2-norm clipping
- **Symbols / units:** C
- **Assumptions:** -
- **Relation to Paper 1/2:** standard
- **Implementation:** `src/fed_matd3_iout/federated/aggregation.py:l2_clip`
- **Test:** `tests/test_federated_aggregation.py::test_l2_clip`
- **Config:** federated.clip_norm
- **Known limitations:** No noise added -> no differential privacy.

### F8 - Server aggregation
**Class:** B  valid under stated assumptions

- **Expression:** `theta^{r+1} = theta^r + eta_g sum_R w_i Clip(Delta_i)`
- **Meaning:** server step
- **Symbols / units:** eta_g
- **Assumptions:** equals FedAvg only for w_i=n_i/sum n, eta_g=1, no clip
- **Relation to Paper 1/2:** new
- **Implementation:** `src/fed_matd3_iout/federated/aggregation.py:aggregate`
- **Test:** `tests/test_federated_aggregation.py::test_aggregate_formula_and_empty_and_mismatch`
- **Config:** federated.server_lr
- **Known limitations:** -

### F9 - Global install
**Class:** C  proposed (needs experimental justification)

- **Expression:** `theta_local <- b theta_g + (1-b) theta_local`
- **Meaning:** download blend (b=1: replace)
- **Symbols / units:** b
- **Assumptions:** critics stay local
- **Relation to Paper 1/2:** new
- **Implementation:** `src/fed_matd3_iout/federated/federated_client.py:FederatedClient.install_global`
- **Test:** `tests/test_federated_aggregation.py::test_client_update_construction_and_versions`
- **Config:** federated.download_blend
- **Known limitations:** -

### F10 - Transfer feasibility
**Class:** C  proposed (needs experimental justification)

- **Expression:** `progress += R*share*dt only when contact up and fading draw succeeds; received at t_tx_done + tau_prop`
- **Meaning:** model upload/download over intermittent link
- **Symbols / units:** bytes
- **Assumptions:** fragmentation optional
- **Relation to Paper 1/2:** new
- **Implementation:** `src/fed_matd3_iout/communication/model_transfer.py:ModelTransfer.step`
- **Test:** `tests/test_contact_scheduler.py::test_transfer_needs_multiple_steps_and_propagation`
- **Config:** federated.allow_fragmentation, acoustic.usv_link.model_share
- **Known limitations:** -

### F11 - Contact scheduler
**Class:** C  proposed (needs experimental justification)

- **Expression:** `start iff link up (and, without fragmentation, airtime <= expected remaining window)`
- **Meaning:** defer/start decision
- **Symbols / units:** -
- **Assumptions:** EWMA window estimate
- **Relation to Paper 1/2:** new
- **Implementation:** `src/fed_matd3_iout/communication/contact_scheduler.py:ContactScheduler.may_start`
- **Test:** `tests/test_contact_scheduler.py::test_defers_when_update_does_not_fit_window`
- **Config:** federated.allow_fragmentation
- **Known limitations:** -

### F12 - No aggregation before receipt
**Class:** A  established

- **Expression:** `buffer only filled by USVServer.receive() after last bit arrives; duplicate update ids are ignored`
- **Meaning:** causality + exactly-once
- **Symbols / units:** -
- **Assumptions:** -
- **Relation to Paper 1/2:** spec rule 9
- **Implementation:** `src/fed_matd3_iout/federated/usv_server.py:USVServer.receive`
- **Test:** `tests/test_federated_aggregation.py::test_no_aggregation_before_receipt`
- **Config:** -
- **Known limitations:** -

### R1 - Team + local reward
**Class:** C  proposed (needs experimental justification)

- **Expression:** `r_i = (1-zeta) r_team + zeta r_local + w_sh F;  r_team = w_u dU/U_ref - w_l L/L_ref - w_e E/E_ref - w_c C_coll - w_o C_bound`
- **Meaning:** normalised; utility already includes delivered value so NO separate delivery term
- **Symbols / units:** -
- **Assumptions:** -
- **Relation to Paper 1/2:** replaces P2 Eq.(14) (different weights/terms)
- **Implementation:** `src/fed_matd3_iout/environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.step`
- **Test:** `tests/test_environment.py::test_reward_collision_penalty`
- **Config:** reward.*, federated.local_reward_weight
- **Known limitations:** Magnitudes measured: random policy -391/episode, goal-seeking reference -21. Only the collision term has a dedicated unit test.

### R2 - Potential-based shaping
**Class:** D  approximation (needs validation)

- **Expression:** `F = (|p-g| - gamma|p'-g|)/D_n`
- **Meaning:** goal-approach shaping; policy-invariant for a FIXED goal (Ng et al. 1999)
- **Symbols / units:** -
- **Assumptions:** fixed goal per step
- **Relation to Paper 1/2:** P2 uses dense -0.6|p-s|
- **Implementation:** `src/fed_matd3_iout/environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.step`
- **Test:** `NONE (documented gap)`
- **Config:** reward.w_shape
- **Known limitations:** Invariance does not strictly hold when the target changes mid-episode (non-stationary potential). TEST GAP.

### MET1 - Ratios
**Class:** A  established

- **Expression:** `collection = collected/G; delivery (PDR) = delivered/G; delivered-of-collected = delivered/collected`
- **Meaning:** explicit denominators
- **Symbols / units:** Mbit
- **Assumptions:** -
- **Relation to Paper 1/2:** P1 Eq.(5) (PDR)
- **Implementation:** `src/fed_matd3_iout/evaluation/metrics.py:episode_metrics`
- **Test:** `tests/test_integration.py::test_metric_definitions_and_denominators`
- **Config:** -
- **Known limitations:** -

### MET2 - Workload imbalance
**Class:** A  established

- **Expression:** `std(collected_i)/mean(collected_i)`
- **Meaning:** coefficient of variation across AUVs
- **Symbols / units:** -
- **Assumptions:** -
- **Relation to Paper 1/2:** new
- **Implementation:** `src/fed_matd3_iout/evaluation/metrics.py:episode_metrics`
- **Test:** `tests/test_integration.py::test_metric_definitions_and_denominators`
- **Config:** -
- **Known limitations:** -
