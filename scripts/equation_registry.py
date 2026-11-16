"""Single source of truth for the equation registry. `python scripts/build_equation_docs.py` renders
docs/MATHEMATICAL_MODEL_AUDIT.md and docs/EQUATION_IMPLEMENTATION_MAP.md; tests/test_equation_registry.py checks that every
referenced module:function and test exists. Classes: A established | B valid under stated assumptions | C proposed modelling choice
needing experimental justification | D approximation requiring validation | E incorrect/inconsistent as printed -> corrected.
Papers: P1 = SAC-TD3 (Ma et al.), P2 = STA-TA (Aon et al.). Test field None = documented test gap."""

M = "src/fed_matd3_iout/"
E = []


def eq(id, name, expr, meaning, symbols, assumptions, cls, paper, impl, test, config, limits):
    E.append(dict(id=id, name=name, expr=expr, meaning=meaning, symbols=symbols, assumptions=assumptions, cls=cls, paper=paper,
                  impl=impl, test=test, config=config, limits=limits))


T = "tests/"
# ------------------------------------------------------------------ queue
eq("Q0", "Poisson arrivals", "A_j(t) ~ Poisson(lambda_j(t) dt)", "new data units generated at sensor j in one step",
   "lambda [units/s], dt [s], A [units]", "lambda constant within a step; integer unit arrivals", "B", "P2 Sec. III-B; P1 Eq.(2)",
   M + "environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.step", T + "test_environment.py::test_env_arrivals_match_lambda",
   "traffic.base_rates, traffic.scale, traffic.rate_change_*", "Rate change is piecewise-constant; no burstiness/correlation.")
eq("Q1", "Pre-service occupancy", "B_pre = B + A", "occupancy after arrivals, before service", "units", "arrival-before-service", "B",
   "P2 Eq.(1)", M + "environment/sensor_queue.py:SensorQueue.arrive", T + "test_queue.py::test_arrival_before_service_order",
   "env.buffer_units", "Event order is never changed.")
eq("Q2", "Overflow", "L = max(0, B_pre - B_max)", "tail-dropped (lost) units", "units", "excess of NEW arrivals is dropped", "B", "P2 Eq.(2)",
   M + "environment/sensor_queue.py:SensorQueue.arrive", T + "test_queue.py::test_buffer_capacity_and_overflow", "env.buffer_units",
   "Drop policy (tail of new arrivals) is a modelling choice; value of lost packets tracked for diagnostics.")
eq("Q3", "Available backlog", "B_avail = min(B_max, B_pre)", "occupancy actually stored", "units", "as Q1-Q2", "B", "P2 Eq.(3)",
   M + "environment/sensor_queue.py:SensorQueue.arrive", T + "test_queue.py::test_buffer_capacity_and_overflow", "env.buffer_units", "")
eq("Q4", "Collected amount", "D = min{B_avail, R_ij dt / q, free_onboard/q}", "units removed from the sensor in one service step (FIFO)",
   "R [Mbit/s], q = Q/B_max [Mbit/unit]", "rate-limited; extra onboard-capacity cap is an addition to P2 (default 20 Mbit assumed)", "B",
   "P2 Eq.(4) (+onboard cap)", M + "environment/sensor_queue.py:SensorQueue.serve", T + "test_queue.py::test_no_negative_and_no_double_collection",
   "env.onboard_capacity_mbit, acoustic.sensor_link.max_rate_mbps", "Single sensor-link rate at hover (worst case r_dc); no per-step rate variation.")
eq("Q5", "Buffer update", "B(t+1) = B_avail - D", "residual backlog", "units", "", "B", "P2 Eq.(5)",
   M + "environment/sensor_queue.py:SensorQueue.serve", T + "test_queue.py::test_partial_batch_split_conserves_units", "", "")
eq("Q6", "Unit conversion", "q = Q / B_max  [Mbit per unit]", "Mbit <-> units conversion", "Q [Mbit], B_max [units]", "", "A", "P2 Sec. III-B",
   M + "environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.__init__", T + "test_mass_balance.py::test_env_end_to_end_mass_balance",
   "env.buffer_mbit, env.buffer_units", "")
eq("Q7", "Sensor mass balance", "G = D + L + B_res", "accounting identity per sensor and fleet (G includes initial contents)", "units or Mbit", "", "A",
   "P2 Eq.(7)", M + "environment/sensor_queue.py:SensorQueue.residual_check", T + "test_mass_balance.py::test_queue_mass_balance_random_ops", "", "")
eq("Q8", "End-to-end mass balance", "G = Delivered + Onboard + L + B_res", "extends Q7 to AUV-carried and USV-delivered data", "Mbit", "", "A",
   "extension", M + "environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.accounting", T + "test_mass_balance.py::test_env_end_to_end_mass_balance", "",
   "Verified to <1e-9 Mbit; dropped data after delivery not modelled.")
eq("Q9", "Hover duration", "h = max(1, ceil(Q b / (B R dt)))  [steps]", "hover steps computed once at contact", "b units, R Mbit/s", "computed once; capped",
   "B", "P2 Eq.(6)", M + "environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.step", T + "test_environment.py::test_hover_duration_rule",
   "env.max_hover_steps", "max_hover_steps cap is an ADDITION (default 60).")
# ------------------------------------------------------------------ prediction / capacity
eq("P0", "Travel time", "T_ij = d_ij / (v_nom + eps)", "planned travel time", "d [m], v_nom [m/s]", "straight-line path at constant nominal speed (P2)",
   "D", "P2 Eq.(15)", M + "allocation/service_arrival_predictor.py:travel_time_s", T + "test_service_arrival.py::test_zero_arrival_and_zero_travel",
   "allocation.v_nom", "Ignores heading changes, boundary rejections, drift and tracking error.")
eq("P1", "Exact expected backlog at arrival", "E[B_arr] = sum_n min(B_max, b+n) Pois(n; lambda T)",
   "expectation of capped backlog, no intermediate service", "b, B_max [units], lambda T [units]", "Poisson arrivals, constant lambda, known T, no service in between",
   "B", "extends P2 Eq.(16)", M + "allocation/service_arrival_predictor.py:expected_backlog_exact", T + "test_service_arrival.py::test_exact_matches_brute_force",
   "allocation.backlog_model=exact, allocation.poisson_tol", "Exact only given T; travel-time uncertainty not modelled. Tail truncation documented (P1-trunc).")
eq("P2", "Expected overflow", "E[L_arr] = sum_n [b+n-B_max]^+ Pois(n; lambda T) = b + lambda T - E[B_arr]", "expected loss before arrival",
   "units", "as P1", "B", "new", M + "allocation/service_arrival_predictor.py:expected_backlog_exact",
   T + "test_service_arrival.py::test_overflow_conservation_identity", "allocation.poisson_tol", "")
eq("P3", "Certainty-equivalent backlog", "B_hat = min(B_max, b + lambda T)", "fast approximation; UPPER BOUND on E[B_arr] (Jensen), NOT the exact expectation",
   "units", "replaces random arrivals by their mean", "D", "P2 Eqs.(16)-(17)", M + "allocation/service_arrival_predictor.py:certainty_equivalent_backlog",
   T + "test_service_arrival.py::test_certainty_equivalent_is_upper_bound_not_exact", "allocation.backlog_model=ce", "Gap largest when b+lambda T ~ B_max.")
eq("P4", "Truncated Poisson sum", "n_hi = min(floor(B_max-b), Q_{1-tol}); tail by sum_{n>N} n p_n = m(1-F(N-1))", "controlled truncation of P1",
   "tol [-]", "tol small", "B", "new", M + "allocation/service_arrival_predictor.py:expected_backlog_exact",
   T + "test_service_arrival.py::test_truncation_accuracy_bound", "allocation.poisson_tol (1e-9)", "BUG FOUND+FIXED: first version valued the tail at B_max (error ~tol*B_max).")
eq("SC1", "Service capacity (constant rate)", "C = R T_contact", "Mbit transferable in a contact", "R [Mbit/s], T [s]", "constant rate", "B", "task spec",
   M + "allocation/service_capacity.py:capacity_constant", T + "test_service_arrival.py::test_service_capacity_constant_and_integral", "allocation.contact_time_s", "")
eq("SC2", "Service capacity (time-varying)", "C = integral R(tau) dtau", "trapezoid quadrature", "Mbit", "R integrable", "D", "task spec",
   M + "allocation/service_capacity.py:capacity_integral", T + "test_service_arrival.py::test_service_capacity_constant_and_integral", "",
   "Utility function only; the environment uses constant hover rate (SC1).")
eq("SC3", "Collectable volume", "D_service = min{B_arr, C/q, free/q}", "predicted collectable units", "units", "", "B", "task spec",
   M + "allocation/service_capacity.py:collectable_units", T + "test_service_arrival.py::test_service_capacity_constant_and_integral", "", "Prediction only; collection removes units once (Q4).")
# ------------------------------------------------------------------ semantics
eq("S1", "Value of information", "VoI(tau) = sigma exp(-mu tau)", "exponentially decaying value", "sigma [0,1], mu [1/s], tau [s]", "exponential decay",
   "C", "P1 Eq.(17)", M + "semantic/value_of_information.py:voi", T + "test_semantic_utility.py::test_voi_decay", "semantic.categories.*.mu",
   "Decay law is a modelling choice. mu(high) NOT given in P1 (assumed = routine).")
eq("S2", "Packet semantic utility", "U_k = d_k c_k sigma_k exp(-mu_k tau_k^e2e)", "utility of a delivered packet; criticality c enters ONCE",
   "d in {0,1}, c >= 1", "tau = delivery time - generation time (incl. queueing, propagation)", "C", "P1 Eq.(19)",
   M + "semantic/semantic_utility.py:delivered_utility", T + "test_semantic_utility.py::test_mission_criticality_applied_once", "semantic.categories.*.c_priority",
   "tau omits processing and USV->shore RF delay (P1 Eq.14).")
eq("S3", "Total utility", "U = sum_k U_k", "total semantic utility; each delivered chunk recorded once", "-", "", "C", "P1 Eq.(19)",
   M + "semantic/semantic_utility.py:UtilityLedger.record", T + "test_semantic_utility.py::test_no_duplicate_utility", "", "")
eq("S4", "Normalised utilities", "U_norm = U / sum_generated c sigma ;  U_per_packet = U / N_del", "U_per_packet is P1's eta_sem; U_norm is generated-referenced",
   "-", "", "C", "P1 Eq.(20) (U_per_packet)", M + "evaluation/metrics.py:episode_metrics", T + "test_integration.py::test_metric_definitions_and_denominators", "",
   "U_norm can exceed 1 (c>1).")
eq("S5", "Packet importance", "sigma = clip(omega_sensor * sigma_factor(category), 0, 1)", "normalised importance; NO criticality factor", "omega in {0.3,0.6,1.0}",
   "static omega per sensor; per-packet category", "C", "P1 Eq.(18) (E[sigma] identical: omega(p_c+p_h+(1-p_c-p_h)gamma_low))",
   M + "semantic/mission_criticality.py:sigma_of", T + "test_semantic_utility.py::test_importance_normalised", "semantic.categories, semantic.hotspot_weights",
   "SIMPLIFICATION: P1 evolves omega_m(t) by a Markov chain and uses clusters; not reproduced.")
eq("S6", "Aggregate semantic density", "u_j = E[c sigma]_j / max_k E[c sigma]_k", "AGGREGATE APPROXIMATION used only in allocation cost (A1) - never in the objective",
   "[0,1]", "category mix known", "D", "new", M + "semantic/mission_criticality.py:mean_semantic_value_density",
   T + "test_semantic_utility.py::test_density_aggregate_approximation", "", "")
# ------------------------------------------------------------------ allocation
eq("A1", "Semantic-service assignment cost", "C_ij = -w_b B~ - w_s U~ + w_d T~ + w_o L~ + w_c S~ + w_l W~ - w_a A~",
   "normalised (all in [0,1]) assignment cost", "see semantic_service_cost.py docstring for each term", "terms commensurate after normalisation", "C",
   "extends P2 Eq.(20) (b,d,sw terms identical, weights 1,1,0.05)", M + "allocation/semantic_service_cost.py:cost_matrix",
   T + "test_task_allocator.py::test_cost_terms_normalised", "allocation.weights.*", "PROPOSED. semantic/overflow/workload/age terms are new; each ablated.")
eq("A2", "One-to-one assignment", "min sum C_ij x_ij s.t. sum_j x_ij = 1, sum_i x_ij <= 1", "linear assignment (Hungarian / shortest augmenting path)",
   "x binary", "rectangular allowed; surplus AUVs idle", "A", "P2 Eqs.(8)-(9),(11)", M + "allocation/task_allocator.py:TaskAllocator.allocate",
   T + "test_task_allocator.py::test_one_to_one_and_optimal", "allocation.method", "")
eq("A3", "Priority score", "rho_ij = lambda_j b_j / B_j - alpha d_ij", "Xu/P2 greedy & Hungarian-priority baselines", "alpha [1/m]", "", "A", "P2 Sec. V-B",
   M + "allocation/task_allocator.py:priority_matrix", T + "test_task_allocator.py::test_greedy_reassigns_only_freed", "allocation.priority_alpha", "")
# ------------------------------------------------------------------ energy
eq("E1", "Propulsion power", "P_prop = P0 + kappa v^3", "hotel/baseline + cubic drag", "W, kappa [W/(m/s)^3]", "simplified, UNCALIBRATED", "C", "replaces quadratic action penalty of P1 Eq.(15)/P2",
   M + "environment/energy.py:propulsion_power", T + "test_environment.py::test_energy_model", "energy.p0_w, energy.kappa", "Coefficients are placeholders; calibrate before publication.")
eq("E2", "Propulsion energy", "E_prop = P_prop dt", "energy per step", "J", "", "A", "-", M + "environment/energy.py:propulsion_energy",
   T + "test_environment.py::test_energy_model", "env.dt_s", "")
eq("E3", "Communication energy", "E_tx = P_tx t_air ; E_rx = P_rx t_rx", "tx for data/model uploads, rx for sensor reception and downloads", "J", "electrical power constant", "B", "P1 Eq.(15)",
   M + "environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.add_comm_energy", T + "test_environment.py::test_energy_components_disjoint", "energy.p_tx_elec_w, acoustic.rx_power_w",
   "Disjoint from E1/E4; processing power of P1 not modelled separately.")
eq("E4", "Computation energy", "E_comp = e_grad * (#gradient steps)", "local training energy", "J", "constant energy per step (assumed)", "C", "P1 Eq.(15) P_proc",
   M + "training/local_training.py:local_update", T + "test_environment.py::test_compute_energy_accounting", "energy.e_grad_j", "Zero in evaluation episodes (no learning); inference energy not modelled.")
# ------------------------------------------------------------------ acoustic
eq("C1", "Thorp absorption", "a(f)=0.11f^2/(1+f^2)+44f^2/(4100+f^2)+2.75e-4 f^2+0.003  [dB/km, f kHz]", "frequency-dependent absorption", "f [kHz]", "valid ~0.4-50 kHz", "A",
   "P1 Eq.(9)", M + "communication/acoustic_channel.py:thorp_absorption_db_per_km", T + "test_acoustic_channel.py::test_thorp_known_value_and_monotone", "acoustic.*.freq_khz", "")
eq("C2", "Transmission loss", "TL = k 10log10(max(d,1 m)/1 m) + a(f) d/1000  [dB]", "spreading + absorption, 1 m reference", "d [m], k=1.5", "far-field, isovelocity, 1 m reference range", "B",
   "P1 Eq.(8) used l=d/1000 km with l^k and no reference range -> class E as printed", M + "communication/acoustic_channel.py:transmission_loss_db",
   T + "test_acoustic_channel.py::test_spreading_exponent_exact", "acoustic.spreading_k",
   "DISCREPANCY: P1 Eq.(8) as printed gives TL<0 dB (gain) for d<1 km; this code uses the 1 m reference consistent with SL re 1 m. Clarify in the paper.")
eq("C3", "Ambient noise", "N(f) = 10log10(sum of turbulence+shipping+wind+thermal in linear power)  [dB re uPa^2/Hz]", "Stojanovic/Coates empirical PSDs", "f [kHz], s, w [m/s]",
   "components added in linear domain", "B", "P1 Eq.(10)", M + "communication/acoustic_channel.py:noise_psd_db", T + "test_acoustic_channel.py::test_source_level_and_noise_model",
   "acoustic.shipping_s, acoustic.wind_mps", "Empirical, valid for stated f range only.")
eq("C4", "Source level", "SL = 170.8 + 10log10(eta P_elec) + DI  [dB re uPa @1m]", "sonar-equation source level", "P [W]", "", "A", "replaces P1 Eq.(11) P_t/(PL N df) (unit mix)",
   M + "communication/acoustic_channel.py:source_level_db", T + "test_acoustic_channel.py::test_source_level_and_noise_model", "acoustic.*.p_elec_w, efficiency", "Powers/efficiencies are ASSUMPTIONS.")
eq("C5", "Received SNR", "SNR_dB = SL - TL - (N(f)_dB + 10log10 B_Hz)", "mean SNR; dB domain throughout", "dB, Hz", "narrowband", "B", "P1 Eq.(11) (corrected units)",
   M + "communication/acoustic_channel.py:AcousticLink.snr_db", T + "test_acoustic_channel.py::test_snr_unit_consistency", "acoustic.*.bandwidth_hz", "")
eq("C6", "Achievable rate", "R = min(R_max, B log2(1+SNR_lin)) / 1e6  [Mbit/s]", "Shannon on mean SNR with modem cap", "B [Hz]", "ignores fading outage and coding gap", "D",
   "P2 uses the rate model of Xu et al. (details not reproduced)", M + "communication/acoustic_channel.py:shannon_rate_mbps", T + "test_acoustic_channel.py::test_rate_capped",
   "acoustic.*.max_rate_mbps", "Rate caps are ASSUMPTIONS and strongly shape results.")
eq("C7", "Propagation delay", "tau = d / v_s", "one-way delay", "d [m], v_s=1500 m/s", "constant sound speed", "B", "P1 Eq.(13)",
   M + "communication/propagation_delay.py:propagation_delay_s", T + "test_acoustic_channel.py::test_propagation_delay", "acoustic.sound_speed_mps", "No refraction/multipath.")
eq("C8", "Rayleigh packet success", "P = exp(-gamma_th / SNR_mean)", "P(instantaneous SNR >= gamma_th), SNR ~ Exp(mean)", "-", "Rayleigh amplitude => exponential power; capture threshold", "B",
   "P1 Eq.(12) (consistent: beta = gamma_th)", M + "communication/acoustic_channel.py:rayleigh_success_prob", T + "test_acoustic_channel.py::test_packet_success_bounds_and_derivation",
   "acoustic.*.snr_threshold_db", "Verified by Monte-Carlo. Per-step draw; no temporal fading correlation.")
eq("C9", "Contact/outage process", "range gate d_slant <= R_usv; Gilbert-Elliott outage P(up->down)=p_d, P(down->up)=p_r", "intermittent AUV-USV windows",
   "p [1/step]", "two-state Markov outage independent per AUV", "C", "new", M + "communication/packet_delivery.py:LinkProcess.step",
   T + "test_acoustic_channel.py::test_gilbert_elliott_stationary_fraction", "acoustic.usv_link.*", "Outage law is an assumption (no data).")
# ------------------------------------------------------------------ MATD3
eq("M1", "Actor", "a_i = mu_i(o_i), tanh output", "deterministic bounded policy", "-", "", "A", "P2 Sec. IV-F", M + "agents/actor.py:Actor.forward", T + "test_matd3.py::test_actor_output_bounds", "matd3.hidden", "")
eq("M2", "Smoothed joint target action", "a'_k = clip(mu'_k(o'_k) + clip(eps,-c,c), -1, 1) for ALL k", "TD3 target-policy smoothing with all target actors", "eps~N(0,sigma^2)", "",
   "A", "TD3/MATD3", M + "agents/matd3_agent.py:MATD3Agent.target_joint_action", T + "test_matd3.py::test_joint_action_uses_all_target_actors", "matd3.target_noise, matd3.noise_clip", "")
eq("M3", "Clipped double-Q target", "y_i = r_i + gamma (1-d) min_k Q'_{i,k}(s', a')", "bootstrapped target; d=0 at time-limit truncation", "-", "", "A", "TD3/MATD3",
   M + "agents/matd3_agent.py:MATD3Agent.compute_target", T + "test_matd3.py::test_twin_target_minimum_and_terminal", "matd3.gamma", "")
eq("M4", "Critic loss", "L = E[(Q_{i,k}(s,a_1..a_M) - y_i)^2]", "centralised twin critics (state + joint action)", "-", "simulator state available at TRAINING only (CTDE)", "A", "MATD3",
   M + "agents/matd3_agent.py:MATD3Agent.update", T + "test_target_updates.py::test_critic_loss_decreases_on_fixed_batch", "matd3.critic_lr, matd3.critic_mode", "Training is NOT fully decentralised.")
eq("M5", "Deterministic policy gradient", "grad J = E[grad_a Q_{i,1}(s, a_{-i}, mu_i(o_i)) grad mu_i]", "other agents' actions from replay", "-", "", "A", "DPG",
   M + "agents/matd3_agent.py:MATD3Agent.update", T + "test_target_updates.py::test_actor_gradient_flows_only_through_own_action", "matd3.actor_lr", "")
eq("M6", "Delayed update + Polyak", "every d-th critic step: actor step; theta' <- tau theta + (1-tau) theta'", "delayed actor and target updates", "tau, d", "", "A", "TD3",
   M + "agents/matd3_agent.py:polyak_update", T + "test_target_updates.py::test_delayed_actor_update_and_target_ordering", "matd3.tau, matd3.policy_delay", "")
# ------------------------------------------------------------------ federated
eq("F1", "Local update", "Delta_i = theta_i^local - theta_i^base", "actor parameter difference vs last synchronised global model", "flat params", "one outstanding update per AUV", "A", "new",
   M + "federated/federated_client.py:FederatedClient.make_update", T + "test_federated_aggregation.py::test_client_update_construction_and_versions", "federated.local_steps_per_update",
   "Delta is snapshotted when transmission STARTS.")
eq("F2", "Staleness", "Delta_i(stale) = r - v_i", "server version minus base version", "rounds", "", "A", "new", M + "federated/staleness.py:staleness",
   T + "test_staleness.py::test_staleness_definition_and_factor", "", "")
eq("F3", "Staleness factor", "w_stale = exp(-kappa Delta)", "down-weighting of stale updates", "kappa", "", "C", "new", M + "federated/staleness.py:staleness_factor",
   T + "test_staleness.py::test_staleness_definition_and_factor", "federated.kappa", "Functional form unvalidated.")
eq("F3b", "Max-staleness gate", "accept iff 0 <= Delta <= Delta_max", "reject over-stale updates", "rounds", "", "C", "new", M + "federated/staleness.py:is_acceptable",
   T + "test_staleness.py::test_max_staleness_gate", "federated.max_staleness", "")
eq("F4", "Semantic-service relevance", "M = beta rho_sem + (1-beta) rho_svc;  rho_sem = clip(U_del/V_coll); rho_svc = C/(C+L_own)", "relevance in [0,1] from MEASURED window statistics",
   "see relevance.py", "window since last update", "C", "new", M + "federated/relevance.py:relevance", T + "test_federated_aggregation.py::test_relevance_measures", "federated.beta_sem",
   "Delivered and collected are windowed separately -> rho_sem may exceed 1 (clipped). Own-sensor overflow depends on allocation, not only on the AUV's policy.")
eq("F5", "Aggregation weight", "Z_i = n_i^alpha exp(-kappa Delta_i) [eps_M + (1-eps_M) M_i]^eta", "NOT unbiased FedAvg; link reliability NOT multiplied in", "n_i = grad steps x batch size",
   "weights change the effective objective", "C", "new", M + "federated/aggregation.py:raw_weights", T + "test_federated_aggregation.py::test_weight_formula_and_normalisation",
   "federated.alpha, eta, eps_m", "No convergence guarantee; no privacy guarantee.")
eq("F6", "Normalisation and cap", "w_i = Z_i / sum_R Z_j; optional w_i <= w_max with iterative renormalisation", "convex weights over RECEIVED+ACCEPTED updates only", "-", "", "C", "new",
   M + "federated/aggregation.py:normalise_with_cap", T + "test_federated_aggregation.py::test_weight_cap_and_renormalisation", "federated.max_client_weight", "Cap below 1/|R| infeasible -> uniform.")
eq("F7", "L2 clipping", "Clip_C(d) = d min(1, C/||d||_2)", "global L2-norm clipping", "C", "", "A", "standard", M + "federated/aggregation.py:l2_clip",
   T + "test_federated_aggregation.py::test_l2_clip", "federated.clip_norm", "No noise added -> no differential privacy.")
eq("F8", "Server aggregation", "theta^{r+1} = theta^r + eta_g sum_R w_i Clip(Delta_i)", "server step", "eta_g", "equals FedAvg only for w_i=n_i/sum n, eta_g=1, no clip", "B", "new",
   M + "federated/aggregation.py:aggregate", T + "test_federated_aggregation.py::test_aggregate_formula_and_empty_and_mismatch", "federated.server_lr", "")
eq("F9", "Global install", "theta_local <- b theta_g + (1-b) theta_local", "download blend (b=1: replace)", "b", "critics stay local", "C", "new",
   M + "federated/federated_client.py:FederatedClient.install_global", T + "test_federated_aggregation.py::test_client_update_construction_and_versions", "federated.download_blend", "")
eq("F10", "Transfer feasibility", "progress += R*share*dt only when contact up and fading draw succeeds; received at t_tx_done + tau_prop", "model upload/download over intermittent link",
   "bytes", "fragmentation optional", "C", "new", M + "communication/model_transfer.py:ModelTransfer.step", T + "test_contact_scheduler.py::test_transfer_needs_multiple_steps_and_propagation",
   "federated.allow_fragmentation, acoustic.usv_link.model_share", "")
eq("F11", "Contact scheduler", "start iff link up (and, without fragmentation, airtime <= expected remaining window)", "defer/start decision", "-", "EWMA window estimate", "C", "new",
   M + "communication/contact_scheduler.py:ContactScheduler.may_start", T + "test_contact_scheduler.py::test_defers_when_update_does_not_fit_window", "federated.allow_fragmentation", "")
eq("F12", "No aggregation before receipt", "buffer only filled by USVServer.receive() after last bit arrives", "causality", "-", "", "A", "spec rule 9",
   M + "federated/usv_server.py:USVServer.receive", T + "test_federated_aggregation.py::test_no_aggregation_before_receipt", "", "")
# ------------------------------------------------------------------ reward
eq("R1", "Team + local reward", "r_i = (1-zeta) r_team + zeta r_local + w_sh F;  r_team = w_u dU/U_ref - w_l L/L_ref - w_e E/E_ref - w_c C_coll - w_o C_bound",
   "normalised; utility already includes delivered value so NO separate delivery term", "-", "", "C", "replaces P2 Eq.(14) (different weights/terms)",
   M + "environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.step", T + "test_environment.py::test_reward_collision_penalty", "reward.*, federated.local_reward_weight",
   "Magnitudes measured: random policy -391/episode, goal-seeking reference -21. Only the collision term has a dedicated unit test.")
eq("R2", "Potential-based shaping", "F = (|p-g| - gamma|p'-g|)/D_n", "goal-approach shaping; policy-invariant for a FIXED goal (Ng et al. 1999)", "-", "fixed goal per step", "D", "P2 uses dense -0.6|p-s|",
   M + "environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.step", None, "reward.w_shape", "Invariance does not strictly hold when the target changes mid-episode (non-stationary potential). TEST GAP.")
# ------------------------------------------------------------------ metrics
eq("MET1", "Ratios", "collection = collected/G; delivery (PDR) = delivered/G; delivered-of-collected = delivered/collected", "explicit denominators", "Mbit", "", "A", "P1 Eq.(5) (PDR)",
   M + "evaluation/metrics.py:episode_metrics", T + "test_integration.py::test_metric_definitions_and_denominators", "", "")
eq("MET2", "Workload imbalance", "std(collected_i)/mean(collected_i)", "coefficient of variation across AUVs", "-", "", "A", "new", M + "evaluation/metrics.py:episode_metrics",
   T + "test_integration.py::test_metric_definitions_and_denominators", "", "")
