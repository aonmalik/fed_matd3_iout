# Equation -> Implementation Map

Generated from `scripts/equation_registry.py`; `tests/test_equation_registry.py` verifies every path/function/test below exists.

| ID | Class | Module : function | Test | Config parameter |
|---|---|---|---|---|
| Q0 | B | `src/fed_matd3_iout/environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.step` | `tests/test_environment.py::test_env_arrivals_match_lambda` | traffic.base_rates, traffic.scale, traffic.rate_change_* |
| Q1 | B | `src/fed_matd3_iout/environment/sensor_queue.py:SensorQueue.arrive` | `tests/test_queue.py::test_arrival_before_service_order` | env.buffer_units |
| Q2 | B | `src/fed_matd3_iout/environment/sensor_queue.py:SensorQueue.arrive` | `tests/test_queue.py::test_buffer_capacity_and_overflow` | env.buffer_units |
| Q3 | B | `src/fed_matd3_iout/environment/sensor_queue.py:SensorQueue.arrive` | `tests/test_queue.py::test_buffer_capacity_and_overflow` | env.buffer_units |
| Q4 | B | `src/fed_matd3_iout/environment/sensor_queue.py:SensorQueue.serve` | `tests/test_queue.py::test_no_negative_and_no_double_collection` | env.onboard_capacity_mbit, acoustic.sensor_link.max_rate_mbps |
| Q5 | B | `src/fed_matd3_iout/environment/sensor_queue.py:SensorQueue.serve` | `tests/test_queue.py::test_partial_batch_split_conserves_units` | - |
| Q6 | A | `src/fed_matd3_iout/environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.__init__` | `tests/test_mass_balance.py::test_env_end_to_end_mass_balance` | env.buffer_mbit, env.buffer_units |
| Q7 | A | `src/fed_matd3_iout/environment/sensor_queue.py:SensorQueue.residual_check` | `tests/test_mass_balance.py::test_queue_mass_balance_random_ops` | - |
| Q8 | A | `src/fed_matd3_iout/environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.accounting` | `tests/test_mass_balance.py::test_env_end_to_end_mass_balance` | - |
| Q9 | B | `src/fed_matd3_iout/environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.step` | `tests/test_environment.py::test_hover_duration_rule` | env.max_hover_steps |
| P0 | D | `src/fed_matd3_iout/allocation/service_arrival_predictor.py:travel_time_s` | `tests/test_service_arrival.py::test_zero_arrival_and_zero_travel` | allocation.v_nom |
| P1 | B | `src/fed_matd3_iout/allocation/service_arrival_predictor.py:expected_backlog_exact` | `tests/test_service_arrival.py::test_exact_matches_brute_force` | allocation.backlog_model=exact, allocation.poisson_tol |
| P2 | B | `src/fed_matd3_iout/allocation/service_arrival_predictor.py:expected_backlog_exact` | `tests/test_service_arrival.py::test_overflow_conservation_identity` | allocation.poisson_tol |
| P3 | D | `src/fed_matd3_iout/allocation/service_arrival_predictor.py:certainty_equivalent_backlog` | `tests/test_service_arrival.py::test_certainty_equivalent_is_upper_bound_not_exact` | allocation.backlog_model=ce |
| P4 | B | `src/fed_matd3_iout/allocation/service_arrival_predictor.py:expected_backlog_exact` | `tests/test_service_arrival.py::test_truncation_accuracy_bound` | allocation.poisson_tol (1e-9) |
| SC1 | B | `src/fed_matd3_iout/allocation/service_capacity.py:capacity_constant` | `tests/test_service_arrival.py::test_service_capacity_constant_and_integral` | allocation.contact_time_s |
| SC2 | D | `src/fed_matd3_iout/allocation/service_capacity.py:capacity_integral` | `tests/test_service_arrival.py::test_service_capacity_constant_and_integral` | - |
| SC3 | B | `src/fed_matd3_iout/allocation/service_capacity.py:collectable_units` | `tests/test_service_arrival.py::test_service_capacity_constant_and_integral` | - |
| S1 | C | `src/fed_matd3_iout/semantic/value_of_information.py:voi` | `tests/test_semantic_utility.py::test_voi_decay` | semantic.categories.*.mu |
| S2 | C | `src/fed_matd3_iout/semantic/semantic_utility.py:delivered_utility` | `tests/test_semantic_utility.py::test_mission_criticality_applied_once` | semantic.categories.*.c_priority |
| S3 | C | `src/fed_matd3_iout/semantic/semantic_utility.py:UtilityLedger.record` | `tests/test_semantic_utility.py::test_no_duplicate_utility` | - |
| S4 | C | `src/fed_matd3_iout/evaluation/metrics.py:episode_metrics` | `tests/test_integration.py::test_metric_definitions_and_denominators` | - |
| S5 | C | `src/fed_matd3_iout/semantic/mission_criticality.py:sigma_of` | `tests/test_semantic_utility.py::test_importance_normalised` | semantic.categories, semantic.hotspot_weights |
| S6 | D | `src/fed_matd3_iout/semantic/mission_criticality.py:mean_semantic_value_density` | `tests/test_semantic_utility.py::test_density_aggregate_approximation` | - |
| A1 | C | `src/fed_matd3_iout/allocation/semantic_service_cost.py:cost_matrix` | `tests/test_task_allocator.py::test_cost_terms_normalised` | allocation.weights.* |
| A2 | A | `src/fed_matd3_iout/allocation/task_allocator.py:TaskAllocator.allocate` | `tests/test_task_allocator.py::test_one_to_one_and_optimal` | allocation.method |
| A3 | A | `src/fed_matd3_iout/allocation/task_allocator.py:priority_matrix` | `tests/test_task_allocator.py::test_greedy_reassigns_only_freed` | allocation.priority_alpha |
| E1 | C | `src/fed_matd3_iout/environment/energy.py:propulsion_power` | `tests/test_environment.py::test_energy_model` | energy.p0_w, energy.kappa |
| E2 | A | `src/fed_matd3_iout/environment/energy.py:propulsion_energy` | `tests/test_environment.py::test_energy_model` | env.dt_s |
| E3 | B | `src/fed_matd3_iout/environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.add_comm_energy` | `tests/test_environment.py::test_energy_components_disjoint` | energy.p_tx_elec_w, acoustic.rx_power_w |
| E4 | C | `src/fed_matd3_iout/training/local_training.py:local_update` | `tests/test_environment.py::test_compute_energy_accounting` | energy.e_grad_j |
| C1 | A | `src/fed_matd3_iout/communication/acoustic_channel.py:thorp_absorption_db_per_km` | `tests/test_acoustic_channel.py::test_thorp_known_value_and_monotone` | acoustic.*.freq_khz |
| C2 | B | `src/fed_matd3_iout/communication/acoustic_channel.py:transmission_loss_db` | `tests/test_acoustic_channel.py::test_spreading_exponent_exact` | acoustic.spreading_k |
| C3 | B | `src/fed_matd3_iout/communication/acoustic_channel.py:noise_psd_db` | `tests/test_acoustic_channel.py::test_source_level_and_noise_model` | acoustic.shipping_s, acoustic.wind_mps |
| C4 | A | `src/fed_matd3_iout/communication/acoustic_channel.py:source_level_db` | `tests/test_acoustic_channel.py::test_source_level_and_noise_model` | acoustic.*.p_elec_w, efficiency |
| C5 | B | `src/fed_matd3_iout/communication/acoustic_channel.py:AcousticLink.snr_db` | `tests/test_acoustic_channel.py::test_snr_unit_consistency` | acoustic.*.bandwidth_hz |
| C6 | D | `src/fed_matd3_iout/communication/acoustic_channel.py:shannon_rate_mbps` | `tests/test_acoustic_channel.py::test_rate_capped` | acoustic.*.max_rate_mbps |
| C7 | B | `src/fed_matd3_iout/communication/propagation_delay.py:propagation_delay_s` | `tests/test_acoustic_channel.py::test_propagation_delay` | acoustic.sound_speed_mps |
| C8 | B | `src/fed_matd3_iout/communication/acoustic_channel.py:rayleigh_success_prob` | `tests/test_acoustic_channel.py::test_packet_success_bounds_and_derivation` | acoustic.*.snr_threshold_db |
| C9 | C | `src/fed_matd3_iout/communication/packet_delivery.py:LinkProcess.step` | `tests/test_acoustic_channel.py::test_gilbert_elliott_stationary_fraction` | acoustic.usv_link.* |
| M1 | A | `src/fed_matd3_iout/agents/actor.py:Actor.forward` | `tests/test_matd3.py::test_actor_output_bounds` | matd3.hidden |
| M2 | A | `src/fed_matd3_iout/agents/matd3_agent.py:MATD3Agent.target_joint_action` | `tests/test_matd3.py::test_joint_action_uses_all_target_actors` | matd3.target_noise, matd3.noise_clip |
| M3 | A | `src/fed_matd3_iout/agents/matd3_agent.py:MATD3Agent.compute_target` | `tests/test_matd3.py::test_twin_target_minimum_and_terminal` | matd3.gamma |
| M4 | A | `src/fed_matd3_iout/agents/matd3_agent.py:MATD3Agent.update` | `tests/test_target_updates.py::test_critic_loss_decreases_on_fixed_batch` | matd3.critic_lr, matd3.critic_mode |
| M5 | A | `src/fed_matd3_iout/agents/matd3_agent.py:MATD3Agent.update` | `tests/test_target_updates.py::test_actor_gradient_flows_only_through_own_action` | matd3.actor_lr |
| M6 | A | `src/fed_matd3_iout/agents/matd3_agent.py:polyak_update` | `tests/test_target_updates.py::test_delayed_actor_update_and_target_ordering` | matd3.tau, matd3.policy_delay |
| F1 | A | `src/fed_matd3_iout/federated/federated_client.py:FederatedClient.make_update` | `tests/test_federated_aggregation.py::test_client_update_construction_and_versions` | federated.local_steps_per_update |
| F2 | A | `src/fed_matd3_iout/federated/staleness.py:staleness` | `tests/test_staleness.py::test_staleness_definition_and_factor` | - |
| F3 | C | `src/fed_matd3_iout/federated/staleness.py:staleness_factor` | `tests/test_staleness.py::test_staleness_definition_and_factor` | federated.kappa |
| F3b | C | `src/fed_matd3_iout/federated/staleness.py:is_acceptable` | `tests/test_staleness.py::test_max_staleness_gate` | federated.max_staleness |
| F4 | C | `src/fed_matd3_iout/federated/relevance.py:relevance` | `tests/test_relevance_attribution.py::test_relevance_bounded_and_meaningful` | federated.beta_sem, federated.maturity_eps |
| F13 | C | `src/fed_matd3_iout/semantic/attribution.py:BatchBook` | `tests/test_relevance_attribution.py::test_batch_collected_in_one_window_delivered_in_another` | federated.maturity_eps |
| F14 | A | `src/fed_matd3_iout/federated/update_packet.py:UpdatePacket.transition` | `tests/test_update_lifecycle.py::test_successful_aggregation_advances_reference_exactly_once` | federated.max_buffer_age_s, federated.upload_timeout_s |
| F15 | A | `src/fed_matd3_iout/federated/communication_protocol.py:FederatedProtocol._settle` | `tests/test_stats_accounting.py::test_old_episode_stats_cannot_contaminate_next_episode` | - |
| F5 | C | `src/fed_matd3_iout/federated/aggregation.py:raw_weights` | `tests/test_federated_aggregation.py::test_weight_formula_and_normalisation` | federated.alpha, eta, eps_m |
| F6 | C | `src/fed_matd3_iout/federated/aggregation.py:normalise_with_cap` | `tests/test_federated_aggregation.py::test_weight_cap_and_renormalisation` | federated.max_client_weight |
| F7 | A | `src/fed_matd3_iout/federated/aggregation.py:l2_clip` | `tests/test_federated_aggregation.py::test_l2_clip` | federated.clip_norm |
| F8 | B | `src/fed_matd3_iout/federated/aggregation.py:aggregate` | `tests/test_federated_aggregation.py::test_aggregate_formula_and_empty_and_mismatch` | federated.server_lr |
| F9 | C | `src/fed_matd3_iout/federated/federated_client.py:FederatedClient.install_global` | `tests/test_federated_aggregation.py::test_client_update_construction_and_versions` | federated.download_blend |
| F10 | C | `src/fed_matd3_iout/communication/model_transfer.py:ModelTransfer.step` | `tests/test_contact_scheduler.py::test_transfer_needs_multiple_steps_and_propagation` | federated.allow_fragmentation, acoustic.usv_link.model_share |
| F11 | C | `src/fed_matd3_iout/communication/contact_scheduler.py:ContactScheduler.may_start` | `tests/test_contact_scheduler.py::test_defers_when_update_does_not_fit_window` | federated.allow_fragmentation |
| F12 | A | `src/fed_matd3_iout/federated/usv_server.py:USVServer.receive` | `tests/test_federated_aggregation.py::test_no_aggregation_before_receipt` | - |
| R1 | C | `src/fed_matd3_iout/environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.step` | `tests/test_environment.py::test_reward_collision_penalty` | reward.*, federated.local_reward_weight |
| R2 | D | `src/fed_matd3_iout/environment/multi_auv_iout_env.py:MultiAUVIoUTEnv.step` | `GAP` | reward.w_shape |
| MET1 | A | `src/fed_matd3_iout/evaluation/metrics.py:episode_metrics` | `tests/test_integration.py::test_metric_definitions_and_denominators` | - |
| MET2 | A | `src/fed_matd3_iout/evaluation/metrics.py:episode_metrics` | `tests/test_integration.py::test_metric_definitions_and_denominators` | - |
