# Experiment Protocol

**Matched design.** Evaluation scenario k uses stream offset `eval_scenario_base + k` for layout, arrivals, fading and comm seeds, identical for every method
(tested). Training episodes use offsets `seed*10000 + e`, disjoint from evaluation. Same environment, reward, and experience budget
(`train_episodes x episode_steps`) for every method. Federated methods get no extra environment interaction.

**Unit of replication = training seed.** Each seed's score is the mean over the same evaluation scenarios; across-seed SD = training variability; within-seed SD = mission-to-mission
variability. Report mean, SD, 95% t-CI; paired t and Wilcoxon on seed scores, Cohen's d (paired), Holm correction. **Paired tests need >= 2 seeds; use >= 5.**

**Methods (11).** greedy_matd3, hungarian_matd3, centralized_matd3, independent_td3, maddpg, mappo, fedavg_matd3, stale_fedmatd3, semantic_fedmatd3, service_fedmatd3, s3a_fedmatd3.
Comparability notes are in `baselines/registry.py`: non-allocation baselines use the proposed STA allocator (only learning differs); weighting-only federated baselines keep the
proposed clipping and cap (only weights differ); FedAvg-MATD3 is conventional (no clip/cap/relevance/staleness) on the same asynchronous protocol; MAPPO is on-policy (not
sample-efficiency comparable); greedy/hungarian use MATD3 mobility without federation.

**Ablations (11)** `baselines/registry.py::ABLATIONS` (mirrored in `configs/ablations.yaml`; a test checks each ablation changes only its own switches).
**Scalability:** (1 AUV,10 sensors), (3,20), (10,40), always one USV. Any two-USV Paper-2 scaling point is adapted to one USV.
**Robustness:** high loss via `acoustic.usv_link.outage_p_drop`, long interruptions via `outage_p_recover`, non-IID via `traffic.mode=regional` and `federated.client_speed`,
rate shifts via `traffic.rate_change_step/factor`, AUV count via scalability.

**Cost (extrapolated from a measured ~10 s/episode, 3 AUVs, 2-core CPU; not measured at scale):** 500 episodes ~ 1.5 h per run; 11 methods x 5 seeds ~ 70+ h on CPU.
Use `--episodes`, `--seeds` and GPU (`run.device`) to scale.
