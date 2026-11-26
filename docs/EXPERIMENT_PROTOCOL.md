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

**Current protocol: 2.1 (preparation; frozen in `configs/experiment_manifest.json`, `config_hash c76394bf7eaa9005`; `scripts/validate_manifest.py` exits 1 on any mismatch). 500 training episodes (Paper 2 Table I), 10 independent training seeds (0-9), 20 matched evaluation scenarios.** Protocol 2.0 (5 seeds) is preserved in `configs/protocols/v2.0/` (`python scripts/validate_manifest.py --protocol 2.0`). See `docs/PROTOCOL_CHANGELOG.md`.
**Cost (MEASURED for `s3a_fedmatd3` only: 3-episode and 6-episode probes at full episode length, 3 AUVs, 2-core CPU: 12.2 and 11.4 s/episode; other methods not yet timed):** 500 episodes ~ 1.6-1.7 h per run. 11 methods x 10 seeds = 110 runs ~ 180-190 h sequential on CPU (~7.7 days), before ablations and scalability (which add runs). With 5 seeds (protocol 2.0) it would have been ~ 90 h. Use `--episodes`, `--seeds` and GPU (`run.device`) to scale. These are extrapolations from short probes, not a timing of a full run.
**Statistical design (10 seeds):** paired tests on matched training seeds. A Holm-corrected exact Wilcoxon test is *attainable* at 10 seeds (minimum p = 2^-9 = 0.00195; times 10 comparisons = 0.0195 < 0.05), which it was not at 5 seeds. **Attainable is not guaranteed:** it requires every paired difference to have the same sign, and says nothing about power for realistic effect sizes. Report effect sizes and CIs of the mean difference, never p-values alone; with nearly identical policies the differences can be constant/degenerate and the tests return NaN rather than a fabricated p.
**Retry/back-off (protocol 2.1):** after a REJECTED/EXPIRED update the client waits min(480 s, 120 s x 2^(k-1)) before the next upload (k = consecutive failures; reset by a commit or a newer global model). It applies identically to all five federated methods.
**Pilot (not run):** `python scripts/run_pilot_2seeds.py` prints the plan (dry run); `--approved` is required to train.
Use `--episodes`, `--seeds` and GPU (`run.device`) to scale.
