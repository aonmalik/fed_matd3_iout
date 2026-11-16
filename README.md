# fed_matd3_iout — S3A-FMATD3

Semantic- and Service-Aware **Federated MATD3** for cooperative multi-AUV data collection under intermittent underwater acoustic communication (Internet of Underwater Things).

> **Status: research code, no validated results yet.** Tests and a smoke test pass; publication-scale experiments have **not** been run, and a short pilot did not yet show learning
> (see `docs/ASSUMPTIONS_AND_LIMITATIONS.md`). Do not cite any performance claim from this repository until you have run the full protocol.

## Research objective
Can semantic- and service-aware asynchronous federated MATD3 improve delivered semantic utility, collection efficiency and robustness of multi-AUV IoUT data collection under
intermittent links, compared with conventional MATD3 and conventional federated learning? (Hypothesis under test, not a finding.)

## System
N sensors (finite FIFO buffers, Poisson traffic, packet-level semantic importance and VoI decay) - M AUVs - **one USV** = relay and federated server. AUVs learn local MATD3 policies,
upload actor deltas only when an acoustic contact allows, and receive the global actor on later contacts. See `docs/SYSTEM_ARCHITECTURE.md`, `docs/ALGORITHM.md`.

## Mathematical model
63 registered equations with validity class (A-E), units, assumptions, paper relation, code location, test: `docs/MATHEMATICAL_MODEL_AUDIT.md`, `docs/EQUATION_IMPLEMENTATION_MAP.md`
(generated from `scripts/equation_registry.py`; a test checks every mapping). Highlights: queue Eqs. Q1-Q8 (arrival before service, mass balance), exact Poisson backlog prediction vs
certainty-equivalent bound (P1-P4), semantic utility `d c sigma exp(-mu tau)` with criticality applied once (S2), proposed assignment cost (A1), staleness/relevance-weighted clipped
aggregation (F1-F8).

## Installation (Python >= 3.11)
```bash
cd fed_matd3_iout
pip install -r requirements.txt          # or: pip install -e ".[dev]"
```
CPU works out of the box; set `run.device` (`auto|cpu|cuda`) in `configs/default.yaml` or `--set run.device=cuda`. The selected device is printed and stored in every run manifest.

## Configuration
All settings live in `configs/*.yaml` (environment, acoustic channel, semantic model, allocation, MATD3, federation, reward). Override from any script:
`--config smoke.yaml` (extra file) and `--set env.n_auvs=5 federated.kappa=0.5` (dotted keys; unknown keys raise). Parameters not given in the papers are marked ASSUMPTION in the YAML.

## Commands (run from the project root)
```bash
python -m pytest tests                                                   # unit + integration tests
python scripts/run_smoke_test.py                                         # 3 AUV / 1 USV / 10 sensors, 3 episodes x 20 steps, CPU
python scripts/validate_manifest.py                                      # manifest / one-USV rule / ablation consistency
python scripts/train_centralized.py --method centralized_matd3 --seeds 0 # standard CTDE MATD3 (no federation)
python scripts/train_federated.py  --method fedavg_matd3       --seeds 0 # conventional federated weights
python scripts/train_federated.py  --method s3a_fedmatd3       --seeds 0 # proposed
python scripts/evaluate.py --run-dir results/<run_dir>                   # re-evaluate a saved run (new folder)
python scripts/run_all_experiments.py --seeds 0 1 2 3 4                  # 11 methods, matched evaluation, tables, paired tests
python scripts/run_ablation.py --seeds 0 1 2 3 4                         # 11 ablations of the proposed method
python scripts/run_scalability.py --seeds 0 1 2 3 4                      # (1,10) (3,20) (10,40) AUV/sensor, one USV
python scripts/generate_figures.py --results results/<set>/results.json  # vector PDF figures
python scripts/build_equation_docs.py                                    # regenerate equation docs from the registry
```
Append `--config smoke.yaml --episodes 2` to any experiment script for a quick plumbing check (not a scientific experiment).

## Baselines (all executable)
greedy_matd3, hungarian_matd3, centralized_matd3, independent_td3, maddpg, mappo (genuine PPO), fedavg_matd3, stale_fedmatd3, semantic_fedmatd3, service_fedmatd3, s3a_fedmatd3.
Comparability caveats: `docs/EXPERIMENT_PROTOCOL.md`. A goal-seeking reference controller (`baselines/reference_controller.py`) is a sanity ceiling, not a baseline.

## Ablations
full; no_semantic_utility; no_service_arrival_prediction; no_service_aware_aggregation; no_semantic_aware_aggregation; no_staleness_correction; no_async_communication; no_update_clipping;
no_assignment_continuity; certainty_equivalent_backlog; centralized_actor_training. Each flips only its own config keys.

## Metrics
Defined with explicit denominators in `evaluation/metrics.py` (collection/delivery ratios, U_total, U_norm, U_per_packet, critical PDR, age, VoI retention, energy components, energy per Mbit
and per utility, travel, unsafe events, boundary violations, workload imbalance, mass-balance error) and federated metrics from `FederatedProtocol.summary()` (rounds, accepted/rejected,
staleness, bytes, energy, latency, link-up fraction). Statistics: `evaluation/statistical_tests.py`.

## Reproducibility and outputs
See `docs/REPRODUCIBILITY.md`. Each run writes `run_manifest.json`, `train.jsonl`, checkpoints, aggregation/communication histories and `evaluation.json` into a new `results/` folder; nothing is overwritten.

## Limitations
See `docs/ASSUMPTIONS_AND_LIMITATIONS.md` (uncalibrated energy/acoustic constants, CTDE critics, no privacy mechanism, weighting changes the effective objective, no convergence proof).
