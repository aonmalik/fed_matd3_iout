# Reproducibility

* Six named random streams (`layout, arrivals, fading, comm, policy_noise, eval`) derived by `SeedSequence([seed, stream, offset])` (`utils/seed.py`); changing one stream does not change another (tested).
* Every run directory (`results/<tag>_s<seed>_<timestamp>/`, never reused) holds `run_manifest.json` (full config, config hash, software versions, git commit if any, device),
  `train.jsonl`, `aggregation.jsonl`, `checkpoints/*.pt` (local + global actors, critics, versions, curve), `aggregation_history_*.json`, `communication_history_*.json`, `evaluation.json`.
* Checkpoints and run directories are never overwritten (unique-name logic, tested); JSON written atomically.
* Same seed => identical training curve, global model and communication bytes on CPU (tested). GPU runs may differ by non-deterministic kernels.
* Re-run a saved configuration: `python scripts/evaluate.py --run-dir results/<run>` (writes a new `reeval_*` folder).
* Preserved but separate: the original Paper-2 manifest is only referenced, never modified.
