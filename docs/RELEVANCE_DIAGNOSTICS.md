# Relevance diagnostics (diagnostics-only; Protocol 2.2 unchanged)

Added after the five-seed pilot showed relevance inactive in about 96% of rounds. **Nothing here changes learning**: `rho_sem`, `rho_svc`, `M`, the S3A formula, eps_M, aggregation weights, clipping, update cadence, maturity thresholds, attribution, settlement, communication and configuration are untouched (config hash / manifest unchanged).

## What is measured
**Relevance status per client update** — `federated/relevance.py::relevance_status(stats)`; logged in every aggregation round as `server.history[i]["relevance_status"]` (aligned with `clients`, `relevance`, `rho_sem`, `rho_svc`). Fields: `sem`, `svc`, `attribution_mode` (`semantic_utility` | `unit_based`), `rho_sem_defined`, `rho_svc_defined`, `svc_zero_with_evidence`, `coll_mbit`, `owned_loss_mbit`, `n_mature_batches`, `n_immature_batches`.

| `sem` | meaning |
|---|---|
| `mature_attribution_evidence` | rho_sem defined (mature batches with positive value in this claim). Neutral label: with `use_semantics=True` it is semantic-utility attribution (U_mature/V_mature); in the semantics-off ablation it is unit-based (units_del_mature/units_coll_mature) and carries no semantic information — see `attribution_mode` |
| `collection_no_mature_evidence` | rho_sem undefined; data collected in the window, or collected batches still open (waiting for delivery/maturity — not a failure) |
| `no_collection` | rho_sem undefined; nothing collected and no open batches |

| `svc` | meaning |
|---|---|
| `service_evidence` | rho_svc defined (C + L_own > 0); the value may be 0 (`svc_zero_with_evidence`) |
| `no_owned_service_flow` | rho_svc undefined (C + L_own = 0) |

Undefined is never reported as zero relevance. (As before, `M` itself is 0 when both components are undefined; the status shows which case it was.) `n_immature_batches` is the client's open-batch count at claim time and includes batches from earlier windows; stats without the key count as 0.

**Round summary** — `validation/aggregation_diagnostics.py::summarize_relevance_status`, included as `aggregation_diagnostics["relevance_status_summary"]` in each run's `evaluation.json`: status counts, joint counts, zero-with-evidence count, and multi-client rounds with every relevance zero / no component evidence / distinct relevance, plus `counts_consistent`. Runs from older code report `available: false`; so do rounds whose `clients` / `relevance` / `relevance_status` lengths differ (`reason: inconsistent_records`, with the offending round indices) — the summary never truncates or partially counts such records.

**Episode boundaries** — `MultiAUVIoUTEnv.episode_boundary_diagnostics()` (also saved as `episode_boundary_diagnostics` in `evaluation.json`): per reset, the partial window statistics that are discarded (`partial_window_discarded`, per-AUV collected / owned-loss Mbit) and the unclaimed batches that are censored (entries, V, units, delivered U so far); totals `dropped_partial_window`, `censored_V_total`; and the finite statistics refused because their episode ended (`stats_refused_episode_count`, `stats_refused_episode_mass`). This only records what the existing reset already discards.

**Mass balance** — `_ACC_KEYS` (communication protocol) now also tracks `units_del_mature`, `units_coll_mature`, `n_mature_batches`. They are additive window keys returned/merged exactly like the existing four; conservation (created = consumed + returned + discarded) is tested for rejected, expired and aggregated updates and for old-episode statistics. `steps` is not tracked.

## What this does NOT establish
- It does not make relevance active, informative or beneficial, and does not show that a changing weight improves learning.
- It does not diagnose *why* a policy collected nothing, nor whether trained policies (500-episode protocol) behave differently from the 5-episode pilot; that needs a later-checkpoint probe (not run).
- Discarded partial windows and censored batches are reported, not recovered or re-attributed. Late deliveries after claim-by-age remain unattributed (existing design limitation).
- The reference-policy measurements in `docs/final_validation_report.json` remain descriptive and were not regenerated.
