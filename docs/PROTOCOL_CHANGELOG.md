# Protocol changelog

| Version | Status | Change |
|---|---|---|
| 1.0 | superseded | Initial build. Manifest `train_episodes: 200` had no source; `default.yaml` used 500. |
| 2.0 | **frozen, preserved** (`configs/protocols/v2.0/`, git tag `protocol-2.0`, `config_hash 7f4a0e4f66569528`, reproduce with `python scripts/validate_manifest.py --protocol 2.0`) | `train_episodes := 500` (Paper 2 Table I). 5 training seeds. Update lifecycle, statistics accounting and relevance attribution as repaired on `audit/federated-lifecycle-fixes`. No retry limiting. |
| 2.1 | **current, preparation only** (`configs/experiment_manifest.json`, `config_hash c76394bf7eaa9005`) | (1) `n_train_seeds` 5 -> 10 (seeds 0-9). (2) `federated.retry`: bounded deterministic exponential back-off (base 120 s, factor 2, cap 480 s) after REJECTED/EXPIRED updates; counters for retries, deferrals, rejected/expired bytes. (3) `install_global`: blend = 1 is an exact replacement (previously `1*theta + 0*local` turned a non-finite local parameter into NaN). Unchanged: aggregation formula, relevance definition, maturity_eps = 0.05, statistics settlement, all learners, environment. |

Rules: any change to any YAML value changes `config_hash`; `scripts/validate_manifest.py` exits 1 until the version is bumped, the changelog extended and the hash re-frozen. Code-only changes that keep the YAML (for example the diagnostics and the validation-report fixes of this phase) do not change the protocol; they are covered by `source_tree_hash` in the validation report instead. Protocol 2.0 behaviour is reproduced by loading `configs/protocols/v2.0` (retry key absent = no limiting); the install_global fix only differs from 2.0 when a local parameter is non-finite.
