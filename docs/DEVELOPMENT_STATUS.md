# Development Status

| Phase | Status |
|---|---|
| 1 Repository inspection | **NOT DONE** - `~/dt_smarl_step8/` absent from the build workspace. Models re-implemented from the two papers. |
| 2 Mathematical specification | Done: 63-entry registry, audit + map generated, discrepancies documented. |
| 3 Environment/physical models | Done + tested. |
| 4 MATD3 | Done + tested (twin target, smoothing, delay, Polyak). |
| 5 Federated learning | Done + tested (async protocol, scheduler, aggregation, staleness, clipping). |
| 6 Integration | Done: smoke test and integration tests pass. |
| 7 Baselines/ablations/metrics | Implemented and each executed at tiny scale (tests); **not run at publication scale**. |
| 8 Reproducibility/reporting | Scripts for tables, paired tests, PDF figures exist; **no full results generated**. |
| 9 Final audit | Partial (see ASSUMPTIONS_AND_LIMITATIONS.md). |

**Next action:** (1) attach/mount `dt_smarl_step8` and reconcile; (2) run `scripts/run_all_experiments.py --seeds 0 1 2 3 4` at full budget (GPU recommended); (3) calibrate energy and acoustic
constants; (4) run ablations, scalability, then `generate_figures.py`.


## Audit repair round (branch `audit/federated-lifecycle-fixes`)
Code-level repairs of the four audit findings (lifecycle, statistics accounting, relevance attribution, smoke configuration), with regression tests.
**No experiment at publication scale has been run; nothing here is evidence that S3A-FMATD3 outperforms any baseline. The project is NOT publication-ready.**

| Item | Status |
|---|---|
| Update lifecycle (one outstanding update, receipt != commit, exactly-once terminal outcome, reference advances only on commit) | Implemented; unit + protocol tests |
| Statistics accounting (update-owned stats, settle once, episode ids) | Implemented; tests |
| Batch-level `rho_sem` attribution with maturity and censoring | Implemented; tests; maturity rule is a modelling choice awaiting approval |
| Smoke config hysteresis (`offload_hi 0.04 > offload_lo 0.01`, `contact_time_s = 5 = max_hover_steps*dt`), `validate_config` | Implemented; tests |

Tests actually executed in this environment (CPU, Python 3.13): baseline before changes 200 passed; after the repair 227 passed (27 new test functions covering the 18 required regression cases); `scripts/run_smoke_test.py` passed all 13 checks (3 episodes, plumbing only, 5 updates created/aggregated, 0 rejected/expired);
`scripts/validate_manifest.py` OK for protocol 2.1 (10 seeds, 500 episodes, config hash pinned) and `--protocol 2.0`; protocol 2.1 preparation: see `docs/FINAL_SCIENTIFIC_AUDIT.md` (protocol 2.0 audit kept in `docs/FINAL_SCIENTIFIC_AUDIT_protocol2.0.md`). No linter/formatter is configured in the project (no ruff/flake8/black/mypy in `pyproject.toml`); `compileall` was run on src/scripts/tests.
Not executed: any full-scale training, ablation or scalability experiment.
