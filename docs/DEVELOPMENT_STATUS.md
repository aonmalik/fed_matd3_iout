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
