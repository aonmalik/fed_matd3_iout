# Protocol 2.2 correction report (CANDIDATE — pilot-stage, not publication-ready)

## 1. Starting state and provenance
* Source: `95f84cd2-fed_matd3_iout_protocol21.zip` (Protocol 2.1 tree). It contained **no `.git` directory**: no starting commit, no branch.
* Backups (all in `~/fed_matd3_iout_protocol21_backup/`, outside the work tree): `pristine_protocol21/` + `.tar.gz` (sha256 `7b7c3de8bbf5e7484aa139098c741f4b094e660784178a03aaffc948534f8ab2`) = the delivered 2.1 tree; `pre_finalize_20261124T104544Z/` + `.tar.gz` (sha256 `3fc9ac6a801fa4d38a0bde83df59b2cca2093fe1e28a581f68c463f58b897800`) = the working tree before the finalisation session; `pre_finalize_sha256_inventory.txt` = SHA-256 of every .py/.yaml/.json/.md file at that moment; `START_STATE.txt`.
* **What came from where.** Tagged in the CHANGELOG: **[S]** = made by the session that wrote this report (Phase 2 communication arbitration, D7 tests, decision log, the documents, Git baseline). **[P]** = already in the working tree when that session's visible work began (MAPPO tanh, metric rename, drain, compute-energy, fixed-workload, relevance diagnostics, non-finite rejection, 2.2 manifest, frozen `v2.1` directory, D8–D10 tests). The [P] items were checked by diff against the pristine backup and by the test suite; they were **not** independently re-derived, and the history of how they were produced is not recorded. Treat them as needing independent review.
* Protocol 2.1 preservation: `configs/protocols/v2.1/` is byte-identical to the pristine tree's configs and to the pre-finalize backup (`cmp`/`sha256sum`); `docs/final_validation_report_protocol2.1.json` is byte-identical to the pristine `docs/final_validation_report.json`.

## 2. Decisions D7–D10 (proposals, NOT approved by the author; D1–D6 untouched and open)
Full text: `docs/PROTOCOL_2.2_DECISION_LOG.md`.
| ID | Selected (provisional) | Status |
|---|---|---|
| D7 | half-duplex; scope = ONE airtime budget per AUV USV-link modem; time-division; control first (≤ half pool while others wait), max-min fair rest | implemented, tested, provisional, requires approval |
| D8 | bounded drain `final_drain_s = 600 s` (derived server-side bound 521.1 s; no total bound exists) | implemented, tested, provisional, requires approval |
| D9 | joules only when measured, else `not_measured` (never 0) + compute time | implemented, tested, provisional, requires approval; **no instrument available ⇒ energy not measured** |
| D10 | separate `fixed_workload_completion_time_s`; backlog-clearance untouched | implemented, tested, provisional, requires approval |

## 3. Code changes in this session [S]
`communication/capacity_arbiter.py` and the airtime/energy changes in `model_transfer.py`, `communication_protocol.py`, `multi_auv_iout_env.py`, `fixtures.py` (first version; later refinements by [P] — half-pool control cap, sensor-airtime cap — were kept); new `tests/test_d7_half_duplex.py` (9 tests). One of my D7 tests initially failed because I had assumed control could take the whole pool; the code's documented behaviour (cap at half the pool while others wait) was correct and the **test** was corrected to match the specification, not the reverse. Docs: decision log, CHANGELOG, REMAINING_ISSUES, this report, README/DEVELOPMENT_STATUS notes.

## 4. Tests and validation (executed; commands in §8)
* Baseline before any 2.2 edit: 319 collected, 318 passed, 1 failed (`test_committed_report_matches_code_configuration_and_commit`: no Git repository).
* Pre-commit run of the 2.2 tree: **431 collected, 430 passed, 1 failed** (the same test: it needs a clean committed tree and a report generated from it).
* Post-commit results (after the local commits and the regenerated report): see §9, filled in from the executed commands.
* Smoke test (tiny CPU config): 13/13 checks pass; it demonstrates plumbing only.
* Lifecycle scenarios: executed inside `scripts/final_validation.py` (section `lifecycle`) and by `tests/test_lifecycle_scenarios.py`, `test_update_lifecycle.py`, `test_stats_accounting.py`, `test_retry_backoff.py`, `test_protocol22_audit.py`.
* Software verification ≠ scientific evidence: the validation report records `empirical_performance_evidence: none`.

## 5. Protocol hashes
2.2: `config_hash` from `python scripts/validate_manifest.py` (canonical tool; value in §9). 2.1: `c76394bf7eaa9005`. 2.0: `7f4a0e4f66569528`. All three validate.

## 6. Known limitations
See `docs/PROTOCOL_2.2_REMAINING_ISSUES.md` (physical-layer gaps, unmeasured compute energy, drain limits, relevance dynamic range with no tuning, MAPPO non-equivalence, smoke-scale evidence only, required pre-publication experiments).

## 7. Source commit and two-stage report procedure
A report cannot contain the hash of the commit that stores it. Procedure: (1) commit **A** = all source/config/tests/docs; (2) generate `docs/final_validation_report.json` from the clean tree at A (`scripts/final_validation.py --run-tests`): it records `commit = A`, `working_tree_clean = true` and `source_tree_hash` (source/config/tests only; docs and reports excluded); (3) commit **B** = the report only; later documentation-only commits do not invalidate it because the test checks that A is an ancestor of HEAD and that the source hash still matches. The commit hashes are in §9.

## 8. Commands
```
python -m compileall -q src scripts tests
python -m pytest -q
python scripts/run_smoke_test.py
python scripts/validate_manifest.py ; python scripts/validate_manifest.py --protocol 2.1 ; python scripts/validate_manifest.py --protocol 2.0
python scripts/final_validation.py --run-tests
```
Not run (by instruction): `scripts/run_pilot_2seeds.py`, any full-scale or pilot training, `git push`.

## 9. Executed results and commit identifiers
(Filled in below after execution.)
