"""The machine-readable validation report must be honest (what was executed, on which commit) and must never go stale."""
import json
import subprocess
from pathlib import Path

from fed_matd3_iout.utils.config import config_hash
from fed_matd3_iout.validation.report import SCHEMA, build_report, source_tree_hash

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "final_validation_report.json"


def test_report_without_test_run_claims_no_test_result_and_no_performance_evidence(cfg):
    rep = build_report(cfg, ROOT, probe_episodes=0, sensitivity_scenarios=(0,), reference_scenarios=(0,))
    assert rep["schema"] == SCHEMA
    assert rep["tests"]["status"] == "not_run" and "passed" not in rep["tests"]            # no invented test result
    ev = rep["evidence"]
    assert ev["full_test_suite_executed"] is False and ev["full_test_suite_passed"] is None
    assert ev["implementation_checks_passed"] is True and ev["scientific_claims_supported"] == []
    assert ev["empirical_performance_evidence"].startswith("none")
    assert all(s["kind"] in ("implementation_check", "measurement") for s in rep["sections"].values())
    json.dumps(rep, default=float)


def test_test_result_is_recorded_as_passed_only_with_real_counts(cfg):
    kw = dict(probe_episodes=0, sensitivity_scenarios=(0,), reference_scenarios=(0,))
    good = {"returncode": 0, "passed": 300, "failed": 0, "errors": 0}
    assert build_report(cfg, ROOT, tests_summary=good, **kw)["evidence"]["full_test_suite_passed"] is True
    for bad in ({"returncode": 0, "passed": 0, "failed": 0, "errors": 0}, {"returncode": 1, "passed": 300, "failed": 1, "errors": 0},
                {"returncode": 0, "passed": 300}, {"returncode": 0, "passed": 300, "failed": 0, "errors": 1}):
        assert build_report(cfg, ROOT, tests_summary=bad, **kw)["evidence"]["full_test_suite_passed"] is False


def test_statistics_section_is_computed_not_hard_coded(cfg):
    rep = build_report(cfg, ROOT, probe_episodes=0, sensitivity_scenarios=(0,), reference_scenarios=(0,))
    st = rep["sections"]["statistics"]
    assert len(st["checks"]) >= 10 and st["passed"] == all(st["checks"].values())
    assert st["design_adequacy"]["kind"] == "attainability_only" and "guarantee" in st["design_adequacy"]["statement"]
    assert st["design"]["n_seeds"] == int(cfg.experiment.n_train_seeds) == 10


def test_report_lists_every_frozen_protocol_with_a_reproduced_hash(cfg):
    rep = build_report(cfg, ROOT, probe_episodes=0, sensitivity_scenarios=(0,), reference_scenarios=(0,))
    table = {p["protocol_version"]: p for p in rep["sections"]["reproducibility"]["protocols"]}
    assert {"2.0", "2.1"} <= set(table)
    assert table["2.0"]["manifest_config_hash"] == "7f4a0e4f66569528" and table["2.0"]["n_train_seeds"] == 5
    assert table["2.1"]["n_train_seeds"] == 10
    assert all(p["hash_reproduced"] and not p["manifest_errors"] for p in table.values())


def test_committed_report_matches_code_configuration_and_commit(cfg):
    assert REPORT.exists(), "run: python scripts/final_validation.py --run-tests --probe-episodes 3"
    rep = json.load(open(REPORT))
    man = json.load(open(ROOT / "configs" / "experiment_manifest.json"))
    assert rep["schema"] == SCHEMA and rep["evidence"]["implementation_checks_passed"] is True
    assert rep["config_hash"] == man["config_hash"] == config_hash(cfg), "report is stale: a YAML value changed after it was generated"
    assert rep["protocol_version"] == man["protocol_version"] == str(cfg.experiment.protocol_version)
    assert rep["tests"].get("returncode") == 0 and rep["tests"].get("failed", 0) == 0 and rep["tests"]["passed"] > 0
    assert rep["evidence"]["full_test_suite_executed"] and rep["evidence"]["full_test_suite_passed"]
    prov = rep["provenance"]
    assert prov["working_tree_clean"] is True, "report must be generated from a committed (clean) tree"
    assert prov["source_tree_hash"] == source_tree_hash(ROOT), "code changed after the report was generated"
    r = subprocess.run(["git", "merge-base", "--is-ancestor", prov["commit"], "HEAD"], cwd=ROOT, capture_output=True)
    assert r.returncode == 0, "recorded commit is not an ancestor of HEAD"


def test_committed_report_has_unambiguous_schema3_sections_and_matches_tree(cfg):
    """Schema 3: main suite / self-check / overall are separate; counts are consistent; the report matches the checked-out source."""
    from fed_matd3_iout.validation import suite as S
    rep = json.load(open(REPORT))
    t = rep["tests"]
    m = t["main_suite"]
    assert m["counts_available"] and m["counts_consistent"] and m["total"] == m["executed"] + m["deselected_count"]
    assert m["executed"] == m["passed"] + m["failed"] + m["errors"] + m["skipped"] and m["exit_code"] == 0 and m["passed_all_executed"]
    assert m["deselected"] == list(S.SELF_CHECKS) and "--deselect" in m["command"]
    assert t["committed_report_self_check"]["executed_separately"] is False and t["committed_report_self_check"]["record_file"].endswith("selfcheck.json")
    prov = rep["provenance"]
    problems = S.verify_report_against_tree(rep, current_source_hash=source_tree_hash(ROOT), current_config_hash=config_hash(cfg),
                                            head_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                                            is_ancestor=lambda c, h: subprocess.run(["git", "merge-base", "--is-ancestor", c, h], cwd=ROOT).returncode == 0)
    assert problems == [], problems
