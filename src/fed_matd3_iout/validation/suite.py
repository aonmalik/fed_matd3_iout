"""Unambiguous machine-readable test-suite accounting (validation schema 3).

Three separate sections:
  A  main_suite        - the full pytest run (total/executed/passed/failed/errors/skipped/deselected, exact command, exit code)
  B  committed_report_self_check - the one test that validates the committed report; it cannot run before the report exists, so it is run
                         separately (stage 2) against the committed report and recorded in docs/final_validation_selfcheck.json
  C  overall           - derived from A and B; never hand-written.
All numbers come from the JUnit file / collection output of an actual run, never from constants."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

SUITE_SCHEMA = "fed_matd3_iout.validation_suite/1"
SELFCHECK_SCHEMA = "fed_matd3_iout.validation_selfcheck/1"
SELF_REF = "tests/test_final_validation_report.py::test_committed_report_matches_code_configuration_and_commit"
SELF_REF_SCHEMA = "tests/test_final_validation_report.py::test_committed_report_has_unambiguous_schema3_sections_and_matches_tree"
SELF_CHECKS = (SELF_REF, SELF_REF_SCHEMA)        # both validate the COMMITTED report, so both are deselected in stage 1 and run in stage 2


def parse_junit(path) -> dict | None:
    """Counts of an executed run from a JUnit xml file; None when the file is missing or unreadable (missing test log)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        root = ET.parse(p).getroot()
    except ET.ParseError:
        return None
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites:
        return None
    tot = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    return {"executed": tot["tests"], "failed": tot["failures"], "errors": tot["errors"], "skipped": tot["skipped"],
            "passed": tot["tests"] - tot["failures"] - tot["errors"] - tot["skipped"]}


def build_main_suite(counts: dict | None, *, total_collected: int | None, deselected: list[str], command: str, returncode: int | None,
                     interrupted: bool = False, failures: list[str] | None = None, wall_s: float | None = None) -> dict:
    """Section A.  ``total_collected`` counts every test of the suite INCLUDING deselected ones."""
    ok_counts = counts is not None and total_collected is not None
    out = {"command": command, "exit_code": returncode, "interrupted": bool(interrupted), "counts_available": bool(ok_counts),
           "deselected": list(deselected), "deselected_count": len(deselected), "failures": list(failures or []), "wall_s": wall_s}
    if ok_counts:
        out.update(total=int(total_collected), executed=counts["executed"], passed=counts["passed"], failed=counts["failed"],
                   errors=counts["errors"], skipped=counts["skipped"])
        out["counts_consistent"] = (out["executed"] + out["deselected_count"] == out["total"]
                                    and out["passed"] + out["failed"] + out["errors"] + out["skipped"] == out["executed"])
    else:
        out.update(total=None, executed=None, passed=None, failed=None, errors=None, skipped=None, counts_consistent=False)
    out["passed_all_executed"] = bool(ok_counts and out["counts_consistent"] and returncode == 0 and not interrupted and counts["failed"] == 0
                                      and counts["errors"] == 0 and counts["passed"] > 0)
    return out


def sha256_file(path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def pending_self_check() -> dict:
    """Section B as written at report-generation time (stage 1): honestly 'not executed here'."""
    return {"executed_separately": False, "status": "deferred_to_stage_2", "command": "python -m pytest -q " + " ".join(SELF_CHECKS), "exit_code": None,
            "passed": None, "commit_checked": None,
            "record_file": "docs/final_validation_selfcheck.json",
            "note": "the self-check validates the COMMITTED report, so it can only run after the report is committed (stage 2); its result "
                    "is stored in the record file, not here, to avoid a report that must contain the hash of its own commit."}


def build_overall(main: dict | None, self_check: dict, *, report_matches_source: bool | None, interrupted: bool = False) -> dict:
    """Section C, derived."""
    main_ok = bool(main and main.get("passed_all_executed"))
    sc_done = bool(self_check.get("executed_separately"))
    sc_pass = bool(sc_done and self_check.get("passed") is True and self_check.get("exit_code") == 0)
    return {"main_suite_completed": bool(main and main.get("counts_available") and not main.get("interrupted")),
            "main_suite_passed": main_ok,
            "any_test_deselected": bool(main and main.get("deselected_count", 0) > 0),
            "self_check_executed_separately": sc_done, "self_check_passed": sc_pass,
            "report_matches_committed_source_and_config": report_matches_source,
            "validation_interrupted": bool(interrupted or (main or {}).get("interrupted")),
            "all_required_checks_completed": bool(main_ok and sc_pass and report_matches_source is True and not interrupted),
            "meaning": "all_required_checks_completed is true only when the main suite passed, the deselected self-check was run separately "
                       "and passed on the committed report, the report matches the source/config, and nothing was interrupted."}


def collect_total(root: Path, python: str) -> int | None:
    r = subprocess.run([python, "-m", "pytest", "--collect-only", "-q", "-p", "no:warnings", "-p", "no:cacheprovider"], cwd=root,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    lines = r.stdout.splitlines()
    n_id = sum(1 for l in lines if "::" in l and not l.startswith(" "))
    n_per_file = sum(int(m.group(1)) for l in lines if (m := re.match(r"^\S+\.py: (\d+)$", l)))      # output shape under double -q
    return n_id or n_per_file or None


def verify_report_against_tree(report: dict, *, current_source_hash: str, current_config_hash: str, head_commit: str,
                               is_ancestor) -> list[str]:
    """Problems linking a committed report to the checked-out tree (empty list = consistent).  ``is_ancestor(commit, head)`` -> bool."""
    problems = []
    prov = report.get("provenance", {})
    if prov.get("working_tree_clean") is not True:
        problems.append("report was generated from a dirty tree")
    if report.get("config_hash") != current_config_hash:
        problems.append("config hash mismatch (stale report)")
    if prov.get("source_tree_hash") != current_source_hash:
        problems.append("source-tree hash mismatch (stale report)")
    c = prov.get("commit")
    if not c or not is_ancestor(c, head_commit):
        problems.append("recorded commit is not an ancestor of HEAD (commit/report mismatch)")
    return problems


def write_self_check_record(path, *, report_path, exit_code: int, commit_checked: str, source_hash: str, report: dict, main: dict | None) -> dict:
    sc = {"executed_separately": True, "command": "python -m pytest -q " + " ".join(SELF_CHECKS), "exit_code": int(exit_code), "passed": exit_code == 0,
          "commit_checked": commit_checked}
    rec = {"schema": SELFCHECK_SCHEMA, "committed_report": str(Path(report_path).name), "committed_report_sha256": sha256_file(report_path),
           "source_commit_validated_by_report": report.get("provenance", {}).get("commit"),
           "commit_checked_by_self_check": commit_checked, "source_tree_hash_at_check": source_hash, "self_check": sc,
           "overall": build_overall(main, sc, report_matches_source=exit_code == 0)}
    Path(path).write_text(json.dumps(rec, indent=2))
    return rec
