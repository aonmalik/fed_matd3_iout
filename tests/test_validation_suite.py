"""Schema-3 test accounting: counts come from execution records; every failure mode is reported, never papered over."""
import json
import textwrap

from fed_matd3_iout.validation import suite as S

JUNIT = textwrap.dedent("""<?xml version="1.0"?><testsuites><testsuite name="pytest" errors="1" failures="2" skipped="3" tests="20">
</testsuite></testsuites>""")


def _main(counts, **kw):
    d = dict(total_collected=21, deselected=[S.SELF_REF], command="python -m pytest ...", returncode=0)
    d.update(kw)
    return S.build_main_suite(counts, **d)


def _ok_counts():
    return {"executed": 20, "passed": 20, "failed": 0, "errors": 0, "skipped": 0}


def test_count_aggregation_from_junit(tmp_path):
    p = tmp_path / "j.xml"
    p.write_text(JUNIT)
    c = S.parse_junit(p)
    assert c == {"executed": 20, "failed": 2, "errors": 1, "skipped": 3, "passed": 14}
    m = _main(c, returncode=1)
    assert m["total"] == 21 and m["executed"] == 20 and m["deselected_count"] == 1 and m["counts_consistent"]
    assert m["passed_all_executed"] is False


def test_multiple_testsuites_are_summed(tmp_path):
    p = tmp_path / "j.xml"
    p.write_text('<testsuites><testsuite tests="3" failures="0" errors="0" skipped="0"/><testsuite tests="4" failures="1" errors="0" skipped="0"/></testsuites>')
    assert S.parse_junit(p)["executed"] == 7 and S.parse_junit(p)["passed"] == 6


def test_clean_run_is_consistent_and_deselection_is_visible():
    m = _main(_ok_counts())
    assert m["passed_all_executed"] and m["counts_consistent"] and m["total"] == m["executed"] + m["deselected_count"] == 21
    o = S.build_overall(m, S.pending_self_check(), report_matches_source=True)
    assert o["any_test_deselected"] and not o["self_check_executed_separately"]
    assert o["all_required_checks_completed"] is False                      # self-check still outstanding -> not complete


def test_inconsistent_totals_are_flagged():
    assert _main(_ok_counts(), total_collected=25)["counts_consistent"] is False
    assert _main(_ok_counts(), total_collected=25)["passed_all_executed"] is False


def test_missing_test_log_means_no_counts_and_no_pass(tmp_path):
    assert S.parse_junit(tmp_path / "nope.xml") is None
    (tmp_path / "bad.xml").write_text("<not xml")
    assert S.parse_junit(tmp_path / "bad.xml") is None
    m = _main(None)
    assert m["counts_available"] is False and m["passed_all_executed"] is False and m["total"] is None
    assert _main(_ok_counts(), total_collected=None)["passed_all_executed"] is False


def test_nonzero_exit_code_is_never_a_pass():
    assert _main(_ok_counts(), returncode=1)["passed_all_executed"] is False
    assert _main(_ok_counts(), returncode=None)["passed_all_executed"] is False


def test_interrupted_validation_is_reported():
    m = _main(_ok_counts(), interrupted=True, returncode=2)
    assert m["interrupted"] and not m["passed_all_executed"]
    o = S.build_overall(m, S.pending_self_check(), report_matches_source=True, interrupted=True)
    assert o["validation_interrupted"] and not o["all_required_checks_completed"] and not o["main_suite_completed"]


def test_zero_passed_is_not_a_pass():
    z = {"executed": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    assert _main(z, total_collected=1)["passed_all_executed"] is False


def test_separate_self_check_reporting_and_overall(tmp_path):
    m = _main(_ok_counts())
    rep = tmp_path / "r.json"
    rep.write_text(json.dumps({"provenance": {"commit": "abc"}, "tests": {"main_suite": m}}))
    rec = S.write_self_check_record(tmp_path / "sc.json", report_path=rep, exit_code=0, commit_checked="def", source_hash="h",
                                    report=json.loads(rep.read_text()), main=m)
    assert rec["self_check"]["executed_separately"] and rec["self_check"]["passed"] and rec["self_check"]["commit_checked"] == "def"
    assert rec["source_commit_validated_by_report"] == "abc" and rec["commit_checked_by_self_check"] == "def"     # two different commits, no circularity
    assert rec["overall"]["all_required_checks_completed"] is True and rec["overall"]["any_test_deselected"] is True
    bad = S.write_self_check_record(tmp_path / "sc2.json", report_path=rep, exit_code=1, commit_checked="def", source_hash="h",
                                    report=json.loads(rep.read_text()), main=m)
    assert bad["self_check"]["passed"] is False and bad["overall"]["all_required_checks_completed"] is False


def test_collect_total_counts_real_tests(tmp_path):
    import sys
    (tmp_path / "test_a.py").write_text("def test_1(): pass\ndef test_2(): pass\n")
    (tmp_path / "test_b.py").write_text("def test_3(): pass\n")
    assert S.collect_total(tmp_path, sys.executable) == 3


def test_both_report_checks_are_listed_as_separately_run_self_checks():
    assert S.SELF_REF in S.SELF_CHECKS and len(S.SELF_CHECKS) == 2 and S.pending_self_check()["command"].count("::") == 2


def test_pending_self_check_never_claims_a_result():
    p = S.pending_self_check()
    assert p["executed_separately"] is False and p["passed"] is None and p["exit_code"] is None


def _rep(**kw):
    r = {"config_hash": "c", "provenance": {"working_tree_clean": True, "source_tree_hash": "s", "commit": "A"}}
    r["provenance"].update(kw)
    return r


def test_commit_report_mismatch_and_stale_hashes_are_detected():
    anc = lambda c, h: c == "A"
    assert S.verify_report_against_tree(_rep(), current_source_hash="s", current_config_hash="c", head_commit="B", is_ancestor=anc) == []
    assert any("commit" in p for p in S.verify_report_against_tree(_rep(commit="Z"), current_source_hash="s", current_config_hash="c", head_commit="B", is_ancestor=anc))
    assert any("source-tree" in p for p in S.verify_report_against_tree(_rep(), current_source_hash="OTHER", current_config_hash="c", head_commit="B", is_ancestor=anc))
    assert any("config" in p for p in S.verify_report_against_tree(_rep(), current_source_hash="s", current_config_hash="x", head_commit="B", is_ancestor=anc))
    assert any("dirty" in p for p in S.verify_report_against_tree(_rep(working_tree_clean=False), current_source_hash="s", current_config_hash="c", head_commit="B", is_ancestor=anc))
