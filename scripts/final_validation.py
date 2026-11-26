"""Run every final-validation check and write the machine-readable report (docs/final_validation_report.json by default).
Does NOT start any publication-scale training or pilot.  --probe-episodes N adds a short default-scale cost probe (N episodes, one run)."""
from _common import *
import argparse, json, re, subprocess, time
from fed_matd3_iout.validation.report import build_report

p = argparse.ArgumentParser()
p.add_argument("--out", default=str(ROOT / "docs" / "final_validation_report.json"))
p.add_argument("--probe-episodes", type=int, default=0)
p.add_argument("--self-check", action="store_true", help="stage 2: run the deselected committed-report test against the committed report and record it in docs/final_validation_selfcheck.json")
p.add_argument("--run-tests", action="store_true", help="also run the full pytest suite and record its summary")
a = p.parse_args()
tests = None
from fed_matd3_iout.validation import suite as S
SELF_REF = S.SELF_REF
SELF_CHECKS = list(S.SELF_CHECKS)
if a.self_check:                                         # stage 2: run the deselected self-check against the committed report and record it
    import subprocess as sp
    r = sp.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *SELF_CHECKS], cwd=ROOT)
    rep0 = json.load(open(a.out))
    from fed_matd3_iout.validation.report import source_tree_hash
    rec = S.write_self_check_record(ROOT / "docs" / "final_validation_selfcheck.json", report_path=a.out, exit_code=r.returncode,
                                    commit_checked=sp.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                                    source_hash=source_tree_hash(ROOT), report=rep0, main=rep0.get("tests", {}).get("main_suite"))
    print(json.dumps(rec["overall"], indent=1)); sys.exit(r.returncode)
if a.run_tests:
    t0 = time.time()
    import tempfile
    junit = Path(tempfile.mkdtemp()) / "junit.xml"
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "-p", "no:cacheprovider", "-rfE", f"--junitxml={junit}", *[x for c in SELF_CHECKS for x in ("--deselect", c)]]   # full suite, no -x
    interrupted = False
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        rc = r.returncode
        out = r.stdout
    except KeyboardInterrupt:
        interrupted, rc, out = True, None, ""
    interrupted = interrupted or rc in (2, -2, 130)     # pytest exit 2 = interrupted
    main = S.build_main_suite(S.parse_junit(junit), total_collected=S.collect_total(ROOT, sys.executable), deselected=SELF_CHECKS,
                              command="python -m pytest -q -p no:warnings -p no:cacheprovider -rfE --junitxml=<tmp> " + " ".join("--deselect " + c for c in SELF_CHECKS),
                              returncode=rc, interrupted=interrupted, failures=[l for l in out.splitlines() if l.startswith(("FAILED", "ERROR"))][:30],
                              wall_s=round(time.time() - t0, 1))
    # legacy flat keys kept (collected/passed/failed/errors/skipped/returncode) so older readers still work; the sections below are authoritative
    tests = {"main_suite": main, "committed_report_self_check": S.pending_self_check(),
             "overall": S.build_overall(main, S.pending_self_check(), report_matches_source=None, interrupted=interrupted),
             "command": main["command"], "deselected": main["deselected"], "returncode": rc, "failures": main["failures"],
             "collected": main["executed"], "passed": main["passed"], "failed": main["failed"], "errors": main["errors"],
             "skipped": main["skipped"], "counts_available": main["counts_available"], "wall_s": main["wall_s"]}
rep = build_report(load_config(), ROOT, a.probe_episodes, tests_summary=tests)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(rep, open(a.out, "w"), indent=2, default=float)
print(json.dumps({k: v.get("passed", "n/a") for k, v in rep["sections"].items()}, indent=1))
print("implementation_checks_passed:", rep["evidence"]["implementation_checks_passed"], "| full_test_suite:",
      rep["evidence"]["full_test_suite_executed"] and rep["evidence"]["full_test_suite_passed"], "| tree clean:", rep["provenance"]["working_tree_clean"],
      "| empirical performance evidence: NONE ->", a.out)
if tests is not None and not tests["counts_available"]:
    print("ERROR: the test run produced no usable counts; the report records full_test_suite_passed = False")
ok = rep["evidence"]["implementation_checks_passed"] and (tests is None or rep["evidence"]["full_test_suite_passed"])
sys.exit(0 if ok else 1)
