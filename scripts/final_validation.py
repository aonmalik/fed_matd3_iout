"""Run every final-validation check and write the machine-readable report (docs/final_validation_report.json by default).
Does NOT start any publication-scale training or pilot.  --probe-episodes N adds a short default-scale cost probe (N episodes, one run)."""
from _common import *
import argparse, json, re, subprocess, time
from fed_matd3_iout.validation.report import build_report

p = argparse.ArgumentParser()
p.add_argument("--out", default=str(ROOT / "docs" / "final_validation_report.json"))
p.add_argument("--probe-episodes", type=int, default=0)
p.add_argument("--run-tests", action="store_true", help="also run the full pytest suite and record its summary")
a = p.parse_args()
tests = None
SELF_REF = "tests/test_final_validation_report.py::test_committed_report_matches_code_configuration_and_commit"
if a.run_tests:
    t0 = time.time()
    import tempfile, xml.etree.ElementTree as ET
    junit = Path(tempfile.mkdtemp()) / "junit.xml"
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "-rfE", f"--junitxml={junit}", "--deselect", SELF_REF]   # full suite, no -x
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    m = {}
    if junit.exists():                                   # counts come from the JUnit file, never from parsing console text
        ts = ET.parse(junit).getroot()
        ts = ts if ts.tag == "testsuite" else ts.find("testsuite")
        n, nf, ne, ns = (int(ts.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
        m = {"collected": n, "passed": n - nf - ne - ns, "failed": nf, "errors": ne, "skipped": ns}
    tests = {"command": " ".join(["python -m pytest -q -p no:warnings -rfE --junitxml=<tmp>", "--deselect", SELF_REF]), "deselected": [SELF_REF],
             "deselected_reason": "this test validates the report that is being generated here (it is run separately against the committed report)",
             "returncode": r.returncode, "failures": [l for l in r.stdout.splitlines() if l.startswith(("FAILED", "ERROR"))][:30], **m,
             "counts_available": bool(m), "wall_s": round(time.time() - t0, 1)}
rep = build_report(load_config(), ROOT, a.probe_episodes, tests_summary=tests)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(rep, open(a.out, "w"), indent=2, default=float)
print(json.dumps({k: v.get("passed", "n/a") for k, v in rep["sections"].items()}, indent=1))
print("implementation_checks_passed:", rep["evidence"]["implementation_checks_passed"], "| full_test_suite:",
      rep["evidence"]["full_test_suite_executed"] and rep["evidence"]["full_test_suite_passed"], "| tree clean:", rep["provenance"]["working_tree_clean"],
      "| empirical performance evidence: NONE ->", a.out)
if tests is not None and not tests["counts_available"]:
    print("ERROR: the test run produced no JUnit counts; the report records full_test_suite_passed = False")
ok = rep["evidence"]["implementation_checks_passed"] and (tests is None or rep["evidence"]["full_test_suite_passed"])
sys.exit(0 if ok else 1)
