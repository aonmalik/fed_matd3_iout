"""run_pilot_2seeds.py must describe and target Protocol 2.2.  Only DRY RUNS are executed here (no --approved: nothing is trained)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pilot_2seeds.py"


def _dry(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=ROOT, timeout=300)


def test_default_output_directory_is_protocol22_and_dry_run_names_protocol_hash_and_directory():
    pilot = ROOT / "results" / "pilot_protocol22"
    existed = pilot.exists()
    r = _dry()
    assert r.returncode == 0, r.stdout + r.stderr
    man = json.load(open(ROOT / "configs" / "experiment_manifest.json"))
    out = r.stdout
    assert "protocol 2.2" in out and "selected protocol: 2.2" in out
    assert man["config_hash"] in out and "effective run config_hash" in out
    assert "output directory:" in out and "pilot_protocol22" in out
    assert "pilot_protocol21" not in out and "DRY RUN: nothing was trained" in out
    assert pilot.exists() == existed                                         # a dry run creates nothing


def test_protocol22_pilot_refuses_a_protocol21_named_output_directory():
    for bad in ("results/pilot_protocol21", "results/pilot_v2.1", "results/pilot_protocol_2.1"):
        r = _dry("--out", bad)
        assert r.returncode == 2 and "REFUSED" in r.stdout, (bad, r.stdout)
        assert not (ROOT / bad).exists()


def test_script_text_declares_protocol22_and_stamps_summary():
    src = SCRIPT.read_text()
    head = src.split('"""')[1]
    assert "Protocol 2.2" in head and "protocol 2.1 preparation" not in head.lower()
    assert 'default=str(ROOT / "results" / "pilot_protocol22")' in src
    assert '"manifest_config_hash"' in src and '"protocol_version": version' in src
    default_line = [l for l in src.splitlines() if "--out" in l][0]
    assert "protocol21" not in default_line                                  # no 2.1-named default
