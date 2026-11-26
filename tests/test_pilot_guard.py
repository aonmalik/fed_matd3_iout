import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pilot_is_a_dry_run_without_explicit_approval(tmp_path):
    out = tmp_path / "pilot"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pilot_2seeds.py"), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0 and "DRY RUN: nothing was trained" in r.stdout
    assert "11 methods x seeds [0, 1]" in r.stdout and "5 episodes (protocol: 500)" in r.stdout
    assert not out.exists()                                                   # nothing written, nothing trained


def test_run_one_records_aggregation_diagnostics(smoke_cfg, tmp_path):
    from fed_matd3_iout.experiment import run_one
    r = run_one(smoke_cfg, "s3a_fedmatd3", 0, tmp_path, verbose=False, n_episodes=2)
    assert "aggregation_diagnostics" in r and (r["aggregation_diagnostics"] is None or "single_update_round_fraction" in r["aggregation_diagnostics"])
