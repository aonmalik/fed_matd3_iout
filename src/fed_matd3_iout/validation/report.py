"""Machine-readable final-validation report (JSON). Every number in it is produced by executing the validators below."""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy
import torch
import yaml

from ..baselines.registry import ABLATIONS, METHODS
from ..evaluation.report_generator import statistical_design_report
from ..utils.config import config_hash
from ..utils.manifest import validate_manifest
from . import aggregation_diagnostics, fairness, lifecycle_scenarios, matd3_checks, relevance_checks, statistics_checks

SCHEMA = "fed_matd3_iout.final_validation/2"


def _git(root: Path, *args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


HASHED_DIRS = ("src", "scripts", "configs", "tests")


def source_tree_hash(root: Path) -> str:
    """sha256 over (path, content) of every source/config/test file (docs and generated reports excluded): independent of git state."""
    import hashlib
    h = hashlib.sha256()
    for d in HASHED_DIRS:
        for p in sorted((root / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".yaml", ".yml", ".json", ".toml"):
                h.update(str(p.relative_to(root)).encode()); h.update(b"\0"); h.update(p.read_bytes()); h.update(b"\0")
    return h.hexdigest()[:16]


def _protocol_table(root: Path) -> list[dict]:
    """Every frozen protocol: manifest hash vs hash recomputed from that protocol's own YAML files + validator verdict."""
    from ..utils.config import load_config
    abl = lambda d: yaml.safe_load(open(d / "ablations.yaml"))["ablations"]
    rows = []
    cur = root / "configs"
    for label, d, cfg_dir in [("current", cur, None)] + [(p.name, p, p) for p in sorted((cur / "protocols").glob("v*"))]:
        m = json.load(open(d / "experiment_manifest.json"))
        c = load_config(cfg_dir=cfg_dir) if cfg_dir else load_config()
        errs = validate_manifest(m, c, METHODS, ABLATIONS, abl(d))
        rows.append({"label": label, "protocol_version": m["protocol_version"], "manifest_config_hash": m["config_hash"],
                     "recomputed_config_hash": config_hash(c), "hash_reproduced": m["config_hash"] == config_hash(c),
                     "n_train_seeds": m["n_train_seeds"], "train_episodes": m["train_episodes"], "manifest_errors": errs,
                     "yaml_file_sha256_16": {f.name: __import__("hashlib").sha256(f.read_bytes()).hexdigest()[:16]
                                              for f in sorted(d.glob("*.yaml"))}})
    return rows


def _all_checks_pass(d: dict) -> bool:
    return all(v for v in d.values() if isinstance(v, bool))


def build_report(cfg, root: Path, probe_episodes: int = 0, sensitivity_scenarios=(0, 1, 2), tests_summary: dict | None = None,
                 reference_scenarios=(0, 1, 2)) -> dict:
    manifest = json.load(open(root / "configs" / "experiment_manifest.json"))
    abl_yaml = yaml.safe_load(open(root / "configs" / "ablations.yaml"))["ablations"]
    errs = validate_manifest(manifest, cfg, METHODS, ABLATIONS, abl_yaml)
    protos = _protocol_table(root)
    sec = {}
    sec["reproducibility"] = {"kind": "implementation_check",
                              "passed": not errs and all(p["hash_reproduced"] and not p["manifest_errors"] for p in protos),
                              "errors": errs, "protocol_version": manifest["protocol_version"],
                              "config_hash": config_hash(cfg), "train_episodes": int(cfg.experiment.train_episodes),
                              "n_train_seeds": int(cfg.experiment.n_train_seeds), "eval_scenarios": int(cfg.experiment.eval_scenarios),
                              "protocols": protos,
                              "decision": "train_episodes=500 (Paper 2 Table I); 200 had no source -- see protocol_changelog"}
    lc = lifecycle_scenarios.run_all(cfg)
    sec["lifecycle"] = {"kind": "implementation_check", "passed": all(v["passed"] for v in lc.values()),
                        "scenarios": {k: {"passed": v["passed"], "n_checks": len(v["checks"]),
                                          "failed": [c for c, ok in v["checks"].items() if not ok], "details": v["details"]}
                                      for k, v in lc.items()},
                        "retry_backoff_impact": lifecycle_scenarios.retry_backoff_impact(cfg)}
    ex = relevance_checks.auv1_example()
    mb = relevance_checks.real_env_mass_balance(cfg, scenarios=(0, 1, 2))
    ws = relevance_checks.weight_stability(cfg.federated)
    ms = relevance_checks.maturity_sensitivity(cfg, scenarios=sensitivity_scenarios)
    rho = [r["pooled_rho_sem"] for r in ms["rows"]]
    dyn = {p: relevance_checks.dynamic_range_reference_policy(cfg, reference_scenarios, 100, p) for p in ("goal_seeking", "zero")}
    for d in dyn.values():
        d.pop("windows", None)
    sec["relevance"] = {"kind": "implementation_check",
                        "passed": bool(ex["AUV1_credited_with_B1_only"] and ex["AUV0_received_none_of_B1"] and mb["passed"]
                                       and ws["bounded"] and ws["weights_valid_everywhere"]),
                        "auv1_example": ex, "mass_balance_real_env": mb, "weight_stability": ws,
                        "maturity_sensitivity": ms,
                        "maturity_default_chosen_a_priori": float(cfg.federated.maturity_eps),
                        "maturity_pooled_rho_range": [float(np.nanmin(rho)), float(np.nanmax(rho))],
                        "dynamic_range_reference_policies": {
                            "kind": "measurement (descriptive, no learning)", **dyn}}
    mt = matd3_checks.run_all(cfg)
    sec["matd3_and_federation"] = {"kind": "implementation_check", "passed": all(v["passed"] for v in mt.values()), "checks": mt}
    fa = fairness.run_audit(cfg)
    sec["baseline_fairness"] = {"kind": "implementation_check", "passed": all(fa["checks"].values()), "checks": fa["checks"],
                                "methods": fa["rows"], "non_equivalent": fa["non_equivalent"],
                                "comparability_matrix": fa["comparability_matrix"], "n_pairs_like_for_like": fa["n_pairs_like_for_like"],
                                "n_pairs_total": fa["n_pairs_total"], "statement": fa["statement"],
                                "gradient_budget_ratio_matd3_to_mappo_minibatches": fa["gradient_budget_ratio_matd3_to_mappo_minibatches"]}
    st = statistics_checks.run_all(int(cfg.experiment.n_train_seeds), len(METHODS) - 1)
    sec["statistics"] = {"kind": "implementation_check", "passed": st["passed"], "checks": st["checks"], "design": st["design"],
                         "design_adequacy": st["design_adequacy"],
                         "tests": "paired t (+CI of the difference, Cohen d_z, Hedges g) and exact/approx Wilcoxon on matched training seeds; "
                                  "Holm-Bonferroni per metric and per test; degenerate (constant) differences give NaN p-values"}
    if probe_episodes:
        sec["training_probe"] = lifecycle_scenarios.training_probe(cfg, probe_episodes, with_diagnostics=True)
        sec["training_probe"].update(kind="measurement", note="Measurement of cost/behaviour at main-protocol episode length for a handful "
                                     "of episodes; NOT a performance result and NOT a comparison.")
    st_git = _git(root, "status", "--porcelain")
    prov = {"commit": _git(root, "rev-parse", "HEAD"), "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "working_tree_clean": st_git == "" and bool(_git(root, "rev-parse", "HEAD")),
            "dirty_files": [l for l in st_git.splitlines()][:50], "source_tree_hash": source_tree_hash(root),
            "note": "commit = HEAD when the report was generated; a report cannot contain the hash of the commit that stores it, so it is "
                    "committed separately AFTER the code commit it describes; source_tree_hash verifies the code is unchanged."}
    checks_ok = all(s["passed"] for s in sec.values() if s.get("kind") == "implementation_check")
    ran = tests_summary is not None
    rep = {"schema": SCHEMA, "protocol_version": manifest["protocol_version"], "config_hash": config_hash(cfg), "provenance": prov,
           "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                           "scipy": scipy.__version__, "device": "cpu"},
           "tests": tests_summary if ran else {"status": "not_run", "note": "this invocation did not run the full test suite; no test result is claimed"},
           "sections": sec,
           "evidence": {"implementation_checks_passed": bool(checks_ok),
                        "full_test_suite_executed": bool(ran), "full_test_suite_passed": (tests_summary.get("returncode") == 0 and tests_summary.get("failed", 1) == 0
                                                      and tests_summary.get("errors", 1) == 0 and tests_summary.get("passed", 0) > 0) if ran else None,
                        "empirical_performance_evidence": "none: no comparative experiment has been run",
                        "scientific_claims_supported": [],
                        "meaning": "'implementation_check' sections show that code does what the protocol says; they are not evidence that any "
                                   "method performs better. 'measurement' sections are descriptive."}}
    rep["all_validators_passed"] = bool(checks_ok)           # implementation checks only (kept for compatibility)
    return rep
