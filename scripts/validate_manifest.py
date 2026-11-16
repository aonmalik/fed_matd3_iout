"""Validate configs/experiment_manifest.json against the architecture rules (exactly one USV, known methods, distinct seeds)."""
from _common import *
import json, yaml
from fed_matd3_iout.baselines.registry import ABLATIONS, METHODS

m = json.load(open(ROOT / "configs" / "experiment_manifest.json"))
errs = []
if m["architecture"]["n_usv"] != 1: errs.append("n_usv must be 1")
for x in m["methods"]:
    if x not in METHODS: errs.append(f"unknown method {x}")
missing = set(METHODS) - set(m["methods"])
if missing: errs.append(f"methods missing from manifest: {sorted(missing)}")
base = [m["seeds"][k] for k in ("layout", "arrivals", "fading", "comm")]
if len(set(base)) != 4: errs.append("stream seeds must be distinct")
if any(x.get("n_usv", 1) != 1 for x in m["scalability"]): errs.append("scalability entry with n_usv != 1")
if sorted(m["scalability"], key=lambda x: x["n_auvs"])[0]["n_auvs"] != 1: errs.append("scalability must include 1 AUV")
if not {1, 3, 10} <= {x["n_auvs"] for x in m["scalability"]}: errs.append("scalability must include 1, 3, 10 AUVs")
y = yaml.safe_load(open(ROOT / "configs" / "ablations.yaml"))["ablations"]
flat = {k: {kk: vv for kk, vv in v.items()} for k, v in y.items()}
if set(flat) != set(ABLATIONS): errs.append("ablations.yaml and registry disagree on names")
else:
    for k in flat:
        if flat[k] != ABLATIONS[k]: errs.append(f"ablation {k} differs between yaml and registry")
print("MANIFEST OK" if not errs else "MANIFEST INVALID:\n  " + "\n  ".join(errs))
sys.exit(1 if errs else 0)
