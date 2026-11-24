"""Validate a protocol manifest against the merged YAML configuration and the method/ablation registries.
Exit code 0 ONLY if there is no mismatch; every mismatch is critical (a publication-scale run must not start).
  (no args)        current protocol: configs/experiment_manifest.json vs configs/*.yaml
  --protocol 2.0   frozen historical protocol: configs/protocols/v2.0/ (proves protocol 2.0 is still reproducible)
  --print-hash     print the merged-config hash (used when deliberately freezing a new protocol version)"""
from _common import *
import json, yaml
from fed_matd3_iout.baselines.registry import ABLATIONS, METHODS
from fed_matd3_iout.utils.config import config_hash
from fed_matd3_iout.utils.manifest import validate_manifest

proto = sys.argv[sys.argv.index("--protocol") + 1] if "--protocol" in sys.argv else None
cdir = ROOT / "configs" / "protocols" / f"v{proto}" if proto else None
cfg = load_config(cfg_dir=cdir) if cdir else load_config()
if "--print-hash" in sys.argv:
    print(config_hash(cfg)); sys.exit(0)
base = cdir or (ROOT / "configs")
m = json.load(open(base / "experiment_manifest.json"))
y = yaml.safe_load(open(base / "ablations.yaml"))["ablations"]
errs = validate_manifest(m, cfg, METHODS, ABLATIONS, y)
if errs:
    print("MANIFEST INVALID:\n  " + "\n  ".join(errs))
    sys.exit(1)
print(f"MANIFEST OK (protocol {m['protocol_version']}, config_hash {m['config_hash']})")
