import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fed_matd3_iout.utils.config import load_config  # noqa: E402


def parse_overrides(items):
    import yaml
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def base_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", nargs="*", default=[], help="extra YAML files (e.g. smoke.yaml)")
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides key=value")
    p.add_argument("--seeds", type=int, nargs="*", default=[0])
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--out", default=str(ROOT / "results"))
    return p


def cfg_from(args):
    return load_config(args.config, parse_overrides(args.set))


def save_results(results: dict, out_root, name: str):
    """results: {tag: {seed: [scenario metric dicts]}} -> new directory (never overwrites) with results.json/tables."""
    from fed_matd3_iout.utils.reproducibility import new_run_dir, atomic_write_json
    from fed_matd3_iout.evaluation import report_generator as rg
    d = new_run_dir(out_root, name)
    atomic_write_json(d / "results.json", results)
    rows = rg.table(results)
    rg.write_markdown(rows, d / "table.md")
    rg.write_csv(rows, d / "table.csv")
    return d
