"""Every third-party package imported by src/scripts/tests must be declared (the audit environment lacked gymnasium)."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPORT_TO_DIST = {"yaml": "pyyaml", "torch": "torch", "numpy": "numpy", "scipy": "scipy", "gymnasium": "gymnasium", "matplotlib": "matplotlib", "pytest": "pytest"}


def _declared():
    req = {re.split(r"[<>=!~\[ ]", l.strip(), maxsplit=1)[0].lower() for l in (ROOT / "requirements.txt").read_text().splitlines() if l.strip() and not l.startswith("#")}
    py = (ROOT / "pyproject.toml").read_text()
    deps = re.search(r"dependencies = \[(.*?)\]", py, re.S).group(1)
    dev = re.search(r"dev = \[(.*?)\]", py, re.S).group(1)
    pj = {re.split(r"[<>=!~\[ ]", d.strip(' "\n'), maxsplit=1)[0].lower() for d in (deps + "," + dev).split(",") if d.strip(' "\n')}
    return req, pj


def _imports():
    mods = set()
    for d in ("src", "scripts", "tests"):
        for f in (ROOT / d).rglob("*.py"):
            for n in ast.walk(ast.parse(f.read_text())):
                if isinstance(n, ast.Import):
                    mods |= {a.name.split(".")[0] for a in n.names}
                elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                    mods.add(n.module.split(".")[0])
    return mods


def test_all_third_party_imports_are_declared_in_both_files():
    req, pj = _declared()
    local = {f.stem for d in ("src", "scripts", "tests") for f in (ROOT / d).rglob("*.py")} | {"fed_matd3_iout"}   # project-local modules/helpers
    third = {m for m in _imports() if m not in sys.stdlib_module_names and m not in local}
    unknown = third - set(IMPORT_TO_DIST)
    assert not unknown, f"undeclared/unmapped third-party imports: {unknown}"
    for m in third:
        d = IMPORT_TO_DIST[m]
        assert d in req, f"{d} missing from requirements.txt"
        assert d in pj, f"{d} missing from pyproject.toml"


def test_gymnasium_is_declared():
    req, pj = _declared()
    assert "gymnasium" in req and "gymnasium" in pj
