import ast
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from equation_registry import E  # noqa: E402


def defined(path: Path, dotted: str) -> bool:
    tree = ast.parse(path.read_text())
    parts = dotted.split(".")
    nodes = tree.body
    for i, p in enumerate(parts):
        hit = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == p]
        if not hit:
            return False
        nodes = hit[0].body if isinstance(hit[0], ast.ClassDef) else []
    return True


@pytest.mark.parametrize("e", E, ids=[e["id"] for e in E])
def test_implementation_and_test_exist(e):
    mod, func = e["impl"].split(":")
    assert (ROOT / mod).exists(), mod
    assert defined(ROOT / mod, func), f"{func} not found in {mod}"
    if e["test"]:
        f, t = e["test"].split("::")
        assert (ROOT / f).exists(), f
        assert defined(ROOT / f, t), f"test {t} missing in {f}"
    assert e["cls"] in "ABCDE"


def test_ids_unique_and_all_classes_documented():
    ids = [e["id"] for e in E]
    assert len(ids) == len(set(ids))
    assert {"Q1", "P1", "P3", "S2", "A1", "C8", "M3", "F5", "F7", "R1"} <= set(ids)
