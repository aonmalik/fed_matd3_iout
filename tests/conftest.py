import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest
from fed_matd3_iout.utils.config import load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def smoke_cfg():
    return load_config(["smoke.yaml"])
