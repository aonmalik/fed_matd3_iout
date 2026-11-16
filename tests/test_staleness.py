import math
import pytest
from fed_matd3_iout.federated.staleness import is_acceptable, staleness, staleness_factor


def test_staleness_definition_and_factor():
    assert staleness(7, 4) == 3
    assert staleness_factor(0, 0.3) == 1.0
    assert staleness_factor(3, 0.3) == pytest.approx(math.exp(-0.9))
    assert staleness_factor(5, 0.3) < staleness_factor(2, 0.3)
    assert staleness_factor(5, 0.0) == 1.0                    # kappa=0 => no correction


def test_max_staleness_gate():
    assert is_acceptable(8, 8) and not is_acceptable(9, 8)
    assert not is_acceptable(-1, 8)                           # base version from the future is invalid
