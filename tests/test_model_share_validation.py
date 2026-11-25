"""acoustic.usv_link.model_share is the airtime POOL fraction (D7): it must be finite and within [0, 1]; the default 0.5 is unchanged."""
import pytest

from fed_matd3_iout.utils.config import ConfigError, load_config


def test_default_is_half_and_valid():
    assert load_config()["acoustic"]["usv_link"]["model_share"] == 0.5


@pytest.mark.parametrize("v", [0.0, 0.25, 0.5, 1.0])
def test_valid_values_accepted(v):
    assert load_config(overrides={"acoustic.usv_link.model_share": v})["acoustic"]["usv_link"]["model_share"] == v


@pytest.mark.parametrize("v", [-0.1, -1e-9, 1.0000001, 2.0, float("nan"), float("inf"), float("-inf"), "abc", None])
def test_invalid_values_rejected(v):
    with pytest.raises(ConfigError, match="model_share"):
        load_config(overrides={"acoustic.usv_link.model_share": v})
