from __future__ import annotations

import json

import pandas as pd
import pytest

from base.plugin_loader import validate_plugin_info
from plugin_api import (
    API_VERSION,
    ChannelBlock,
    ParamSpec,
    PluginInfo,
    TransformResult,
    complete_params,
    default_params,
)


def test_default_and_complete_params_are_fresh_and_preserve_extra_keys():
    info = PluginInfo(
        plugin_id="example.scale",
        display_name="Scale",
        version="1.0.0",
        params_schema=(
            ParamSpec("factor", "Factor", float, 1.0, minimum=0.0),
            ParamSpec("enabled", "Enabled", bool, True),
        ),
    )

    defaults = default_params(info)
    values = complete_params(info, {"factor": 2.5, "future_option": "kept"})

    assert defaults == {"factor": 1.0, "enabled": True}
    assert values == {"factor": 2.5, "enabled": True, "future_option": "kept"}
    assert defaults is not values
    json.dumps(values)


def test_plugin_api_data_types_do_not_require_qt():
    series = pd.Series([1.0, float("nan")], index=pd.date_range("2025-01-01", periods=2, freq="s"))
    block = ChannelBlock("Ex", series)
    result = TransformResult(series.copy(), {"count": 2})

    assert block.channel == "Ex"
    assert result.series.index.equals(block.series.index)


@pytest.mark.parametrize(
    ("info", "match"),
    [
        (PluginInfo("", "Name", "1.0"), "plugin_id"),
        (PluginInfo("example.x", "Name", "1.0", api_version=API_VERSION + 1), "API version"),
        (PluginInfo("example.x", "Name", "1.0", category="output"), "category"),
        (PluginInfo("example.x", "Name", "1.0", capabilities=("network",)), "capabilities"),
        (
            PluginInfo(
                "example.x",
                "Name",
                "1.0",
                params_schema=(
                    ParamSpec("x", "X", int, 1),
                    ParamSpec("x", "X again", int, 2),
                ),
            ),
            "duplicate parameter",
        ),
    ],
)
def test_plugin_info_validation_rejects_invalid_contracts(info, match):
    with pytest.raises((TypeError, ValueError), match=match):
        validate_plugin_info(info)


def test_param_validation_rejects_non_json_default():
    info = PluginInfo(
        "example.bad-default",
        "Bad default",
        "1.0",
        params_schema=(ParamSpec("value", "Value", str, object()),),
    )

    with pytest.raises(ValueError, match="JSON serializable"):
        validate_plugin_info(info)
