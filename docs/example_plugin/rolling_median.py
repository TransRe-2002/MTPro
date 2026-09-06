"""Minimal third-party MTPro plugin example."""
from __future__ import annotations

from plugin_api import ParamSpec, PluginInfo, TransformResult


class RollingMedianPlugin:
    info = PluginInfo(
        plugin_id="example.rolling_median",
        display_name="Rolling Median",
        version="1.0.0",
        category="auto",
        description="Replace samples with a centered rolling median.",
        params_schema=(
            ParamSpec("window", "Window", int, 5, minimum=1, maximum=10001),
        ),
    )

    def process(self, block, params, ctx):
        window = max(1, int(params["window"]))
        if window % 2 == 0:
            window += 1
        ctx.progress(0.1, "calculating rolling median")
        series = block.series.rolling(window, center=True, min_periods=1).median()
        series[block.series.isna()] = float("nan")
        ctx.progress(1.0, "done")
        return TransformResult(series, {"window": window})
