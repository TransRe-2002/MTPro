from __future__ import annotations

from core.algorithms.filters import detrend_series
from plugin_api import ParamSpec, PluginInfo, TransformResult


class DetrendPlugin:
    info = PluginInfo(
        plugin_id="detrend",
        display_name="去趋势",
        version="1.0.0",
        category="auto",
        description="按连续有效段去除常数或线性趋势，不跨越缺失值拟合。",
        params_schema=(
            ParamSpec("kind", "趋势类型", str, "linear", options=("linear", "constant")),
            ParamSpec("min_segment_points", "最短有效段点数", int, 4, minimum=2),
        ),
    )

    def process(self, block, params, ctx):
        ctx.progress(0.1, "去除趋势")
        result = detrend_series(
            block.series,
            kind=str(params["kind"]),
            min_segment_points=int(params["min_segment_points"]),
        )
        ctx.progress(1.0, "去趋势完成")
        return TransformResult(
            result.cleaned,
            {
                "processed_segments": [list(item) for item in result.processed_segments],
                "skipped_segments": [list(item) for item in result.skipped_segments],
            },
        )
