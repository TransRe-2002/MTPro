from __future__ import annotations

from core.algorithms.filters import detrend_series
from plugin_api import ParamSpec, PluginInfo, TransformResult


class DetrendPlugin:
    pipeline_preview = True

    info = PluginInfo(
        plugin_id="detrend",
        display_name="线性去趋势",
        version="1.0.0",
        category="auto",
        description="根据整条通道的原始有效样点拟合并去除全局线性趋势。",
        params_schema=(
            ParamSpec("min_segment_points", "最少有效点数", int, 4, minimum=2),
        ),
    )

    def process(self, block, params, ctx):
        ctx.progress(0.1, "去除趋势")
        result = detrend_series(
            block.series,
            # Keep old recipes containing kind="constant" executable while
            # new UI entries expose this tool specifically as linear detrend.
            kind=str(params.get("kind", "linear")),
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
