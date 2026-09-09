from __future__ import annotations

from core.algorithms.filters import median_filter_series
from plugin_api import ParamSpec, PluginInfo, TransformResult


class MedianFilterPlugin:
    pipeline_preview = True

    info = PluginInfo(
        plugin_id="median_filter",
        display_name="滑动中值滤波",
        version="1.0.0",
        category="auto",
        description="按连续有效段实施居中中值滤波，抑制短时脉冲且保留缺失值。",
        params_schema=(
            ParamSpec("window_points", "窗口点数（奇数）", int, 5, minimum=3),
            ParamSpec("min_segment_points", "最短有效段点数", int, 5, minimum=3),
        ),
    )

    def process(self, block, params, ctx):
        ctx.progress(0.1, "执行中值滤波")
        result = median_filter_series(
            block.series,
            window_points=int(params["window_points"]),
            min_segment_points=int(params["min_segment_points"]),
        )
        ctx.progress(1.0, "中值滤波完成")
        return TransformResult(
            result.cleaned,
            {
                "processed_segments": [list(item) for item in result.processed_segments],
                "skipped_segments": [list(item) for item in result.skipped_segments],
            },
        )
