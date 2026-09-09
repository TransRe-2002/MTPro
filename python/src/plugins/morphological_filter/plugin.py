from __future__ import annotations

from core.algorithms.filters import morphological_filter_series
from plugin_api import ParamSpec, PluginInfo, TransformResult


class MorphologicalFilterPlugin:
    pipeline_preview = True

    info = PluginInfo(
        plugin_id="morphological_filter",
        display_name="形态滤波",
        version="1.0.0",
        category="auto",
        description="以平坦结构元素的开/闭运算压制短时正负突变；窗口大小决定时间尺度。",
        params_schema=(
            ParamSpec("window_points", "结构元素点数（奇数）", int, 21, minimum=3),
            ParamSpec(
                "operation",
                "运算顺序",
                str,
                "open_close",
                options=("opening", "closing", "open_close", "close_open"),
            ),
            ParamSpec("min_segment_points", "最短有效段点数", int, 21, minimum=3),
        ),
    )

    def process(self, block, params, ctx):
        ctx.progress(0.1, "执行形态滤波")
        result = morphological_filter_series(
            block.series,
            window_points=int(params["window_points"]),
            operation=str(params["operation"]),
            min_segment_points=int(params["min_segment_points"]),
        )
        ctx.progress(1.0, "形态滤波完成")
        return TransformResult(
            result.cleaned,
            {
                "processed_segments": [list(item) for item in result.processed_segments],
                "skipped_segments": [list(item) for item in result.skipped_segments],
            },
        )
