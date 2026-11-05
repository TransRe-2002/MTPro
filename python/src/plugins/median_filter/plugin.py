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
        description=(
            "对线性插值后的序列实施居中中值滤波，并在输出恢复原缺失位置。"
            "窗口必须为奇数；从 3–5 点开始试，窗口越大越容易削弱窄峰和快速天然变化。"
            "适合孤立尖峰，不适合长异常段、阶跃或连续高频噪声。"
            "NaN 在滤波前插值、滤波后恢复；长缺口附近请检查边缘结果。"
        ),
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
