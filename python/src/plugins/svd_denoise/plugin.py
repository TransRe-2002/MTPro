from __future__ import annotations

from core.algorithms.svd import svd_denoise_series
from plugin_api import ParamSpec, PluginInfo, TransformResult


class SVDDenoisePlugin:
    info = PluginInfo(
        plugin_id="svd_denoise",
        display_name="Hankel-SVD 去噪",
        version="0.1.0",
        category="auto",
        description="构造 Hankel 轨迹矩阵，保留指定低秩分量并沿反对角线平均重构。",
        params_schema=(
            ParamSpec("embedding_points", "嵌入窗口点数", int, 101, minimum=2, maximum=10000),
            ParamSpec("rank", "保留秩", int, 8, minimum=1, maximum=10000),
            ParamSpec("min_segment_points", "最短有效段点数", int, 128, minimum=4),
        ),
    )

    def process(self, block, params, ctx):
        ctx.progress(0.1, "构造 Hankel 矩阵并执行 SVD")
        result = svd_denoise_series(
            block.series,
            embedding_points=int(params["embedding_points"]),
            rank=int(params["rank"]),
            min_segment_points=int(params["min_segment_points"]),
        )
        ctx.progress(1.0, "Hankel-SVD 重构完成")
        return TransformResult(
            result.cleaned,
            {
                "processed_segments": [
                    {
                        "start": item.start,
                        "stop": item.stop,
                        "embedding_points": item.embedding_points,
                        "rank": item.rank,
                        "singular_values": list(item.singular_values),
                    }
                    for item in result.processed_segments
                ],
                "skipped_segments": [list(item) for item in result.skipped_segments],
            },
        )
