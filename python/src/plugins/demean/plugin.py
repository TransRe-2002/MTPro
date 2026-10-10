from __future__ import annotations

import numpy as np

from core.algorithms.filters import demean_series
from plugin_api import PluginInfo, TransformResult


class DemeanPlugin:
    info = PluginInfo(
        plugin_id="demean",
        display_name="去均值",
        version="1.0.0",
        category="auto",
        description="对整条通道减去原始有效样点的全局均值，并恢复缺失位置。",
    )

    def process(self, block, params, ctx):
        del params
        ctx.progress(0.1, "去除全局均值")
        result = demean_series(block.series)
        finite = block.series.to_numpy(dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        ctx.progress(1.0, "去均值完成")
        return TransformResult(
            result.cleaned,
            {
                "original_mean": float(np.mean(finite)) if finite.size else None,
                "processed_segments": [list(item) for item in result.processed_segments],
            },
        )

