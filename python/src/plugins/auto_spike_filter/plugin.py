from __future__ import annotations

import numpy as np
import pandas as pd

from core.algorithms.spike import detect_auto_spikes
from plugin_api import ParamSpec, PluginInfo, TransformResult


class AutoSpikeFilterPlugin:
    """对单通道短脉冲飞点执行鲁棒自动筛选。"""

    info = PluginInfo(
        plugin_id="auto_spike_filter",
        display_name="auto spike filter",
        version="1.0.0",
        category="auto",
        description=(
            "以重叠窗口的中位数/MAD 为局部基线，自动将不超过最大宽度的"
            "极端脉冲飞点标记为 NaN；长异常段和潜在台阶不自动删除。"
        ),
        params_schema=(
            ParamSpec("window_points", "窗口点数", int, 101, minimum=3),
            ParamSpec("overlap_percent", "窗口重叠 (%)", float, 50.0, minimum=0.0, maximum=90.0),
            ParamSpec("threshold_sigma", "鲁棒阈值 (σ)", float, 10.0, minimum=0.001),
            ParamSpec("max_pulse_width", "最大脉冲宽度", int, 5, minimum=1),
        ),
    )

    def process(self, block, params, ctx):
        values = block.series.to_numpy(dtype=np.float64, copy=True)
        detection = detect_auto_spikes(
            values,
            window_points=int(params["window_points"]),
            overlap_percent=float(params["overlap_percent"]),
            threshold_sigma=float(params["threshold_sigma"]),
            max_pulse_width=int(params["max_pulse_width"]),
        )
        values[detection.removed_indices] = np.nan
        diagnostics = {
            "windows": detection.window_count,
            "candidates": int(detection.candidate_indices.size),
            "removed": int(detection.removed_indices.size),
            "long_runs": [list(item) for item in detection.long_runs],
            "protected_runs": [list(item) for item in detection.protected_runs],
        }
        return TransformResult(
            pd.Series(values, index=block.series.index, name=block.series.name),
            diagnostics,
        )

    def create_editor(self, host, block, params):
        from plugins.auto_spike_filter.editor import AutoSpikeFilter
        from plugins.common.legacy_editor import slice_standalone_editor

        def extract(editor, _source_block, _result, _old):
            return editor._params()

        return slice_standalone_editor(AutoSpikeFilter, host, block, params, extract)
