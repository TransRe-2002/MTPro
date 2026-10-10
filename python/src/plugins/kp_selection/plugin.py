from __future__ import annotations

import numpy as np

from core.algorithms.kp_selection import select_kp_samples
from plugin_api import ParamSpec, PluginInfo, TransformResult


class KpSelectionPlugin:
    info = PluginInfo(
        plugin_id="kp_selection",
        display_name="Kp 阈值选取",
        version="1.0.0",
        category="auto",
        description="按 Kp 上下限选取有效时间段，条件外样点设为 NaN。",
        params_schema=(
            ParamSpec("lower_enabled", "启用下限", bool, False),
            ParamSpec("lower_value", "Kp 下限", float, 0.0, minimum=0.0, maximum=9.0),
            ParamSpec("lower_inclusive", "下限包含等号", bool, True),
            ParamSpec("upper_enabled", "启用上限", bool, True),
            ParamSpec("upper_value", "Kp 上限", float, 4.0, minimum=0.0, maximum=9.0),
            ParamSpec("upper_inclusive", "上限包含等号", bool, False),
            ParamSpec(
                "minimum_duration_hours", "最短连续时长（小时）", float, 0.0,
                minimum=0.0,
            ),
            ParamSpec(
                "trim_start_hours", "区间起点向后裁去（小时）", float, 0.0,
                minimum=0.0,
            ),
            ParamSpec(
                "trim_end_hours", "区间终点向前裁去（小时）", float, 0.0,
                minimum=0.0,
            ),
            ParamSpec(
                "missing_policy", "Kp 缺失时段", str, "exclude",
                options=("exclude", "keep"),
            ),
        ),
    )

    def process(self, block, params, ctx):
        kp_data = block.auxiliary.get("kp_data")
        ctx.progress(0.1, "计算 Kp 有效时段")
        selection = select_kp_samples(block.series.index, kp_data, params)
        values = block.series.to_numpy(dtype=np.float64, copy=True)
        before = int(np.count_nonzero(np.isfinite(values)))
        values[~selection.keep_mask] = np.nan
        after = int(np.count_nonzero(np.isfinite(values)))
        ctx.progress(1.0, "Kp 时段选取完成")
        cleaned = block.series.copy(deep=True)
        cleaned.iloc[:] = values
        return TransformResult(
            cleaned,
            {
                "retained_samples": after,
                "removed_samples": before - after,
                "selected_intervals": len(selection.selected_intervals),
                "selected_duration_hours": selection.selected_duration_hours,
            },
        )
