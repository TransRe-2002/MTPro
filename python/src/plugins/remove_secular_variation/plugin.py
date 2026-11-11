"""Pipeline plugin for removing a smooth secular-variation baseline."""
from __future__ import annotations

import numpy as np

from core.algorithms.secular_variation import remove_local_secular_variation_series, remove_secular_variation_series
from plugin_api import ParamSpec, PluginInfo, TransformResult


class SecularVariationPlugin:
    info = PluginInfo(
        plugin_id="remove_secular_variation",
        display_name="去除缓慢背景变化（SV）",
        version="1.0.0",
        category="auto",
        description=(
            "按 TS.removeSV 的局部均值、锚点平滑和三次样条流程估计长周期背景并相减。"
            "参数以采样点为单位；原始缺测位置保持 NaN。适合长时间序列，不等同于 Hankel-SVD 去噪。"
        ),
        params_schema=(
            ParamSpec("smoothing_window_points", "平滑窗口（点）", int, 8_760, minimum=1),
            ParamSpec("anchor_spacing_points", "锚点间隔（点）", int, 4_380, minimum=1),
            ParamSpec("min_valid_fraction", "锚点最小有效比例", float, 0.25, minimum=0.0, maximum=1.0),
            ParamSpec("region", "背景估计范围", list, []),
            ParamSpec("effect_both_sides", "两侧延续校正", bool, False),
        ),
    )

    def create_editor(self, host, block, params):
        from plugins.remove_secular_variation.editor import SecularVariationEditor

        return SecularVariationEditor(host, block, params)

    def process(self, block, params, ctx):
        if ctx.is_cancelled():
            raise RuntimeError("secular-variation subtraction was cancelled before it started")
        ctx.progress(0.1, "估计缓慢背景基线")
        compute = remove_local_secular_variation_series if params["effect_both_sides"] else remove_secular_variation_series
        positional = (block.series, tuple(params["region"])) if params["effect_both_sides"] else (block.series,)
        result = compute(
            *positional,
            smoothing_window_points=int(params["smoothing_window_points"]),
            anchor_spacing_points=int(params["anchor_spacing_points"]),
            min_valid_fraction=float(params["min_valid_fraction"]),
        )
        if result.status != "applied":
            raise ValueError("没有可处理的有效数据段，请调整平滑窗口或锚点间隔")
        if ctx.is_cancelled():
            raise RuntimeError("secular-variation subtraction was cancelled")
        baseline = result.estimated_baseline.to_numpy(dtype=np.float64)
        finite = baseline[np.isfinite(baseline)]
        ctx.progress(1.0, "缓慢背景基线处理完成")
        return TransformResult(
            result.cleaned,
            {
                "status": result.status,
                "anchor_count": result.anchor_count,
                "valid_anchor_count": result.valid_anchor_count,
                "processed_segments": result.processed_segments,
                "skipped_segments": result.skipped_segments,
                "baseline_mean": float(np.mean(finite)) if finite.size else None,
                "baseline_rms": float(np.sqrt(np.mean(np.square(finite)))) if finite.size else None,
            },
        )
