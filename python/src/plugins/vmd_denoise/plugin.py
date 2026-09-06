"""Pipeline-facing VMD decomposition and selected-mode reconstruction."""
from __future__ import annotations

import pandas as pd

from core.algorithms.vmd import VMDConfig, vmd_denoise_series
from plugin_api import ParamSpec, PluginInfo, TransformResult


class VMDDenoisePlugin:
    """Decompose a finite time-series run with VMD and reconstruct chosen modes."""

    info = PluginInfo(
        plugin_id="vmd_denoise",
        display_name="VMD 模态分解去噪",
        version="0.1.0",
        category="auto",
        description=(
            "以变分模态分解将连续有效数据分成窄带模态，并仅重构指定模态。"
            "模态按中心频率从低到高编号；默认 all 仅分解并重构，不自动丢弃信号。"
        ),
        params_schema=(
            ParamSpec("mode_count", "模态数 K", int, 4, minimum=1, maximum=32),
            ParamSpec("alpha", "带宽约束 α", float, 2000.0, minimum=0.001),
            ParamSpec("tau", "拉格朗日步长 τ", float, 0.0, minimum=0.0),
            ParamSpec("dc", "首模态固定直流", bool, False),
            ParamSpec("init", "中心频率初始化", str, "uniform", options=("uniform", "zero")),
            ParamSpec("tolerance", "收敛阈值", float, 1.0e-7, minimum=1.0e-12),
            ParamSpec("max_iter", "最大迭代次数", int, 500, minimum=1, maximum=10000),
            ParamSpec("min_segment_points", "最短有效段点数", int, 32, minimum=4),
            ParamSpec("keep_modes", "保留模态", str, "all"),
        ),
    )

    def create_editor(self, host, block, params):
        from plugins.vmd_denoise.editor import VMDDenoiseEditor

        return VMDDenoiseEditor(host, params)

    def process(self, block, params, ctx):
        if ctx.is_cancelled():
            raise RuntimeError("VMD processing was cancelled before it started")
        ctx.progress(0.05, "准备 VMD 分解")
        config = VMDConfig(
            mode_count=int(params["mode_count"]),
            alpha=float(params["alpha"]),
            tau=float(params["tau"]),
            dc=bool(params["dc"]),
            init=str(params["init"]),
            tolerance=float(params["tolerance"]),
            max_iter=int(params["max_iter"]),
            min_segment_points=int(params["min_segment_points"]),
        )
        result = vmd_denoise_series(
            block.series,
            config,
            keep_modes=str(params["keep_modes"]),
            sample_rate_hz=block.meta.sample_rate,
        )
        if ctx.is_cancelled():
            raise RuntimeError("VMD processing was cancelled")
        diagnostics = {
            "selected_modes": list(result.selected_modes),
            "processed_segments": [
                {
                    "start": item.start,
                    "stop": item.stop,
                    "center_frequencies_hz": list(item.center_frequencies_hz),
                    "mode_energies": list(item.mode_energies),
                    "reconstruction_residual_rms": item.reconstruction_residual_rms,
                    "iterations": item.iterations,
                    "convergence": item.convergence,
                    "used_rust": item.used_rust,
                }
                for item in result.processed_segments
            ],
            "skipped_segments": [list(item) for item in result.skipped_segments],
        }
        ctx.progress(1.0, "VMD 重构完成")
        return TransformResult(
            pd.Series(result.cleaned, index=block.series.index, name=block.series.name),
            diagnostics,
        )
