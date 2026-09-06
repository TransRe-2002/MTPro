"""Pipeline-facing EMD decomposition and selected-IMF reconstruction."""
from __future__ import annotations

import pandas as pd

from core.algorithms.emd import EMDConfig, emd_denoise_series
from plugin_api import ParamSpec, PluginInfo, TransformResult


class EMDDenoisePlugin:
    """Classical spline-envelope EMD for single-channel finite time-series runs."""

    info = PluginInfo(
        plugin_id="emd_denoise",
        display_name="EMD 经验模态分解去噪",
        version="0.1.0",
        category="auto",
        description=(
            "以经验模态分解将连续有效数据提取为 IMF 和残差。"
            "IMF 按提取顺序从高频到低频编号；默认保留全部 IMF 与残差。"
        ),
        params_schema=(
            ParamSpec("max_imfs", "最大 IMF 数", int, 8, minimum=1, maximum=32),
            ParamSpec("max_siftings", "单 IMF 最大筛分次数", int, 100, minimum=1, maximum=10000),
            ParamSpec("sift_tolerance", "筛分相对变化阈值", float, 0.2, minimum=1.0e-12),
            ParamSpec("min_segment_points", "最短有效段点数", int, 32, minimum=4),
            ParamSpec("keep_imfs", "保留 IMF", str, "all"),
            ParamSpec("keep_residue", "保留低频残差", bool, True),
        ),
    )

    def create_editor(self, host, block, params):
        from plugins.emd_denoise.editor import EMDDenoiseEditor

        return EMDDenoiseEditor(host, params)

    def process(self, block, params, ctx):
        if ctx.is_cancelled():
            raise RuntimeError("EMD processing was cancelled before it started")
        ctx.progress(0.05, "准备 EMD 筛分")
        config = EMDConfig(
            max_imfs=int(params["max_imfs"]),
            max_siftings=int(params["max_siftings"]),
            sift_tolerance=float(params["sift_tolerance"]),
            min_segment_points=int(params["min_segment_points"]),
        )
        result = emd_denoise_series(
            block.series,
            config,
            keep_imfs=str(params["keep_imfs"]),
            keep_residue=bool(params["keep_residue"]),
        )
        if ctx.is_cancelled():
            raise RuntimeError("EMD processing was cancelled")
        diagnostics = {
            "processed_segments": [
                {
                    "start": item.start,
                    "stop": item.stop,
                    "selected_imfs": list(item.selected_imfs),
                    "imf_energies": list(item.imf_energies),
                    "residue_energy": item.residue_energy,
                    "siftings": list(item.siftings),
                    "reconstruction_residual_rms": item.reconstruction_residual_rms,
                    "used_rust": item.used_rust,
                }
                for item in result.processed_segments
            ],
            "skipped_segments": [list(item) for item in result.skipped_segments],
        }
        ctx.progress(1.0, "EMD 重构完成")
        return TransformResult(
            pd.Series(result.cleaned, index=block.series.index, name=block.series.name),
            diagnostics,
        )
