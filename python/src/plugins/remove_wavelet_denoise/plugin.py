from __future__ import annotations

import numpy as np
import pandas as pd

from plugins.common.edit_codec import apply_operations, encode_edit
from core.algorithms.step import wavelet_multiscale_denoise
from plugin_api import ParamSpec, PluginInfo, TransformResult


class WaveletDenoisePlugin:
    info = PluginInfo(
        plugin_id="remove_wavelet_denoise",
        display_name="remove wavelet denoise",
        version="1.1.0",
        category="auto",
        params_schema=(
            ParamSpec("region", "处理区间", list, []),
            ParamSpec("transform", "变换", str, "dwt", options=("dwt", "swt", "wp")),
            ParamSpec("wavelet", "小波", str, "db4"),
            ParamSpec("level", "层数", int, 6, minimum=1, maximum=12),
            ParamSpec("level_policy", "层数策略", str, "manual", options=("manual", "auto_by_period")),
            ParamSpec("protect_period_seconds", "保护周期（秒）", float, 3600.0, minimum=0.001),
            ParamSpec("threshold", "阈值比例", float, 1.0, minimum=0.0),
            ParamSpec("threshold_method", "阈值策略", str, "universal", options=("universal", "sure")),
            ParamSpec("padding", "边缘扩展", int, 256, minimum=0),
            ParamSpec("padding_unit", "边缘扩展单位", str, "points", options=("points", "seconds")),
            ParamSpec("padding_seconds", "边缘扩展（秒）", float, 0.0, minimum=0.0),
            ParamSpec("mode", "模式", str, "soft", options=("soft", "hard", "zero", "manual")),
            ParamSpec("detail_gains", "细节增益", list, []),
            ParamSpec("operations", "确认的编辑", list, []),
        ),
    )

    def process(self, block, params, ctx):
        source = block.series.to_numpy(dtype=np.float64)
        if params["operations"]:
            corrected = apply_operations(source, params["operations"])
            diagnostics = {"mode": "replay"}
        else:
            region = params["region"] or [0, max(0, source.size - 1)]
            result = wavelet_multiscale_denoise(
                source,
                tuple(region),
                transform_kind=str(params["transform"]),
                wavelet_name=str(params["wavelet"]),
                level=int(params["level"]),
                sample_rate_hz=block.meta.sample_rate or 1.0,
                threshold_scale=float(params["threshold"]),
                padding_points=int(params["padding"]),
                padding_seconds=(
                    float(params["padding_seconds"])
                    if params["padding_unit"] == "seconds" and float(params["padding_seconds"]) > 0.0 else None
                ),
                mode=str(params["mode"]),
                threshold_method=str(params["threshold_method"]),
                level_policy=str(params["level_policy"]),
                protect_period_seconds=(
                    float(params["protect_period_seconds"])
                    if params["level_policy"] == "auto_by_period" else None
                ),
                detail_gains=params["detail_gains"] or None,
            )
            corrected = result.corrected
            diagnostics = {"mode": "auto", "changed_coefficients": result.changed_coefficients}
        return TransformResult(pd.Series(corrected, index=block.series.index, name=block.series.name), diagnostics)

    def create_editor(self, host, block, params):
        from plugins.common.legacy_editor import slice_processor_editor
        from plugins.remove_wavelet_denoise.editor import RemoveWaveletDenoise

        def extract(editor, source_block, result, _old):
            last = editor.last_result
            return {
                "region": list(last.region) if last is not None else [],
                "transform": editor._transform_kind(),
                "wavelet": editor.wavelet_combo.currentText(),
                "level": int(editor.level_spin.value()),
                "level_policy": editor._level_policy(),
                "protect_period_seconds": float(editor.protect_period_spin.value()),
                "threshold": float(editor.threshold_scale_spin.value()),
                "threshold_method": editor._threshold_method(),
                "padding": editor._padding_points(),
                "padding_unit": editor._padding_unit(),
                "padding_seconds": (
                    float(editor.padding_seconds_spin.value())
                    if editor._padding_unit() == "seconds" else 0.0
                ),
                "mode": editor.mode_combo.currentText(),
                "detail_gains": editor._current_detail_gains(),
                "operations": [encode_edit(source_block.series.to_numpy(dtype=np.float64), result)],
            }

        return slice_processor_editor(RemoveWaveletDenoise, host, block, params, extract)
