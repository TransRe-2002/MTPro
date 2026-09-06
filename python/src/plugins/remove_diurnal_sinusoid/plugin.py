from __future__ import annotations

import numpy as np
import pandas as pd

from plugins.common.edit_codec import apply_operations, encode_edit
from core.algorithms.diurnal import subtract_diurnal_signal_series
from plugin_api import ParamSpec, PluginInfo, TransformResult


class DiurnalSinusoidPlugin:
    info = PluginInfo(
        plugin_id="remove_diurnal_sinusoid",
        display_name="remove diurnal sinusoid",
        version="1.0.0",
        category="auto",
        params_schema=(
            ParamSpec("harmonics", "谐波", list, [1, 2]),
            ParamSpec("window_days", "窗口天数", float, 3.0, minimum=0.01),
            ParamSpec("step_days", "步长天数", float, 1.0, minimum=0.01),
            ParamSpec("trend_degree", "趋势阶数", int, 1, minimum=0),
            ParamSpec("min_valid_fraction", "最小有效比例", float, 0.6, minimum=0.0, maximum=1.0),
            ParamSpec("min_valid_samples", "最小有效点数", int, 128, minimum=1),
            ParamSpec("operations", "确认的编辑", list, []),
        ),
    )

    def process(self, block, params, ctx):
        source = block.series.to_numpy(dtype=np.float64)
        if params["operations"]:
            corrected = apply_operations(source, params["operations"])
            diagnostics = {"mode": "replay"}
        else:
            result = subtract_diurnal_signal_series(
                block.series,
                harmonics=tuple(int(value) for value in params["harmonics"]),
                window_days=float(params["window_days"]),
                step_days=float(params["step_days"]),
                trend_degree=int(params["trend_degree"]),
                min_valid_fraction=float(params["min_valid_fraction"]),
                min_valid_samples=int(params["min_valid_samples"]),
            )
            corrected = result.cleaned.to_numpy(dtype=np.float64)
            diagnostics = {"mode": "auto", "windows": len(result.window_diagnostics)}
        return TransformResult(pd.Series(corrected, index=block.series.index, name=block.series.name), diagnostics)

    def create_editor(self, host, block, params):
        from plugins.common.legacy_editor import slice_processor_editor
        from plugins.remove_diurnal_sinusoid.editor import RemoveDiurnalSinusoid

        def extract(editor, source_block, result, _old):
            harmonics = editor.harmonics_combo.currentData() or ()
            return {
                "harmonics": [int(value) for value in harmonics],
                "window_days": float(editor.window_days_edit.text()),
                "step_days": float(editor.step_days_edit.text()),
                "trend_degree": int(editor.trend_degree_edit.text()),
                "min_valid_fraction": float(editor.min_valid_fraction_edit.text()),
                "min_valid_samples": int(editor.min_valid_samples_edit.text()),
                "operations": [encode_edit(source_block.series.to_numpy(dtype=np.float64), result)],
            }

        return slice_processor_editor(RemoveDiurnalSinusoid, host, block, params, extract)
