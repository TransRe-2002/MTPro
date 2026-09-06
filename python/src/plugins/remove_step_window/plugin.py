from __future__ import annotations

import numpy as np
import pandas as pd

from plugins.common.edit_codec import apply_operations, encode_edit
from core.algorithms.step import windowed_mean_destep_with_diagnostics
from plugin_api import ParamSpec, PluginInfo, TransformResult


class StepWindowPlugin:
    info = PluginInfo(
        plugin_id="remove_step_window",
        display_name="remove step window",
        version="1.0.0",
        category="auto",
        params_schema=(
            ParamSpec("threshold", "差分阈值", float, 100.0, minimum=0.0),
            ParamSpec("window", "平均窗口", int, 1000, minimum=1),
            ParamSpec("max_steps", "最多台阶", int, 20, minimum=1),
            ParamSpec("operations", "确认的编辑", list, []),
        ),
    )

    def process(self, block, params, ctx):
        source = block.series.to_numpy(dtype=np.float64)
        if params["operations"]:
            corrected = apply_operations(source, params["operations"])
            diagnostics = {"mode": "replay"}
        else:
            corrected, applied, candidates = windowed_mean_destep_with_diagnostics(
                source,
                float(params["threshold"]),
                avg_window=int(params["window"]),
                max_steps=int(params["max_steps"]),
            )
            diagnostics = {"mode": "auto", "applied_steps": applied, "candidates": len(candidates)}
        return TransformResult(pd.Series(corrected, index=block.series.index, name=block.series.name), diagnostics)

    def create_editor(self, host, block, params):
        from plugins.common.legacy_editor import slice_processor_editor
        from plugins.remove_step_window.editor import RemoveStepByWindow

        def extract(editor, source_block, result, _old):
            return {
                "threshold": float(editor.de_step_threshold.text()),
                "window": int(editor.avg_window_edit.text()),
                "max_steps": int(editor.max_steps_edit.text()),
                "operations": [encode_edit(source_block.series.to_numpy(dtype=np.float64), result)],
            }

        return slice_processor_editor(RemoveStepByWindow, host, block, params, extract)
