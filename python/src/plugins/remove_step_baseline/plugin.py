from __future__ import annotations

import numpy as np
import pandas as pd

from plugins.common.edit_codec import apply_operations, encode_edit
from core.algorithms.step import robust_baseline_destep
from plugin_api import ParamSpec, PluginInfo, TransformResult


class StepBaselinePlugin:
    info = PluginInfo(
        plugin_id="remove_step_baseline",
        display_name="remove step baseline",
        version="1.0.0",
        category="auto",
        params_schema=(
            ParamSpec("region", "过渡区间", list, []),
            ParamSpec("baseline_degree", "基线阶数", int, 1, minimum=0, maximum=1),
            ParamSpec("transition_degree", "过渡阶数", int, 2, minimum=1, maximum=10),
            ParamSpec("context_points", "上下文点数", int, 200, minimum=1),
            ParamSpec("loss", "鲁棒损失", str, "soft_l1", options=("soft_l1", "huber", "linear", "cauchy", "arctan")),
            ParamSpec("operations", "确认的编辑", list, []),
        ),
    )

    def process(self, block, params, ctx):
        source = block.series.to_numpy(dtype=np.float64)
        if params["operations"]:
            corrected = apply_operations(source, params["operations"])
            diagnostics = {"mode": "replay"}
        else:
            region = params["region"]
            if not region:
                finite = source[np.isfinite(source)]
                index = int(np.argmax(np.abs(np.diff(finite)))) if finite.size > 1 else 0
                region = [index, min(source.size - 1, index + 1)]
            result = robust_baseline_destep(
                source,
                tuple(region),
                baseline_degree=int(params["baseline_degree"]),
                transition_degree=int(params["transition_degree"]),
                context_points=int(params["context_points"]),
                loss=str(params["loss"]),
            )
            corrected = result.corrected
            diagnostics = {"mode": "auto", "step_height": result.step_height, "cost": result.cost}
        return TransformResult(pd.Series(corrected, index=block.series.index, name=block.series.name), diagnostics)

    def create_editor(self, host, block, params):
        from plugins.common.legacy_editor import slice_processor_editor
        from plugins.remove_step_baseline.editor import RemoveStepByBaseline

        def extract(editor, source_block, result, _old):
            last = editor.last_result
            return {
                "region": list(last.region) if last is not None else [],
                "baseline_degree": int(editor.base_degree_spin.value()),
                "transition_degree": int(editor.transition_degree_spin.value()),
                "context_points": int(editor.context_spin.value()),
                "loss": editor.loss_combo.currentText(),
                "operations": [encode_edit(source_block.series.to_numpy(dtype=np.float64), result)],
            }

        return slice_processor_editor(RemoveStepByBaseline, host, block, params, extract)
