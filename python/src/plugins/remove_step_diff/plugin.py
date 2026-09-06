from __future__ import annotations

import numpy as np
import pandas as pd

from core.algorithms.step import zero_diff_indices
from plugin_api import ParamSpec, PluginInfo, TransformResult


class StepDiffPlugin:
    info = PluginInfo(
        plugin_id="remove_step_diff",
        display_name="remove step diff",
        version="1.0.0",
        category="manual",
        params_schema=(ParamSpec("indices", "差分点索引", list, []),),
    )

    def process(self, block, params, ctx):
        corrected, removed = zero_diff_indices(
            block.series.to_numpy(dtype=np.float64), params["indices"]
        )
        return TransformResult(
            pd.Series(corrected, index=block.series.index, name=block.series.name),
            {"removed_steps": removed},
        )

    def create_editor(self, host, block, params):
        from plugins.common.legacy_editor import slice_processor_editor
        from plugins.remove_step_diff.editor import RemoveStepByDiff

        def extract(_editor, source_block, result, _old):
            before = source_block.series.to_numpy(dtype=np.float64)
            after = np.asarray(result, dtype=np.float64)
            finite = np.isfinite(before) & np.isfinite(after)
            before_diff = np.diff(before[finite])
            after_diff = np.diff(after[finite])
            changed = ~np.isclose(before_diff, after_diff, equal_nan=True)
            removed = changed & np.isclose(after_diff, 0.0, atol=1e-12)
            return {"indices": np.flatnonzero(removed).astype(int).tolist()}

        return slice_processor_editor(RemoveStepByDiff, host, block, params, extract)
