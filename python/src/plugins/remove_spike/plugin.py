from __future__ import annotations

import numpy as np
import pandas as pd

from plugin_api import ParamSpec, PluginInfo, TransformResult


class SpikeRemovalPlugin:
    info = PluginInfo(
        plugin_id="remove_spike",
        display_name="remove spike",
        version="1.0.0",
        category="manual",
        capabilities=("channel_sync",),
        params_schema=(ParamSpec("indices", "删除点索引", list, []),),
    )

    def process(self, block, params, ctx):
        values = block.series.to_numpy(dtype=np.float64, copy=True)
        indices = sorted({int(index) for index in params["indices"]})
        valid = [index for index in indices if 0 <= index < values.size]
        values[valid] = np.nan
        return TransformResult(
            pd.Series(values, index=block.series.index, name=block.series.name),
            {"removed": len(valid)},
        )

    def create_editor(self, host, block, params):
        from plugins.common.legacy_editor import slice_standalone_editor
        from plugins.remove_spike.editor import RemoveSpike

        def extract(_editor, source_block, result, _old):
            before = source_block.series.to_numpy(dtype=np.float64)
            after = np.asarray(result, dtype=np.float64)
            indices = np.flatnonzero(np.isfinite(before) & np.isnan(after))
            return {"indices": indices.astype(int).tolist()}

        return slice_standalone_editor(RemoveSpike, host, block, params, extract)
