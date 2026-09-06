from __future__ import annotations

import numpy as np
import pandas as pd

from plugin_api import ParamSpec, PluginInfo, TransformResult


class RegionRemovalPlugin:
    info = PluginInfo(
        plugin_id="remove_region",
        display_name="remove region",
        version="1.0.0",
        category="manual",
        capabilities=("channel_sync", "region_sync"),
        params_schema=(ParamSpec("time_ranges", "删除时间区间", list, []),),
    )

    def process(self, block, params, ctx):
        values = block.series.to_numpy(dtype=np.float64, copy=True)
        index = block.series.index
        removed = 0
        for raw_start, raw_end in params["time_ranges"]:
            start = pd.Timestamp(raw_start)
            end = pd.Timestamp(raw_end)
            if index.tz is not None:
                if start.tzinfo is None:
                    start = start.tz_localize(index.tz)
                else:
                    start = start.tz_convert(index.tz)
                if end.tzinfo is None:
                    end = end.tz_localize(index.tz)
                else:
                    end = end.tz_convert(index.tz)
            mask = (index >= min(start, end)) & (index <= max(start, end))
            removed += int(np.count_nonzero(mask & np.isfinite(values)))
            values[mask] = np.nan
        return TransformResult(
            pd.Series(values, index=index, name=block.series.name),
            {"removed": removed, "ranges": len(params["time_ranges"])},
        )

    def create_editor(self, host, block, params):
        from plugins.common.legacy_editor import slice_standalone_editor
        from plugins.remove_region.editor import RemoveRegion

        def extract(_editor, source_block, result, _old):
            before = source_block.series.to_numpy(dtype=np.float64)
            after = np.asarray(result, dtype=np.float64)
            selected = np.flatnonzero(np.isfinite(before) & np.isnan(after))
            ranges: list[list[str]] = []
            if selected.size:
                run_start = 0
                for position in range(1, selected.size + 1):
                    if position == selected.size or selected[position] != selected[position - 1] + 1:
                        run = selected[run_start:position]
                        ranges.append([
                            source_block.series.index[int(run[0])].isoformat(),
                            source_block.series.index[int(run[-1])].isoformat(),
                        ])
                        run_start = position
            return {"time_ranges": ranges}

        return slice_standalone_editor(RemoveRegion, host, block, params, extract)
