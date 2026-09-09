"""Qt-free multi-target recipe execution on the shared compute pool."""
from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import Future
from typing import Any
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from core.recipe import Recipe, RecipeStep
from base.plugin_runtime import run_plugin_process
from plugin_api import ChannelBlock, ChannelMeta
from utils.concurrency import compute_pool


POINT_INDEX_PARAM_NAMES = frozenset({"indices"})


@dataclass(frozen=True)
class BatchTarget:
    data_id: int
    channel: str
    series: pd.Series
    meta: ChannelMeta = field(default_factory=ChannelMeta)

    @property
    def key(self) -> tuple[int, str]:
        return self.data_id, self.channel


@dataclass(frozen=True)
class BatchTargetResult:
    target: BatchTarget
    status: str
    series: pd.Series
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    skipped_steps: tuple[str, ...] = ()
    error: str = ""


class BatchRunner:
    def __init__(self, plugins: Mapping[str, object]):
        self.plugins = dict(plugins)

    def submit(self, recipe: Recipe, targets: Iterable[BatchTarget]) -> dict[tuple[int, str], Future]:
        return {
            target.key: compute_pool.submit(self.run_target, recipe, target)
            for target in targets
        }

    def run_target(self, recipe: Recipe, target: BatchTarget) -> BatchTargetResult:
        index = target.series.index
        name = target.series.name
        current_values = target.series.to_numpy(dtype=np.float64, copy=True)
        diagnostics: list[Mapping[str, Any]] = []
        skipped: list[str] = []
        try:
            for position, step in enumerate(recipe.steps, 1):
                plugin = self.plugins.get(step.plugin)
                if plugin is None:
                    raise KeyError(f"plugin is not loaded: {step.plugin}")
                if self._point_indices_misaligned(step, index):
                    skipped.append(f"step {position} ({step.plugin}): sample interval mismatch")
                    continue
                selection = self._selection(index, step.time_range)
                selection_index = index[selection]
                if len(selection_index) == 0:
                    skipped.append(f"step {position} ({step.plugin}): empty time range")
                    continue
                # Plugins receive an isolated region copy so an implementation
                # cannot mutate the batch working buffer through pandas.  The
                # full channel itself is allocated only once per target.
                sliced = pd.Series(
                    np.array(current_values[selection], copy=True),
                    index=selection_index,
                    name=name,
                )
                block = ChannelBlock(target.channel, sliced, target.meta)
                result, _params = run_plugin_process(plugin, block, step.params)
                current_values[selection] = result.series.to_numpy(
                    dtype=np.float64,
                    copy=False,
                )
                diagnostics.append(dict(result.diagnostics))
        except Exception as exc:
            return BatchTargetResult(
                target=target,
                status="error",
                series=pd.Series(current_values, index=index, name=name, copy=False),
                diagnostics=tuple(diagnostics),
                skipped_steps=tuple(skipped),
                error=repr(exc),
            )
        return BatchTargetResult(
            target=target,
            status="ok",
            series=pd.Series(current_values, index=index, name=name, copy=False),
            diagnostics=tuple(diagnostics),
            skipped_steps=tuple(skipped),
        )

    @staticmethod
    def _selection(
        index: pd.DatetimeIndex,
        time_range: tuple[str, str] | None,
    ) -> slice | np.ndarray:
        if time_range is None:
            return slice(0, len(index))
        start = pd.Timestamp(time_range[0])
        end = pd.Timestamp(time_range[1])
        if index.tz is not None:
            start = start.tz_localize(index.tz) if start.tzinfo is None else start.tz_convert(index.tz)
            end = end.tz_localize(index.tz) if end.tzinfo is None else end.tz_convert(index.tz)
        lower, upper = min(start, end), max(start, end)
        if index.is_monotonic_increasing:
            return slice(
                int(index.searchsorted(lower, side="left")),
                int(index.searchsorted(upper, side="right")),
            )
        return np.flatnonzero((index >= lower) & (index <= upper))

    @staticmethod
    def _point_indices_misaligned(step: RecipeStep, index: pd.DatetimeIndex) -> bool:
        has_indices = any(step.params.get(name) for name in POINT_INDEX_PARAM_NAMES)
        if not has_indices or step.sample_interval_seconds is None or len(index) < 2:
            return False
        target_interval = float((index[1] - index[0]).total_seconds())
        return not np.isclose(target_interval, step.sample_interval_seconds, rtol=0.0, atol=1e-9)
