"""Qt-free multi-target recipe execution on the shared compute pool."""
from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import Future
from typing import Any, Iterable, Mapping

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
        current = target.series.copy()
        diagnostics: list[Mapping[str, Any]] = []
        skipped: list[str] = []
        try:
            for position, step in enumerate(recipe.steps, 1):
                plugin = self.plugins.get(step.plugin)
                if plugin is None:
                    raise KeyError(f"plugin is not loaded: {step.plugin}")
                if self._point_indices_misaligned(step, current.index):
                    skipped.append(f"step {position} ({step.plugin}): sample interval mismatch")
                    continue
                selection = self._selection(current.index, step.time_range)
                if selection.size == 0:
                    skipped.append(f"step {position} ({step.plugin}): empty time range")
                    continue
                sliced = current.iloc[selection]
                block = ChannelBlock(target.channel, sliced, target.meta)
                result, _params = run_plugin_process(plugin, block, step.params)
                output = current.to_numpy(dtype=np.float64, copy=True)
                output[selection] = result.series.to_numpy(dtype=np.float64)
                current = pd.Series(output, index=current.index, name=current.name)
                diagnostics.append(dict(result.diagnostics))
        except Exception as exc:
            return BatchTargetResult(
                target=target,
                status="error",
                series=current,
                diagnostics=tuple(diagnostics),
                skipped_steps=tuple(skipped),
                error=repr(exc),
            )
        return BatchTargetResult(
            target=target,
            status="ok",
            series=current,
            diagnostics=tuple(diagnostics),
            skipped_steps=tuple(skipped),
        )

    @staticmethod
    def _selection(index: pd.DatetimeIndex, time_range: tuple[str, str] | None) -> np.ndarray:
        if time_range is None:
            return np.arange(len(index), dtype=np.intp)
        start = pd.Timestamp(time_range[0])
        end = pd.Timestamp(time_range[1])
        if index.tz is not None:
            start = start.tz_localize(index.tz) if start.tzinfo is None else start.tz_convert(index.tz)
            end = end.tz_localize(index.tz) if end.tzinfo is None else end.tz_convert(index.tz)
        return np.flatnonzero((index >= min(start, end)) & (index <= max(start, end)))

    @staticmethod
    def _point_indices_misaligned(step: RecipeStep, index: pd.DatetimeIndex) -> bool:
        has_indices = any(step.params.get(name) for name in POINT_INDEX_PARAM_NAMES)
        if not has_indices or step.sample_interval_seconds is None or len(index) < 2:
            return False
        target_interval = float((index[1] - index[0]).total_seconds())
        return not np.isclose(target_interval, step.sample_interval_seconds, rtol=0.0, atol=1e-9)
