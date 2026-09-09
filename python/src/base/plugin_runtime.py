"""Qt-free plugin execution contract enforcement."""
from __future__ import annotations

import json
import logging
from typing import Any
from collections.abc import Callable, Mapping

import pandas as pd

from plugin_api import ChannelBlock, TransformResult, complete_params


logger = logging.getLogger(__name__)


class PluginProcessContext:
    def __init__(
        self,
        progress_callback: Callable[[float, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ):
        self._progress_callback = progress_callback
        self._cancelled = cancelled or (lambda: False)

    def progress(self, fraction: float, message: str = "") -> None:
        value = max(0.0, min(1.0, float(fraction)))
        if self._progress_callback is not None:
            self._progress_callback(value, str(message))

    def is_cancelled(self) -> bool:
        return bool(self._cancelled())

    def log(self, message: str) -> None:
        logger.info("plugin: %s", message)


def run_plugin_process(
    plugin: object,
    block: ChannelBlock,
    params: Mapping[str, Any],
    ctx=None,
) -> tuple[TransformResult, dict[str, Any]]:
    info = plugin.info
    values = complete_params(info, params)
    json.dumps(values)
    result = plugin.process(block, values, ctx or PluginProcessContext())
    if not isinstance(result, TransformResult):
        raise TypeError("plugin process() must return TransformResult")
    if not isinstance(result.series, pd.Series):
        raise TypeError("TransformResult.series must be a pandas Series")
    if len(result.series) != len(block.series):
        raise ValueError("plugin result length differs from input block")
    if not result.series.index.equals(block.series.index):
        raise ValueError("plugin result index differs from input block")
    return result, values
