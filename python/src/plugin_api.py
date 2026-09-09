"""MTPro data-processing plugin API (API version 1).

A plugin is a class with an ``info: PluginInfo`` class attribute and a
``process()`` method.  It may additionally provide ``create_editor()``.  This
module deliberately has no runtime Qt dependency so processors can be loaded
and executed by tests, scripts, and worker threads.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol
from collections.abc import Callable, Mapping

import pandas as pd

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget


API_VERSION = 1
SUPPORTED_CAPABILITIES = ("channel_sync", "region_sync")
SUPPORTED_PARAM_TYPES = (float, int, bool, str, list)


@dataclass(frozen=True)
class ParamSpec:
    """Declaration for one JSON-serializable plugin parameter."""

    name: str
    label: str
    type: type
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    options: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PluginInfo:
    """Static plugin metadata."""

    plugin_id: str
    display_name: str
    version: str
    api_version: int = API_VERSION
    category: str = "auto"
    description: str = ""
    capabilities: tuple[str, ...] = ()
    params_schema: tuple[ParamSpec, ...] = ()


@dataclass(frozen=True)
class ChannelMeta:
    chid: int = -1
    sample_rate: float = 0.0
    units: str = ""


@dataclass(frozen=True)
class ChannelBlock:
    """A read-only, single-channel input block passed to ``process()``."""

    channel: str
    series: pd.Series
    meta: ChannelMeta = field(default_factory=ChannelMeta)


@dataclass(frozen=True)
class TransformResult:
    """A transformed series plus optional diagnostics."""

    series: pd.Series
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class ProcessContext(Protocol):
    def progress(self, fraction: float, message: str = "") -> None: ...

    def is_cancelled(self) -> bool: ...

    def log(self, message: str) -> None: ...


class ProcessCancelled(Exception):
    """Raised by a processor after observing cooperative cancellation."""


class Processor(Protocol):
    info: PluginInfo

    def process(
        self,
        block: ChannelBlock,
        params: Mapping[str, Any],
        ctx: ProcessContext,
    ) -> TransformResult: ...


class EditorHost(Protocol):
    def apply_params(
        self,
        params: Mapping[str, Any],
        broadcast: frozenset[str] = frozenset(),
    ) -> None: ...

    def run_heavy(
        self,
        compute_fn: Callable[[], Any],
        apply_fn: Callable[[Any], None],
        description: str = "",
    ) -> None: ...

    def log(self, message: str) -> None: ...


class EditorFactory(Protocol):
    def create_editor(
        self,
        host: EditorHost,
        block: ChannelBlock,
        params: Mapping[str, Any],
    ) -> "QWidget" | None: ...


class PluginProcessor(Processor, EditorFactory, Protocol):
    """Convenience protocol for processors with custom editors."""


def default_params(info: PluginInfo) -> dict[str, Any]:
    """Return a fresh default parameter mapping for ``info``."""

    return {spec.name: deepcopy(spec.default) for spec in info.params_schema}


def complete_params(info: PluginInfo, params: Mapping[str, Any]) -> dict[str, Any]:
    """Fill omitted parameters with schema defaults, preserving extra keys."""

    values = default_params(info)
    values.update(dict(params))
    return values
