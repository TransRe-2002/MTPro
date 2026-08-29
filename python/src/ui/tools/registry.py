from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from PySide6.QtWidgets import QWidget


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    display_name: str
    category: str
    kind: str
    requires_channel: bool
    aliases: tuple[str, ...] = ()
    fields: tuple[tuple[str, str, type, object], ...] = ()
    has_channel_sync: bool = False
    has_region_sync: bool = False
    classic_factory: Optional[Callable[..., QWidget]] = None
    pipeline_editor_factory: Optional[Callable[..., QWidget]] = None


_REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> None:
    keys = (spec.tool_id, spec.display_name, *spec.aliases)
    for key in keys:
        _REGISTRY[key] = spec


def get(key: str) -> Optional[ToolSpec]:
    return _REGISTRY.get(key)


def all_specs() -> list[ToolSpec]:
    seen: set[str] = set()
    result: list[ToolSpec] = []
    for spec in _REGISTRY.values():
        if spec.tool_id not in seen:
            seen.add(spec.tool_id)
            result.append(spec)
    return result


def canonical_id(key: str) -> str:
    spec = _REGISTRY.get(key)
    return spec.tool_id if spec is not None else key


def canonical_display_name(key: str) -> str:
    spec = _REGISTRY.get(key)
    return spec.display_name if spec is not None else key


def is_station_tool(key: str) -> bool:
    spec = _REGISTRY.get(key)
    return spec is not None and not spec.requires_channel


def create_classic_widget(
    key: str,
    data_id: int,
    channel_name: Optional[str],
) -> Optional[QWidget]:
    spec = _REGISTRY.get(key)
    if spec is None or spec.classic_factory is None:
        return None
    if spec.requires_channel:
        return spec.classic_factory(data_id, channel_name)
    return spec.classic_factory(data_id)


def create_pipeline_editor(
    key: str,
    preview_channel,
    parent,
) -> Optional[QWidget]:
    """为 pipeline block 创建交互式编辑器，并同步注入 block 切片数据。

    ``preview_channel`` 是 ``PreviewChannelSlice(name, cts)``，其中 ``cts`` 是
    该 block 区间 ``[start:end]`` 的 ``pd.Series``。channel 工具复用 classic
    构造签名 ``(data_id, channel_name)`` 创建（data_id 用 0 占位），随后通过
    ``load_series()`` 把切片直接灌进去——不再经过 broker 异步请求。

    构造时不传 parent（避免触发子类的独立窗口 Window flag），由调用方
    ``layout.addWidget()`` 负责重定父。
    """
    spec = _REGISTRY.get(key)
    if spec is None or spec.pipeline_editor_factory is None:
        return None
    factory = spec.pipeline_editor_factory
    if spec.requires_channel:
        widget = factory(0, preview_channel.name)
    else:
        widget = factory(0)
    loader = getattr(widget, "load_series", None)
    if callable(loader):
        loader(preview_channel.name, preview_channel.cts)
    return widget


import ui.tools._registrations  # noqa: F401,E402  registers built-in tools