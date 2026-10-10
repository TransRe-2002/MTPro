from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from collections.abc import Callable

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
    output_factory: Callable[..., QWidget] | None = None
    pipeline_editor_factory: Callable[..., QWidget] | None = None


_REGISTRY: dict[str, ToolSpec] = {}
_PLUGINS: dict[str, object] = {}

_COMPAT_ALIASES = {
    "remove_step_diff": ("remove step", "remove_step"),
    "remove_wavelet_denoise": ("remove pulse wavelet",),
}


def register(spec: ToolSpec) -> None:
    keys = (spec.tool_id, spec.display_name, *spec.aliases)
    for key in keys:
        _REGISTRY[key] = spec


def toolspec_from_plugin(loaded_plugin) -> ToolSpec:
    """Adapt a validated plugin instance to the existing tool registry."""

    from ui.widgets.plugin_editor import create_pipeline_plugin_editor

    info = loaded_plugin.info
    has_editor = (
        callable(getattr(loaded_plugin.plugin, "create_editor", None))
        or bool(getattr(loaded_plugin.plugin, "pipeline_preview", False))
    )
    fields = tuple(
        (param.name, param.label, param.type, param.default)
        for param in info.params_schema
    )
    return ToolSpec(
        tool_id=info.plugin_id,
        display_name=info.display_name,
        category=info.category,
        kind="interactive" if has_editor else "param",
        requires_channel=True,
        aliases=_COMPAT_ALIASES.get(info.plugin_id, ()),
        fields=fields,
        has_channel_sync="channel_sync" in info.capabilities,
        has_region_sync="region_sync" in info.capabilities,
        pipeline_editor_factory=partial(create_pipeline_plugin_editor, loaded_plugin.plugin),
    )


def register_loaded_plugins(loaded_plugins) -> None:
    for loaded_plugin in loaded_plugins:
        _PLUGINS[loaded_plugin.info.plugin_id] = loaded_plugin.plugin
        register(toolspec_from_plugin(loaded_plugin))


def get_plugin(key: str):
    spec = get(key)
    if spec is None:
        return None
    return _PLUGINS.get(spec.tool_id)


def get(key: str) -> ToolSpec | None:
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


def create_output_widget(
    key: str,
    data_id: int,
) -> QWidget | None:
    """Create a station-level output tool without a processing data path."""
    spec = _REGISTRY.get(key)
    if spec is None or spec.requires_channel or spec.output_factory is None:
        return None
    return spec.output_factory(data_id)


def create_pipeline_editor(
    key: str,
    preview_channel,
    parent,
    params: dict | None = None,
) -> QWidget | None:
    """为 pipeline block 创建交互式编辑器，并同步注入 block 切片数据。

    ``preview_channel`` 是 ``PreviewChannelSlice(name, cts)``，其中 ``cts`` 是
    该 block 区间 ``[start:end]`` 的 ``pd.Series``。channel 工具使用
    ``(data_id, channel_name)`` 构造签名创建（data_id 用 0 占位），随后通过
    ``load_series()`` 把切片直接灌进去。

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
    if params is not None and hasattr(widget, "params"):
        widget.params.update(params)
    loader = getattr(widget, "load_series", None)
    if callable(loader):
        loader(preview_channel.name, preview_channel.cts)
    return widget


import ui.tools.registrations  # noqa: F401,E402  registers built-in tools

# Keep direct DataProcessWidget/test construction useful outside MainWindow.
from base.plugin_loader import load_all_plugins  # noqa: E402

_BUILTIN_REPORT = load_all_plugins(scan_paths=[], include_entry_points=False)
register_loaded_plugins(_BUILTIN_REPORT.loaded)
