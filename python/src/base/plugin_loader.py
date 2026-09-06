"""Discovery, validation, and failure isolation for MTPro plugins."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import tomllib
from typing import Any, Iterable, Iterator, Mapping
import zipfile

from plugin_api import (
    API_VERSION,
    SUPPORTED_CAPABILITIES,
    SUPPORTED_PARAM_TYPES,
    ParamSpec,
    PluginInfo,
)
from utils.app_state import AppState, default_app_state_path, load_app_state


ENTRY_POINT_GROUP = "mtpro.plugins"
PLUGIN_PATH_ENV = "MTPRO_PLUGIN_PATH"


@dataclass(frozen=True)
class LoadFailure:
    source: str
    plugin_id: str
    reason: str


@dataclass(frozen=True)
class LoadedPlugin:
    plugin: object
    info: PluginInfo
    source: str


@dataclass
class LoadReport:
    loaded: list[LoadedPlugin] = field(default_factory=list)
    failed: list[LoadFailure] = field(default_factory=list)

    def by_id(self) -> dict[str, LoadedPlugin]:
        return {item.info.plugin_id: item for item in self.loaded}


@dataclass(frozen=True)
class _Manifest:
    plugin_id: str
    name: str
    version: str
    api: int
    entry: str


def default_plugin_dir() -> Path:
    return default_app_state_path().parent / "plugins"


def configured_plugin_paths(state: AppState | None = None) -> list[Path]:
    """Resolve scan roots in precedence order and remove duplicates."""

    values: list[str] = []
    env_value = os.environ.get(PLUGIN_PATH_ENV, "")
    if env_value:
        values.extend(part for part in env_value.split(os.pathsep) if part.strip())
    actual_state = load_app_state() if state is None else state
    values.extend(actual_state.plugin_paths)
    values.append(str(default_plugin_dir()))

    paths: list[Path] = []
    seen: set[str] = set()
    for value in values:
        path = Path(value).expanduser().resolve(strict=False)
        key = os.path.normcase(str(path))
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def validate_plugin_info(info: PluginInfo, expected_id: str = "") -> None:
    if not isinstance(info, PluginInfo):
        raise TypeError("plugin info must be a PluginInfo instance")
    if not info.plugin_id.strip():
        raise ValueError("plugin_id must not be empty")
    if expected_id and info.plugin_id != expected_id:
        raise ValueError(
            f"manifest id {expected_id!r} does not match PluginInfo.plugin_id {info.plugin_id!r}"
        )
    if not info.display_name.strip():
        raise ValueError("display_name must not be empty")
    if not info.version.strip():
        raise ValueError("version must not be empty")
    if info.api_version != API_VERSION:
        raise ValueError(
            f"unsupported API version {info.api_version}; host supports {API_VERSION}"
        )
    if info.category not in {"manual", "auto"}:
        raise ValueError("category must be 'manual' or 'auto'")
    unsupported = set(info.capabilities) - set(SUPPORTED_CAPABILITIES)
    if unsupported:
        raise ValueError(f"unsupported capabilities: {sorted(unsupported)!r}")
    if len(info.capabilities) != len(set(info.capabilities)):
        raise ValueError("capabilities must not contain duplicates")

    names: set[str] = set()
    for spec in info.params_schema:
        _validate_param_spec(spec)
        if spec.name in names:
            raise ValueError(f"duplicate parameter name: {spec.name}")
        names.add(spec.name)


def _validate_param_spec(spec: ParamSpec) -> None:
    if not isinstance(spec, ParamSpec):
        raise TypeError("params_schema entries must be ParamSpec instances")
    if not spec.name.strip() or not spec.label.strip():
        raise ValueError("parameter name and label must not be empty")
    if spec.type not in SUPPORTED_PARAM_TYPES:
        raise ValueError(f"unsupported parameter type for {spec.name}: {spec.type!r}")
    if spec.minimum is not None and spec.maximum is not None and spec.minimum > spec.maximum:
        raise ValueError(f"minimum exceeds maximum for parameter {spec.name}")
    if spec.options is not None and spec.type is not str:
        raise ValueError(f"options are only valid for str parameter {spec.name}")
    try:
        json.dumps(spec.default)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"default for parameter {spec.name} is not JSON serializable") from exc


def load_all_plugins(
    *,
    state: AppState | None = None,
    scan_paths: Iterable[str | Path] | None = None,
    builtin_classes: Iterable[type] | None = None,
    include_entry_points: bool = True,
) -> LoadReport:
    """Load every plugin source without allowing one failure to abort startup."""

    report = LoadReport()
    seen_ids: set[str] = set()

    if builtin_classes is None:
        try:
            from plugins import BUILTIN_PLUGINS
            builtin_classes = BUILTIN_PLUGINS
        except Exception as exc:
            report.failed.append(LoadFailure("builtin", "", repr(exc)))
            builtin_classes = ()
    for index, plugin_class in enumerate(builtin_classes):
        source = f"builtin:{getattr(plugin_class, '__name__', index)}"
        _try_load_class(plugin_class, source, report, seen_ids)

    if include_entry_points:
        _load_entry_points(report, seen_ids)

    roots = (
        [Path(path).expanduser().resolve(strict=False) for path in scan_paths]
        if scan_paths is not None
        else configured_plugin_paths(state)
    )
    for root in roots:
        _load_scan_root(root, report, seen_ids)
    return report


def _load_entry_points(report: LoadReport, seen_ids: set[str]) -> None:
    try:
        entry_points = importlib.metadata.entry_points()
        selected = entry_points.select(group=ENTRY_POINT_GROUP)
    except Exception as exc:
        report.failed.append(LoadFailure("entry_points", "", repr(exc)))
        return
    for entry_point in selected:
        source = f"entry_point:{entry_point.name}"
        try:
            plugin_class = entry_point.load()
            _try_load_class(plugin_class, source, report, seen_ids)
        except Exception as exc:
            report.failed.append(LoadFailure(source, "", repr(exc)))


def _load_scan_root(root: Path, report: LoadReport, seen_ids: set[str]) -> None:
    if not root.exists():
        return
    if not root.is_dir():
        report.failed.append(LoadFailure(f"dir:{root}", "", "scan root is not a directory"))
        return
    try:
        candidates = sorted(root.iterdir(), key=lambda path: path.name.casefold())
    except OSError as exc:
        report.failed.append(LoadFailure(f"dir:{root}", "", repr(exc)))
        return

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "plugin.toml").is_file():
            _load_directory_candidate(candidate, report, seen_ids)
        elif candidate.is_file() and candidate.suffix.casefold() == ".zip":
            _load_zip_candidate(candidate, report, seen_ids)


def _load_directory_candidate(
    candidate: Path,
    report: LoadReport,
    seen_ids: set[str],
) -> None:
    source = f"dir:{candidate}"
    plugin_id = ""
    try:
        with (candidate / "plugin.toml").open("rb") as fh:
            raw_manifest = tomllib.load(fh)
        plugin_id = _manifest_id_hint(raw_manifest)
        manifest = _parse_manifest(raw_manifest)
        plugin_id = manifest.plugin_id
        with _temporary_sys_path(candidate):
            plugin_class = _import_entry(manifest.entry)
        _try_load_class(plugin_class, source, report, seen_ids, manifest)
    except Exception as exc:
        report.failed.append(LoadFailure(source, plugin_id, repr(exc)))


def _load_zip_candidate(
    candidate: Path,
    report: LoadReport,
    seen_ids: set[str],
) -> None:
    source = f"zip:{candidate}"
    plugin_id = ""
    try:
        with zipfile.ZipFile(candidate) as archive:
            try:
                raw = archive.read("plugin.toml")
            except KeyError as exc:
                raise ValueError("zip root does not contain plugin.toml") from exc
        raw_manifest = tomllib.loads(raw.decode("utf-8"))
        plugin_id = _manifest_id_hint(raw_manifest)
        manifest = _parse_manifest(raw_manifest)
        plugin_id = manifest.plugin_id
        with _temporary_sys_path(candidate):
            plugin_class = _import_entry(manifest.entry)
        _try_load_class(plugin_class, source, report, seen_ids, manifest)
    except Exception as exc:
        report.failed.append(LoadFailure(source, plugin_id, repr(exc)))


def _parse_manifest(raw: Mapping[str, Any]) -> _Manifest:
    section = raw.get("plugin")
    if not isinstance(section, Mapping):
        raise ValueError("manifest must contain a [plugin] table")
    required = ("id", "name", "version", "api", "entry")
    missing = [name for name in required if name not in section]
    if missing:
        raise ValueError(f"manifest is missing fields: {', '.join(missing)}")
    api = int(section["api"])
    if api != API_VERSION:
        raise ValueError(f"unsupported manifest API version {api}; host supports {API_VERSION}")
    entry = str(section["entry"]).strip()
    if ":" not in entry:
        raise ValueError("entry must use 'module:ClassName' syntax")
    return _Manifest(
        plugin_id=str(section["id"]).strip(),
        name=str(section["name"]).strip(),
        version=str(section["version"]).strip(),
        api=api,
        entry=entry,
    )


def _manifest_id_hint(raw: Mapping[str, Any]) -> str:
    section = raw.get("plugin")
    if not isinstance(section, Mapping):
        return ""
    return str(section.get("id", "")).strip()


def _import_entry(entry: str) -> object:
    module_name, attribute = entry.split(":", 1)
    if not module_name.strip() or not attribute.strip():
        raise ValueError("entry must use 'module:ClassName' syntax")
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def _try_load_class(
    plugin_class: object,
    source: str,
    report: LoadReport,
    seen_ids: set[str],
    manifest: _Manifest | None = None,
) -> None:
    plugin_id = manifest.plugin_id if manifest is not None else ""
    try:
        if not inspect.isclass(plugin_class):
            raise TypeError("plugin entry must be a class")
        plugin = plugin_class()
        info = getattr(plugin, "info", None)
        validate_plugin_info(info, plugin_id)
        plugin_id = info.plugin_id
        if manifest is not None:
            if manifest.api != info.api_version:
                raise ValueError("manifest and PluginInfo API versions do not match")
            if manifest.name != info.display_name:
                raise ValueError("manifest name does not match PluginInfo.display_name")
            if manifest.version != info.version:
                raise ValueError("manifest version does not match PluginInfo.version")
        if not callable(getattr(plugin, "process", None)):
            raise TypeError("plugin process attribute must be callable")
        create_editor = getattr(plugin, "create_editor", None)
        if create_editor is not None and not callable(create_editor):
            raise TypeError("plugin create_editor attribute must be callable")
        if plugin_id in seen_ids:
            raise ValueError(f"duplicate plugin_id: {plugin_id}")
        seen_ids.add(plugin_id)
        report.loaded.append(LoadedPlugin(plugin, info, source))
    except Exception as exc:
        report.failed.append(LoadFailure(source, plugin_id, repr(exc)))


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:
            pass
