from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
import tomllib


APP_NAME = "MTPro"
APP_VERSION = "0.1.0"


@dataclass
class AppState:
    app_name: str = APP_NAME
    app_version: str = APP_VERSION
    kp_hdf5_path: str = ""
    last_open_dir: str = ""
    active_tab: str = "波形浏览"
    workspace_splitter_sizes: list[int] = field(default_factory=list)
    workspace_with_log_splitter_sizes: list[int] = field(default_factory=list)
    log_visible: bool = True
    recent_files: list[str] = field(default_factory=list)
    plugin_paths: list[str] = field(default_factory=list)


def default_app_state_path() -> Path:
    env_path = os.environ.get("MTPRO_APP_STATE_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser()

    if sys.platform.startswith("win"):
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        app_data = os.environ.get("APPDATA", "").strip()
        if local_app_data:
            base_dir = Path(local_app_data).expanduser()
        elif app_data:
            base_dir = Path(app_data).expanduser()
        else:
            base_dir = Path.home() / "AppData" / "Local"
    else:
        xdg_config_home = os.environ.get("XDG_CONFIG_HOME", "").strip()
        if xdg_config_home:
            base_dir = Path(xdg_config_home).expanduser()
        else:
            base_dir = Path.home() / ".config"
    return base_dir / "mtpro" / "app_state.toml"


def _normalize_path(value: str) -> str:
    if not value:
        return ""
    return str(Path(value).expanduser().resolve(strict=False))


def _default_kp_hdf5_path(state_path: Path) -> str:
    return str(state_path.with_name("kp.h5"))


def _normalize_recent_files(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        path = _normalize_path(value)
        if not path or path in seen:
            continue
        normalized.append(path)
        seen.add(path)
    return normalized


def _normalize_int_list(values: list[object]) -> list[int]:
    result: list[int] = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _normalize_path_list(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        path = _normalize_path(value)
        if not path or path in seen:
            continue
        normalized.append(path)
        seen.add(path)
    return normalized


def load_app_state(path: Path | None = None) -> AppState:
    state_path = default_app_state_path() if path is None else Path(path).expanduser()
    default_state = AppState(kp_hdf5_path=_default_kp_hdf5_path(state_path))

    if not state_path.exists():
        return default_state

    try:
        with state_path.open("rb") as fh:
            raw = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return default_state

    app_section = raw.get("app", {})
    paths_section = raw.get("paths", {})
    recent_section = raw.get("recent", {})

    app_name = app_section.get("name", default_state.app_name)
    app_version = app_section.get("version", default_state.app_version)
    kp_hdf5_path = _normalize_path(paths_section.get("kp_hdf5", "")) or default_state.kp_hdf5_path
    last_open_dir = _normalize_path(paths_section.get("last_open_dir", ""))
    active_tab = str(app_section.get("active_tab", default_state.active_tab))
    log_visible = bool(app_section.get("log_visible", default_state.log_visible))
    workspace_splitter_sizes_raw = app_section.get("workspace_splitter_sizes", [])
    workspace_with_log_splitter_sizes_raw = app_section.get("workspace_with_log_splitter_sizes", [])
    recent_files_raw = recent_section.get("files", [])
    recent_files = recent_files_raw if isinstance(recent_files_raw, list) else []
    plugin_paths_raw = paths_section.get("plugins", [])
    plugin_paths = plugin_paths_raw if isinstance(plugin_paths_raw, list) else []

    return AppState(
        app_name=str(app_name),
        app_version=str(app_version),
        kp_hdf5_path=kp_hdf5_path,
        last_open_dir=last_open_dir,
        active_tab=active_tab,
        workspace_splitter_sizes=_normalize_int_list(
            workspace_splitter_sizes_raw if isinstance(workspace_splitter_sizes_raw, list) else []
        ),
        workspace_with_log_splitter_sizes=_normalize_int_list(
            workspace_with_log_splitter_sizes_raw if isinstance(workspace_with_log_splitter_sizes_raw, list) else []
        ),
        log_visible=log_visible,
        recent_files=_normalize_recent_files([str(item) for item in recent_files]),
        plugin_paths=_normalize_path_list([str(item) for item in plugin_paths]),
    )


def save_app_state(state: AppState, path: Path | None = None) -> Path:
    state_path = default_app_state_path() if path is None else Path(path).expanduser()
    normalized_state = AppState(
        app_name=state.app_name or APP_NAME,
        app_version=state.app_version or APP_VERSION,
        kp_hdf5_path=_normalize_path(state.kp_hdf5_path) or _default_kp_hdf5_path(state_path),
        last_open_dir=_normalize_path(state.last_open_dir),
        active_tab=state.active_tab or "波形浏览",
        workspace_splitter_sizes=_normalize_int_list(state.workspace_splitter_sizes),
        workspace_with_log_splitter_sizes=_normalize_int_list(state.workspace_with_log_splitter_sizes),
        log_visible=bool(state.log_visible),
        recent_files=_normalize_recent_files(state.recent_files),
        plugin_paths=_normalize_path_list(state.plugin_paths),
    )

    lines = [
        "[app]",
        f"name = {json.dumps(normalized_state.app_name, ensure_ascii=False)}",
        f"version = {json.dumps(normalized_state.app_version, ensure_ascii=False)}",
        f"active_tab = {json.dumps(normalized_state.active_tab, ensure_ascii=False)}",
        f"log_visible = {json.dumps(normalized_state.log_visible)}",
        f"workspace_splitter_sizes = {json.dumps(normalized_state.workspace_splitter_sizes, ensure_ascii=False)}",
        f"workspace_with_log_splitter_sizes = {json.dumps(normalized_state.workspace_with_log_splitter_sizes, ensure_ascii=False)}",
        "",
        "[paths]",
        f"kp_hdf5 = {json.dumps(normalized_state.kp_hdf5_path, ensure_ascii=False)}",
        f"last_open_dir = {json.dumps(normalized_state.last_open_dir, ensure_ascii=False)}",
        f"plugins = {json.dumps(normalized_state.plugin_paths, ensure_ascii=False)}",
        "",
        "[recent]",
        f"files = {json.dumps(normalized_state.recent_files, ensure_ascii=False)}",
        "",
    ]
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError:
        return state_path
    return state_path


def remember_recent_file(
    state: AppState,
    filename: str,
) -> AppState:
    normalized = _normalize_path(filename)
    recent_files = [normalized] + [path for path in state.recent_files if _normalize_path(path) != normalized]
    state.recent_files = _normalize_recent_files(recent_files)
    state.last_open_dir = str(Path(normalized).parent)
    return state


def remove_recent_file(state: AppState, filename: str) -> AppState:
    normalized = _normalize_path(filename)
    state.recent_files = [path for path in state.recent_files if _normalize_path(path) != normalized]
    return state
