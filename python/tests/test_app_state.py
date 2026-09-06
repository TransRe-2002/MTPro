from __future__ import annotations

from pathlib import Path

import utils.app_state as app_state_module
from utils.app_state import (
    APP_NAME,
    APP_VERSION,
    AppState,
    default_app_state_path,
    load_app_state,
    remember_recent_file,
    remove_recent_file,
    save_app_state,
)


def test_app_state_roundtrip_toml(tmp_path):
    state_path = tmp_path / "app_state.toml"
    state = AppState(
        app_name=APP_NAME,
        app_version=APP_VERSION,
        kp_hdf5_path=str(tmp_path / "cache" / "kp.h5"),
        last_open_dir=str(tmp_path / "data"),
        active_tab="数据处理",
        workspace_splitter_sizes=[240, 1200],
        workspace_with_log_splitter_sizes=[800, 120],
        log_visible=False,
        recent_files=[
            str(tmp_path / "a.mat"),
            str(tmp_path / "b.mat"),
        ],
        plugin_paths=[str(tmp_path / "plugins"), str(tmp_path / "plugins")],
    )

    save_app_state(state, state_path)
    loaded = load_app_state(state_path)

    assert loaded.app_name == APP_NAME
    assert loaded.app_version == APP_VERSION
    assert loaded.kp_hdf5_path == str((tmp_path / "cache" / "kp.h5").resolve(strict=False))
    assert loaded.last_open_dir == str((tmp_path / "data").resolve(strict=False))
    assert loaded.active_tab == "数据处理"
    assert loaded.workspace_splitter_sizes == [240, 1200]
    assert loaded.workspace_with_log_splitter_sizes == [800, 120]
    assert loaded.log_visible is False
    assert loaded.recent_files == [
        str((tmp_path / "a.mat").resolve(strict=False)),
        str((tmp_path / "b.mat").resolve(strict=False)),
    ]
    assert loaded.plugin_paths == [str((tmp_path / "plugins").resolve(strict=False))]


def test_remember_recent_file_deduplicates_and_moves_to_front(tmp_path):
    first = str((tmp_path / "first.mat").resolve(strict=False))
    second = str((tmp_path / "second.mat").resolve(strict=False))
    state = AppState(recent_files=[first, second], last_open_dir=str(tmp_path))

    remember_recent_file(state, first)

    assert state.recent_files == [first, second]
    assert state.last_open_dir == str(tmp_path.resolve(strict=False))


def test_remove_recent_file_drops_missing_path(tmp_path):
    first = str((tmp_path / "first.mat").resolve(strict=False))
    second = str((tmp_path / "second.mat").resolve(strict=False))
    state = AppState(recent_files=[first, second])

    remove_recent_file(state, first)

    assert state.recent_files == [second]


def test_default_app_state_path_uses_xdg_config_home(monkeypatch, tmp_path):
    monkeypatch.delenv("MTPRO_APP_STATE_PATH", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(app_state_module.sys, "platform", "linux")

    path = default_app_state_path()

    assert path == (tmp_path / "xdg" / "mtpro" / "app_state.toml")


def test_default_app_state_path_uses_localappdata_on_windows(monkeypatch, tmp_path):
    monkeypatch.delenv("MTPRO_APP_STATE_PATH", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setattr(app_state_module.sys, "platform", "win32")

    path = default_app_state_path()

    assert path == (tmp_path / "LocalAppData" / "mtpro" / "app_state.toml")
