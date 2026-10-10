from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox

from base.data_manager import DataManager
from ui.mainwindow import MainWindow


def test_main_window_batches_multiple_recent_files_once(qtbot, tmp_path, monkeypatch):
    first = tmp_path / "first.mat"
    second = tmp_path / "second.mat"
    first.touch()
    second.touch()

    window = MainWindow(DataManager())
    qtbot.addWidget(window)
    prompted: list[list[str]] = []
    opened: list[tuple[list[str], str]] = []

    def prompt_timezone(filenames: list[str]) -> bool:
        prompted.append(filenames)
        window._batch_tz = "Asia/Shanghai"
        return True

    monkeypatch.setattr(window, "_prompt_timezone_for_files", prompt_timezone)
    monkeypatch.setattr(
        window,
        "_open_files_batch",
        lambda filenames, tz: opened.append((filenames, tz)),
    )

    window._open_recent_files([str(first), str(second), str(first)])

    expected = [str(first.resolve()), str(second.resolve())]
    assert prompted == [expected]
    assert opened == [(expected, "Asia/Shanghai")]


def test_main_window_uses_single_file_flow_for_one_recent_file(
    qtbot,
    tmp_path,
    monkeypatch,
):
    filename = tmp_path / "single.mat"
    filename.touch()
    window = MainWindow(DataManager())
    qtbot.addWidget(window)
    opened: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        window,
        "_open_file_path",
        lambda path, from_recent=False: opened.append((path, from_recent)),
    )

    window._open_recent_file(str(filename))

    assert opened == [(str(Path(filename).resolve()), True)]


def test_main_window_directs_recent_txt_to_import_wizard(
    qtbot,
    tmp_path,
    monkeypatch,
):
    filename = tmp_path / "legacy.txt"
    filename.touch()
    window = MainWindow(DataManager())
    qtbot.addWidget(window)
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    monkeypatch.setattr(
        window,
        "_open_file_path",
        lambda *_args, **_kwargs: pytest.fail("TXT must not use direct open"),
    )

    window._open_recent_file(str(filename))

    assert warnings
    assert "TXT 请使用“从 MT TXT 文件导入”" in warnings[0][1]
