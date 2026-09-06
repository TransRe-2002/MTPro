"""active-only viewer 缓存策略 + 现场状态快照/恢复 + 异步显示构建。"""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
from PySide6.QtWidgets import QApplication

from base.data_manager import DataManager
from ui.mainwindow import MainWindow

from test_mainwindow_refresh import DemoData


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _spin(app, seconds: float = 1.5) -> None:
    """处理事件一小会儿，让 io 池的异步显示构建回到主线程。"""
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def _make_window():
    app = _app()
    dm = DataManager()
    mw = MainWindow(dm)
    return app, dm, mw


def _stop_broker(mw) -> None:
    # 旧处理窗口曾在初始化时把自己的线程停止回调接到了
    # aboutToQuit 上；测试退出前手动触发一次，避免 QThread 带运行销毁
    _app().aboutToQuit.emit()


def test_active_only_keeps_single_viewer_and_builds_async():
    app, dm, mw = _make_window()
    try:
        kA = dm.add(DemoData("A", ["Ex", "Hy"]))
        dm.add(DemoData("B", ["Ex", "Hy"]))

        dm.set_active(kA)
        _spin(app)

        # 只有一个 viewer，且异步构建已完成（金字塔与曲线就位）
        assert list(mw.data_viewers.keys()) == [kA]
        viewer = mw.data_viewers[kA]
        assert set(viewer._pyramids.keys()) == {"Ex", "Hy"}
        assert "Ex" in viewer.plot_rows["Ex"].widget.curves
    finally:
        _stop_broker(mw)


def test_switch_captures_state_and_destroys_old_viewer():
    app, dm, mw = _make_window()
    try:
        kA = dm.add(DemoData("A", ["Ex", "Hy"]))
        kB = dm.add(DemoData("B", ["Ex", "Hy"]))

        dm.set_active(kA)
        _spin(app)
        viewer_a = mw.data_viewers[kA]
        viewer_a.x_view_size = pd.Timedelta(seconds=5)
        viewer_a.x_view_start = pd.Timedelta(seconds=5)

        dm.set_active(kB)
        _spin(app)

        # 旧 viewer 已销毁，现场已快照
        assert list(mw.data_viewers.keys()) == [kB]
        snap = mw.viewer_state[kA]
        assert snap["x_view_start"] == pd.Timedelta(seconds=5)
        assert snap["x_view_size"] == pd.Timedelta(seconds=5)
        assert snap["channel_order"] == ["Ex", "Hy"]
    finally:
        _stop_broker(mw)


def test_reactivation_restores_viewport_and_rebuilds_pyramids():
    app, dm, mw = _make_window()
    try:
        kA = dm.add(DemoData("A", ["Ex", "Hy"]))
        kB = dm.add(DemoData("B", ["Ex", "Hy"]))

        dm.set_active(kA)
        _spin(app)
        first = mw.data_viewers[kA]
        first.x_view_size = pd.Timedelta(seconds=5)
        first.x_view_start = pd.Timedelta(seconds=5)

        dm.set_active(kB)
        _spin(app)
        dm.set_active(kA)
        _spin(app)

        second = mw.data_viewers[kA]
        assert second is not first            # 重建的新 viewer
        assert second.x_view_start == pd.Timedelta(seconds=5)
        assert second.x_view_size == pd.Timedelta(seconds=5)
        assert set(second._pyramids.keys()) == {"Ex", "Hy"}
    finally:
        _stop_broker(mw)


def test_data_removed_drops_saved_view_state():
    app, dm, mw = _make_window()
    try:
        kA = dm.add(DemoData("A", ["Ex", "Hy"]))
        kB = dm.add(DemoData("B", ["Ex", "Hy"]))

        dm.set_active(kA)
        _spin(app)
        dm.set_active(kB)
        _spin(app)
        assert kA in mw.viewer_state

        dm.remove(kA)
        _spin(app)
        assert kA not in mw.viewer_state
    finally:
        _stop_broker(mw)
