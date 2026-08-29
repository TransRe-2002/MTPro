"""CompareWidget 的 LOD 异步构建 + E/B 路由 + 视口替换。"""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
from PySide6.QtWidgets import QApplication

from ui.compare_widget import CompareWidget

from test_mainwindow_refresh import DemoData


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _wait_display(widget, app, timeout: float = 6.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        widget_e = widget.widget_data.loc["E", "widget"]
        if widget._pyramids and widget_e is not None and widget_e.curves:
            return True
        time.sleep(0.02)
    return False


def test_compare_widget_builds_pyramids_and_routes_by_channel_prefix():
    _app()
    data = DemoData("S1", ["Ex", "Bx"], start="2024-01-01 00:00:00+00:00", values=[1.0, 2.0, 4.0, 8.0])
    widget = CompareWidget([(data, "Ex"), (data, "Bx")])
    widget.show()

    assert _wait_display(widget, _app()), "异步构建未完成"
    assert set(widget._pyramids.keys()) == {"S1 - Ex", "S1 - Bx"}

    e_widget = widget.widget_data.loc["E", "widget"]
    b_widget = widget.widget_data.loc["B", "widget"]
    assert "S1 - Ex" in e_widget.curves
    assert "S1 - Bx" in b_widget.curves
    widget.close()


def test_compare_widget_updates_curve_data_on_viewport_change():
    _app()
    data = DemoData("S1", ["Ex"], start="2024-01-01 00:00:00+00:00", values=[1.0, 2.0, 4.0, 8.0])
    widget = CompareWidget([(data, "Ex")])
    widget.show()
    assert _wait_display(widget, _app())

    e_widget = widget.widget_data.loc["E", "widget"]
    before = e_widget.curves["S1 - Ex"].getData()

    widget.x_view_size = pd.Timedelta(seconds=10)
    widget.x_view_start = pd.Timedelta(seconds=5)
    widget._on_viewport_changed()

    after = e_widget.curves["S1 - Ex"].getData()
    assert after is not None and len(after[0]) >= 1
    widget.close()