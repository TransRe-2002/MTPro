import sys
import types
import numpy as np
import pandas as pd
import xarray as xr

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QLabel

from base.data_manager import DataManager
from core.em_data import EMData
from ui.brief_plot import BOTTOM_AXIS_HEIGHT, LEFT_AXIS_WIDTH
from ui.data_tree_view import DataTreeViewer
from ui.data_view_widget import DataViewWidget

gfz_client_stub = types.ModuleType("gfz_client")
gfz_client_stub.GFZClient = object
sys.modules.setdefault("gfz_client", gfz_client_stub)

from ui.mainwindow import DATA_MANAGER_INITIAL_WIDTH, MainWindow


class DemoData(EMData):
    def __init__(self, name: str, channels: list[str], npts: int = 3):
        super().__init__("/tmp/demo.mat")
        self.name = name
        self.npts = npts
        self.start_time = pd.Timestamp("2024-01-01 00:00:00")
        self.dt = pd.Timedelta(seconds=5)
        self.datetime_index = pd.date_range(self.start_time, periods=npts, freq="5s")
        self.end_time = self.datetime_index[-1]
        self.kp_data = None
        self.latitude = 0.0
        self.longitude = 0.0
        self.e_units = "mV/km"
        self.m_units = "nT"
        self.data = xr.Dataset(coords={"time": self.datetime_index})
        for ch_name in channels:
            self.data[ch_name] = xr.DataArray(
                [float(i + 1) for i in range(npts)],
                dims=("time",),
                coords={"time": self.datetime_index},
            )

    def restore_data(self, ch: str):
        return None


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_uses_non_opaque_resize_for_primary_splitter(qtbot):
    _app()
    window = MainWindow(DataManager())
    qtbot.addWidget(window)

    assert window.workspace_splitter.opaqueResize() is False


def test_main_window_uses_narrow_data_manager_initial_width(qtbot):
    _app()
    window = MainWindow(DataManager())
    qtbot.addWidget(window)

    left_width, right_width = window.workspace_splitter.sizes()

    assert left_width < right_width
    assert DATA_MANAGER_INITIAL_WIDTH == 220


def test_data_view_widget_uses_non_opaque_resize_for_plot_splitter(qtbot):
    _app()
    widget = DataViewWidget()
    qtbot.addWidget(widget)

    assert widget.splitter.opaqueResize() is False


def test_data_view_waveform_axes_and_labels_survive_refresh():
    _app()
    data = DemoData("demo", ["Ex", "Ey", "Bx", "By", "Bz"], npts=900)
    widget = DataViewWidget()

    try:
        widget.init_data(data)
        QApplication.processEvents()

        plots = {resources.widget for resources in widget.plot_rows.values()}
        assert plots == {widget.plot_widget}
        plot = widget.plot_widget
        assert plot.getAxis("left").width() == LEFT_AXIS_WIDTH
        assert plot.getAxis("bottom").height() == BOTTOM_AXIS_HEIGHT
        assert set(plot.curves) == set(widget.plot_rows)

        for ch in widget.plot_rows:
            assert plot.curves[ch].isVisible()
        plot.clear_curves()
        for ch in widget.plot_rows:
            series = data.get_channel_series(ch)
            plot.add_curve(data.datetime_index, series.values, ch)
        plot.update_plot(data.start_time, data.end_time)

        assert plot.getAxis("left").width() == LEFT_AXIS_WIDTH
        assert plot.getAxis("bottom").height() == BOTTOM_AXIS_HEIGHT
    finally:
        widget.shutdown()
        widget.close()
        QApplication.processEvents()


def test_unknown_channels_are_available_but_not_checked_by_default():
    _app()
    data = DemoData("demo", ["Ex", "Err", "Hy"], npts=900)
    widget = DataViewWidget()

    try:
        widget.init_data(data)
        QApplication.processEvents()

        assert "Err" in widget._checkboxes
        assert not widget._checkboxes["Err"].isChecked()
        assert not widget.plot_widget.curves["Err"].isVisible()
        legend_names = {label.text for _, label in widget.plot_widget.plot_item.legend.items}
        assert "Err" not in legend_names

        widget._checkboxes["Err"].setChecked(True)
        QApplication.processEvents()

        assert widget.plot_widget.curves["Err"].isVisible()
        offsets = widget._stack_offsets
        assert offsets["Ex"] != offsets["Err"] != offsets["Hy"]
        legend_names = {label.text for _, label in widget.plot_widget.plot_item.legend.items}
        assert "Err" in legend_names
        for ch, offset in offsets.items():
            _, values = widget.plot_widget.curves[ch].getData()
            assert abs(float(np.nanmedian(values)) - offset) < 0.25
    finally:
        widget.shutdown()
        widget.close()
        QApplication.processEvents()


def test_main_window_switches_sidebar_between_summary_and_processing_tools(qtbot):
    _app()
    window = MainWindow(DataManager())
    qtbot.addWidget(window)

    window.tab_widget.setCurrentIndex(1)
    assert window.data_tree_viewer.summary_stack.currentWidget() is window.data_tree_viewer.tool_group

    window.tab_widget.setCurrentIndex(0)
    assert window.data_tree_viewer.summary_stack.currentWidget() is window.data_tree_viewer.info_group


def test_main_window_keeps_welcome_alive_after_entering_workspace(qtbot, tmp_path):
    _app()
    window = MainWindow(DataManager())
    qtbot.addWidget(window)

    window._continue_to_workspace()
    recent_path = tmp_path / "second.mat"
    window.app_state.recent_files = [str(recent_path)]
    window._refresh_recent_file_views()

    assert window.central_stack.currentWidget() is window.workspace_with_log_splitter
    assert window.welcome.recent_model.rowCount() == 1


def test_data_tree_viewer_reports_highlighted_channel(qtbot):
    _app()
    manager = DataManager()
    key = manager.add(DemoData("demo", ["Ex1", "Hy1"]))
    viewer = DataTreeViewer(manager)
    qtbot.addWidget(viewer)

    dataset_index = viewer.model.index_for_key(key)
    channel_index = viewer.model.index(1, 0, dataset_index)
    viewer.tree_view.setCurrentIndex(channel_index)

    assert viewer.current_dataset_key() == key
    assert viewer.current_channel_ref() == (key, "Hy1")


def test_data_tree_viewer_reports_dataset_without_channel(qtbot):
    _app()
    manager = DataManager()
    key = manager.add(DemoData("demo", ["Ex1", "Hy1"]))
    viewer = DataTreeViewer(manager)
    qtbot.addWidget(viewer)

    dataset_index = viewer.model.index_for_key(key)
    viewer.tree_view.setCurrentIndex(dataset_index)

    assert viewer.current_dataset_key() == key
    assert viewer.current_channel_ref() is None
