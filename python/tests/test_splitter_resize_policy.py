import sys
import types
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

        for ch, resources in widget.plot_rows.items():
            plot = resources.widget
            container = resources.row_container
            assert plot is not None
            assert container is not None

            assert plot.getAxis("left").width() == LEFT_AXIS_WIDTH
            assert plot.getAxis("bottom").height() == BOTTOM_AXIS_HEIGHT
            assert plot.getAxis("bottom").labelText == ch
            direct_labels = container.findChildren(
                QLabel,
                options=Qt.FindChildOption.FindDirectChildrenOnly,
            )
            assert ch not in [label.text() for label in direct_labels]

        for ch, resources in widget.plot_rows.items():
            plot = resources.widget
            assert plot is not None
            series = data.get_channel_series(ch)
            if ch == "Ex":
                plot.plot_item.showAxis("left", False)
                plot.plot_item.showAxis("bottom", False)
            plot.clear_curves()
            plot.add_curve(data.datetime_index, series.values, ch)
            plot.update_plot(data.start_time, data.end_time)

        for ch, resources in widget.plot_rows.items():
            plot = resources.widget
            assert plot is not None

            assert plot.getAxis("left").width() == LEFT_AXIS_WIDTH
            assert plot.getAxis("bottom").height() == BOTTOM_AXIS_HEIGHT
            assert plot.getAxis("bottom").labelText == ch
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
