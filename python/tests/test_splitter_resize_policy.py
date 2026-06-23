import sys
import types
import pandas as pd
import xarray as xr

from PySide6.QtWidgets import QApplication

from base.data_manager import DataManager
from core.em_data import EMData
from ui.data_tree_view import DataTreeViewer
from ui.data_view_widget import DataViewWidget

gfz_client_stub = types.ModuleType("gfz_client")
gfz_client_stub.GFZClient = object
sys.modules.setdefault("gfz_client", gfz_client_stub)

from ui.mainwindow import DATA_MANAGER_INITIAL_WIDTH, MainWindow


class DemoData(EMData):
    def __init__(self, name: str, channels: list[str]):
        super().__init__("/tmp/demo.mat")
        self.name = name
        self.npts = 3
        self.start_time = pd.Timestamp("2024-01-01 00:00:00")
        self.end_time = pd.Timestamp("2024-01-01 00:00:10")
        self.dt = pd.Timedelta(seconds=5)
        self.datetime_index = pd.date_range(self.start_time, periods=3, freq="5s")
        self.kp_data = None
        self.latitude = 0.0
        self.longitude = 0.0
        self.e_units = "mV/km"
        self.m_units = "nT"
        self.data = xr.Dataset(coords={"time": self.datetime_index})
        for ch_name in channels:
            self.data[ch_name] = xr.DataArray(
                [1.0, 2.0, 3.0],
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
