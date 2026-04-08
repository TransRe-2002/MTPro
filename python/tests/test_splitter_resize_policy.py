import sys
import types

from PySide6.QtWidgets import QApplication

from base.data_manager import DataManager
from ui.data_view_widget import DataViewWidget

gfz_client_stub = types.ModuleType("gfz_client")
gfz_client_stub.GFZClient = object
sys.modules.setdefault("gfz_client", gfz_client_stub)

from ui.mainwindow import MainWindow


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_uses_non_opaque_resize_for_primary_splitter(qtbot):
    _app()
    window = MainWindow(DataManager())
    qtbot.addWidget(window)

    assert window.vertical_splitter.opaqueResize() is False


def test_data_view_widget_uses_non_opaque_resize_for_plot_splitter(qtbot):
    _app()
    widget = DataViewWidget()
    qtbot.addWidget(widget)

    assert widget.splitter.opaqueResize() is False
