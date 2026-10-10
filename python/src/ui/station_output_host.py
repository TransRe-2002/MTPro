from __future__ import annotations


from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QMdiArea, QMdiSubWindow, QWidget

from base.data_manager import DataManager
from output_tools.station_data import station_data_package
from ui.tools import registry as tool_registry


def _mdi_target_size(
    mdi_area: QMdiArea,
    min_width: int = 900,
    min_height: int = 680,
) -> tuple[int, int]:
    viewport_size = mdi_area.viewport().size()
    available_width = viewport_size.width() or mdi_area.size().width()
    available_height = viewport_size.height() or mdi_area.size().height()
    target_width = max(min_width, available_width - 24) if available_width > 0 else min_width
    target_height = max(min_height, available_height - 24) if available_height > 0 else min_height
    return target_width, target_height


class StationOutputHost(QWidget):
    """MDI host for read-only, station-level impedance output tools."""

    station_data_ready = Signal(int, str, object)

    def __init__(self, data_manager: DataManager | None = None, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.data_id = 0
        self.em_data = None
        self._package_cache: dict[int, dict] = {}
        self.mdi_area = QMdiArea(self)

        if self.data_manager is not None:
            self.data_manager.data_changed.connect(self._on_data_changed)
            self.data_manager.data_removed.connect(self._on_data_removed)
            self.data_manager.data_restored.connect(self._on_data_restored)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.mdi_area)

    def init_data(self, data, data_id: int = 0) -> None:
        self.data_id = data_id
        self.em_data = data
        self._package_cache.clear()
        if data is None:
            self.close_all_windows()

    def open_output_window(
        self,
        tool_id: str,
        data_id: int,
        position: QPoint | None = None,
    ) -> QMdiSubWindow | None:
        if self.data_manager is None:
            return None

        spec = tool_registry.get(tool_id)
        if spec is None or spec.requires_channel:
            return None
        data = self.data_manager.get(data_id)
        if data is None:
            return None

        existing = self._find_window(spec.tool_id, data_id)
        if existing is not None:
            self.present_window(existing)
            return existing

        widget = tool_registry.create_output_widget(spec.tool_id, data_id)
        if widget is None:
            return None

        window = QMdiSubWindow()
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        window.setWindowFlags(window.windowFlags() & ~Qt.WindowType.WindowSystemMenuHint)
        window.setProperty("tool_id", spec.tool_id)
        window.setProperty("data_id", data_id)
        window.setWindowTitle(f"{spec.display_name} - #{data_id} {data.name}")
        window.setWidget(widget)
        self.mdi_area.addSubWindow(window)
        self._apply_window_size(window, position)

        data_requested = getattr(widget, "data_requested", None)
        if data_requested is not None:
            data_requested.connect(self._provide_station_data)
        self.station_data_ready.connect(widget._on_data_ready)
        self._provide_station_data(data_id, "")
        return window

    def present_window(self, window: QMdiSubWindow | None) -> None:
        if window is None:
            return
        window.show()
        window.showMaximized()
        self.mdi_area.setActiveSubWindow(window)

    def close_all_windows(self) -> None:
        for window in list(self.mdi_area.subWindowList()):
            if window.close():
                self.mdi_area.removeSubWindow(window)

    def close_windows_for_data(self, data_id: int) -> None:
        for window in list(self.mdi_area.subWindowList()):
            if window.property("data_id") != data_id:
                continue
            if window.close():
                self.mdi_area.removeSubWindow(window)

    def invalidate_data(self, data_id: int, refresh: bool = True) -> None:
        """Discard a stale station package and refresh interested output tools."""
        self._package_cache.pop(data_id, None)
        if refresh and self.mdi_area.subWindowList():
            self._provide_station_data(data_id, "")

    def _on_data_changed(self, data_id: int, structural: bool) -> None:
        del structural
        self.invalidate_data(data_id)

    def _on_data_restored(self, data_id: int, channel: str | None) -> None:
        for window in self.mdi_area.subWindowList():
            widget = window.widget()
            invalidate = getattr(widget, "on_source_restored", None)
            if invalidate is not None:
                invalidate(data_id)

    def _on_data_removed(self, data_id: int) -> None:
        self._package_cache.pop(data_id, None)

    def _provide_station_data(self, data_id: int, channel: str = "") -> None:
        if self.data_manager is None:
            return
        package = self._package_cache.get(data_id)
        if package is None:
            package = station_data_package(self.data_manager, data_id)
            if package is not None:
                self._package_cache[data_id] = package
        if package is not None:
            self.station_data_ready.emit(data_id, channel, package)

    def _find_window(self, tool_id: str, data_id: int) -> QMdiSubWindow | None:
        for window in self.mdi_area.subWindowList():
            if window.property("tool_id") == tool_id and window.property("data_id") == data_id:
                return window
        return None

    def _apply_window_size(
        self,
        window: QMdiSubWindow,
        position: QPoint | None,
    ) -> None:
        width, height = _mdi_target_size(self.mdi_area)
        if position is None:
            window.resize(width, height)
        else:
            window.setGeometry(position.x(), position.y(), width, height)

    def closeEvent(self, event) -> None:
        self.close_all_windows()
        super().closeEvent(event)
