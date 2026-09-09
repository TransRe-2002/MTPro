"""Dedicated impedance-output workspace.

The two station-level output processors are intentionally kept outside the
transform plugin registry.  This page gives them a stable home while their
future multi-channel output API is designed.
"""
from __future__ import annotations


from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from base.data_manager import DataManager
from ui.impedance_preview import ImpedanceResultPool
from ui.station_output_host import StationOutputHost


class ImpedanceOutputWidget(QWidget):
    preview_requested = Signal(str)

    def __init__(self, data_manager: DataManager | None = None, pool: ImpedanceResultPool | None = None, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self._data_id = 0
        self._data = None
        self.host = StationOutputHost(data_manager, self)
        self.pool = pool or ImpedanceResultPool(self)

        title = QLabel("阻抗输出", self)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        hint = QLabel("选择当前数据集后运行阻抗估计或远参考法。", self)
        hint.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.robust_button = QToolButton(self)
        self.robust_button.setText("Robust 估计")
        self.remote_button = QToolButton(self)
        self.remote_button.setText("远参考法")
        self.robust_button.clicked.connect(lambda: self._open("robust estimate"))
        self.remote_button.clicked.connect(lambda: self._open("remote reference"))

        toolbar = QHBoxLayout()
        toolbar.addWidget(title)
        toolbar.addWidget(hint)
        toolbar.addStretch(1)
        toolbar.addWidget(self.robust_button)
        toolbar.addWidget(self.remote_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(toolbar)
        layout.addWidget(self.host, 1)

    def init_data(self, data, data_id: int = 0) -> None:
        self._data = data
        self._data_id = data_id
        self.host.init_data(data, data_id)
        enabled = data is not None and data_id != 0
        self.robust_button.setEnabled(enabled)
        self.remote_button.setEnabled(enabled)

    def _open(self, tool_type: str) -> None:
        if self._data is None or self._data_id == 0:
            return
        window = self.host.open_output_window(tool_type, self._data_id)
        if window is not None:
            widget = window.widget()
            save_signal = getattr(widget, "snapshot_save_requested", None)
            preview_signal = getattr(widget, "preview_requested", None)
            if save_signal is not None and not window.property("impedance_pool_connected"):
                save_signal.connect(lambda payload, owner=widget: self._save_snapshot(payload, owner))
                if preview_signal is not None:
                    preview_signal.connect(lambda owner=widget: self._request_preview(owner))
                window.setProperty("impedance_pool_connected", True)
        self.host.present_window(window)

    def _save_snapshot(self, payload: dict, owner: QWidget) -> None:
        snapshot = self.pool.add_payload(payload)
        owner.setProperty("saved_impedance_snapshot_id", snapshot.snapshot_id)

    def _request_preview(self, owner: QWidget) -> None:
        snapshot_id = str(owner.property("saved_impedance_snapshot_id") or "")
        self.preview_requested.emit(snapshot_id)

    def close_windows_for_data(self, data_id: int) -> None:
        self.host.close_windows_for_data(data_id)

    def pending_calculation_count(self) -> int:
        count = 0
        for window in self.host.mdi_area.subWindowList():
            widget = window.widget()
            if getattr(widget, "_bands", None) is not None and not getattr(widget, "_saved_current", True):
                count += 1
        return count

    def running_calculation_count(self) -> int:
        return sum(
            getattr(window.widget(), "_estimation_future", None) is not None
            for window in self.host.mdi_area.subWindowList()
        )

    def discard_and_close_all(self) -> None:
        for window in self.host.mdi_area.subWindowList():
            widget = window.widget()
            cancel = getattr(widget, "_cancel_estimation", None)
            if getattr(widget, "_estimation_future", None) is not None and cancel is not None:
                cancel()
            if hasattr(widget, "_saved_current"):
                widget._saved_current = True
        self.host.close_all_windows()
