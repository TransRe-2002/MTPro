"""Dedicated impedance-output workspace.

The two station-level output processors are intentionally kept outside the
transform plugin registry.  This page gives them a stable home while their
future multi-channel output API is designed.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from base.data_manager import DataManager
from ui.data_process import DataProcessLegacyWidget
from ui.impedance_preview import ImpedanceResultPool


class ImpedanceOutputWidget(QWidget):
    def __init__(self, data_manager: Optional[DataManager] = None, pool: ImpedanceResultPool | None = None, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self._data_id = 0
        self._data = None
        self.host = DataProcessLegacyWidget(data_manager, self)
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
        window = self.host.open_tool_window(tool_type, self._data_id, None)
        if window is not None:
            widget = window.widget()
            signal = getattr(widget, "result_ready", None)
            if signal is not None and not window.property("impedance_pool_connected"):
                signal.connect(lambda result, method=tool_type: self.pool.add(method, result))
                window.setProperty("impedance_pool_connected", True)
        self.host.present_subwindow(window)
