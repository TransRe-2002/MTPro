"""时间范围裁剪对话框（数据管理器就地预处理）。

用起止时间选择器裁剪数据集首尾；时间以台站时区显示，内部
换算为 tz-aware Timestamp 交给 ``EMData.trim_time_range``。
"""

from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDateTimeEdit,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from core.em_data import EMData
from ui.dialogs.decimate_dialog import format_duration
from ui.widgets.qt_time import pts_to_qdt, qdt_to_pts

_DISPLAY_FORMAT = "yyyy-MM-dd HH:mm:ss"


class TrimDialog(QDialog):
    """裁剪数据集时间范围。结构性变更。"""

    def __init__(self, em_data: EMData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("裁剪时间范围 — MTPro")
        self.setMinimumWidth(460)

        self._em_data = em_data
        self._tz = em_data.timezone or "UTC"

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        info_label = QLabel(
            "裁剪将就地删除指定范围之外的数据（不可撤销）。\n"
            f"时间按台站时区（{self._tz}）显示。",
            self,
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        duration = em_data.end_time - em_data.start_time
        range_label = QLabel(
            f"当前范围：{pts_to_qdt(em_data.start_time, self._tz).toString(_DISPLAY_FORMAT)}"
            f" ～ {pts_to_qdt(em_data.end_time, self._tz).toString(_DISPLAY_FORMAT)}"
            f"（{em_data.npts} 点，时长 {format_duration(duration)}）",
            self,
        )
        range_label.setStyleSheet("QLabel { color: #4070a0; font-weight: bold; }")
        range_label.setWordWrap(True)
        layout.addWidget(range_label)

        form = QFormLayout()
        form.setSpacing(6)

        min_qdt = pts_to_qdt(em_data.start_time, self._tz)
        max_qdt = pts_to_qdt(em_data.end_time, self._tz)

        self._start_edit = QDateTimeEdit(min_qdt, self)
        self._start_edit.setDisplayFormat(_DISPLAY_FORMAT)
        self._start_edit.setDateTimeRange(min_qdt, max_qdt)
        self._start_edit.setCalendarPopup(True)
        form.addRow("起始时间：", self._start_edit)

        self._end_edit = QDateTimeEdit(max_qdt, self)
        self._end_edit.setDisplayFormat(_DISPLAY_FORMAT)
        self._end_edit.setDateTimeRange(min_qdt, max_qdt)
        self._end_edit.setCalendarPopup(True)
        form.addRow("结束时间：", self._end_edit)

        layout.addLayout(form)

        self._preview_label = QLabel(self)
        self._preview_label.setWordWrap(True)
        layout.addWidget(self._preview_label)

        self._error_label = QLabel(self)
        self._error_label.setStyleSheet("QLabel { color: red; font-size: 11px; }")
        self._error_label.setWordWrap(True)
        self._error_label.hide()
        layout.addWidget(self._error_label)

        layout.addStretch(1)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self._start_edit.dateTimeChanged.connect(self._update_preview)
        self._end_edit.dateTimeChanged.connect(self._update_preview)
        self._update_preview()

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def time_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """返回 (起始, 结束) 的 tz-aware Timestamp（台站时区）。"""
        start = qdt_to_pts(self._start_edit.dateTime(), tz=self._tz)
        end = qdt_to_pts(self._end_edit.dateTime(), tz=self._tz)
        return start, end

    # ------------------------------------------------------------------
    # 私有槽
    # ------------------------------------------------------------------

    def _expected_npts(self) -> int:
        start, end = self.time_range()
        dt_seconds = self._em_data.dt.total_seconds()
        if dt_seconds <= 0 or end <= start:
            return 0
        return int((end - start).total_seconds() / dt_seconds) + 1

    def _update_preview(self) -> None:
        npts = self._expected_npts()
        if npts > 0:
            new_duration = self._em_data.dt * (npts - 1)
            self._preview_label.setText(
                f"裁剪后约 {npts} 点（原 {self._em_data.npts} 点），"
                f"时长 {format_duration(new_duration)}"
            )
        else:
            self._preview_label.setText("裁剪后无数据")
        self._error_label.hide()

    def _on_accept(self) -> None:
        start, end = self.time_range()
        if start >= end:
            self._show_error("起始时间必须早于结束时间。")
            return
        if self._expected_npts() < 2:
            self._show_error("裁剪范围内至少需要保留 2 个采样点。")
            return

        self._error_label.hide()
        self.accept()

    def _show_error(self, message: str) -> None:
        self._error_label.setText(message)
        self._error_label.show()
