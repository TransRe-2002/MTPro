"""Parameter dialog shared by immediate and Pipeline Kp selection."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from PySide6.QtCore import QSize, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from core.algorithms.kp_selection import (
    DEFAULT_KP_SELECTION_PARAMS,
    KpSelectionResult,
    format_kp_condition,
    normalize_kp_selection_params,
    select_kp_samples,
)
from ui.dialogs.plugin_parameters import create_pipeline_link_all_checkbox


class KpSelectionTimeline(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._keep = np.empty(0, dtype=bool)
        self._known = np.empty(0, dtype=bool)
        self.setMinimumHeight(34)

    def sizeHint(self) -> QSize:
        return QSize(560, 38)

    def set_result(self, result: KpSelectionResult | None) -> None:
        if result is None:
            self._keep = np.empty(0, dtype=bool)
            self._known = np.empty(0, dtype=bool)
        else:
            self._keep = np.asarray(result.keep_mask, dtype=bool)
            self._known = np.asarray(result.known_mask, dtype=bool)
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        area = self.rect().adjusted(1, 5, -1, -5)
        painter.fillRect(area, QColor("#d2d5da"))
        if self._keep.size == 0 or area.width() <= 0:
            painter.setPen(QColor("#858b94"))
            painter.drawRect(area)
            return

        colors = {
            "keep": QColor("#4b9b69"),
            "reject": QColor("#c85c5c"),
            "missing": QColor("#9aa0a8"),
        }
        width = max(1, area.width())
        size = self._keep.size
        for offset in range(width):
            left = offset * size // width
            right = max(left + 1, (offset + 1) * size // width)
            known = self._known[left:right]
            keep = self._keep[left:right]
            if not np.any(known):
                color = colors["missing"]
            elif np.any(keep):
                color = colors["keep"]
            else:
                color = colors["reject"]
            painter.fillRect(area.left() + offset, area.top(), 1, area.height(), color)
        painter.setPen(QColor("#727780"))
        painter.drawRect(area)


class KpSelectionDialog(QDialog):
    def __init__(
        self,
        dataset_name: str,
        index: pd.DatetimeIndex,
        kp_data: pd.DataFrame | None,
        params: Mapping[str, object] | None = None,
        parent=None,
        *,
        mode: str = "pipeline",
        allow_link_all: bool = False,
        link_all: bool = False,
    ):
        super().__init__(parent)
        self._index = pd.DatetimeIndex(index)
        self._kp_data = kp_data
        self._mode = mode
        current = dict(DEFAULT_KP_SELECTION_PARAMS)
        current.update(dict(params or {}))
        self.params: dict[str, object] = current
        self.result: KpSelectionResult | None = None
        self.setWindowTitle("按 Kp 阈值选取时间段")
        self.setMinimumWidth(620)

        root = QVBoxLayout(self)
        title = QLabel(f"数据集：{dataset_name}", self)
        title.setStyleSheet("font-weight: 600;")
        root.addWidget(title)
        self.coverage_label = QLabel(self._coverage_text(), self)
        self.coverage_label.setWordWrap(True)
        root.addWidget(self.coverage_label)

        bounds_group = QGroupBox("选取范围", self)
        bounds = QGridLayout(bounds_group)
        self.lower_enabled = QCheckBox("下限", bounds_group)
        self.lower_operator = QComboBox(bounds_group)
        self.lower_operator.addItems(("≥", ">"))
        self.lower_value = self._kp_spinbox(bounds_group)
        self.upper_enabled = QCheckBox("上限", bounds_group)
        self.upper_operator = QComboBox(bounds_group)
        self.upper_operator.addItems(("<", "≤"))
        self.upper_value = self._kp_spinbox(bounds_group)
        bounds.addWidget(self.lower_enabled, 0, 0)
        bounds.addWidget(QLabel("Kp", bounds_group), 0, 1)
        bounds.addWidget(self.lower_operator, 0, 2)
        bounds.addWidget(self.lower_value, 0, 3)
        bounds.addWidget(self.upper_enabled, 1, 0)
        bounds.addWidget(QLabel("Kp", bounds_group), 1, 1)
        bounds.addWidget(self.upper_operator, 1, 2)
        bounds.addWidget(self.upper_value, 1, 3)
        bounds.setColumnStretch(4, 1)
        root.addWidget(bounds_group)

        interval_group = QGroupBox("连续时段", self)
        interval_form = QFormLayout(interval_group)
        self.minimum_duration = self._hours_spinbox(interval_group)
        self.trim_start = self._hours_spinbox(interval_group)
        self.trim_end = self._hours_spinbox(interval_group)
        self.missing_policy = QComboBox(interval_group)
        self.missing_policy.addItem("排除", "exclude")
        self.missing_policy.addItem("保留", "keep")
        interval_form.addRow("最短连续时长（小时）", self.minimum_duration)
        interval_form.addRow("区间起点向后裁去（小时）", self.trim_start)
        interval_form.addRow("区间终点向前裁去（小时）", self.trim_end)
        interval_form.addRow("Kp 缺失时段", self.missing_policy)
        root.addWidget(interval_group)

        self.condition_label = QLabel(self)
        self.condition_label.setStyleSheet("font-weight: 600;")
        root.addWidget(self.condition_label)
        self.timeline = KpSelectionTimeline(self)
        root.addWidget(self.timeline)
        legend = QLabel(
            '<span style="color:#4b9b69">■</span> 保留　'
            '<span style="color:#c85c5c">■</span> 排除　'
            '<span style="color:#9aa0a8">■</span> Kp 缺失',
            self,
        )
        root.addWidget(legend)
        self.summary_label = QLabel(self)
        self.summary_label.setWordWrap(True)
        root.addWidget(self.summary_label)

        if mode == "immediate":
            note = QLabel(
                "确认后将立即修改该数据集的全部时序通道；条件外样点设为 NaN。",
                self,
            )
            note.setWordWrap(True)
            note.setStyleSheet("color: #a45a27;")
            root.addWidget(note)

        self.link_all_checkbox: QCheckBox | None = None
        action_row = QHBoxLayout()
        action_row.addStretch(1)
        if allow_link_all:
            self.link_all_checkbox = create_pipeline_link_all_checkbox(
                self,
                checked=link_all,
            )
            action_row.addWidget(self.link_all_checkbox)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok_button.setText("确认并立即处理" if mode == "immediate" else "保存 Block 参数")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        action_row.addWidget(self.buttons)
        root.addLayout(action_row)

        self._load_values(current)
        self._connect_updates()
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(80)
        self._preview_timer.timeout.connect(self._update_preview)
        self._update_enabled_state()
        self._update_preview()

    @staticmethod
    def _kp_spinbox(parent) -> QDoubleSpinBox:
        widget = QDoubleSpinBox(parent)
        widget.setRange(0.0, 9.0)
        widget.setDecimals(2)
        widget.setSingleStep(0.1)
        return widget

    @staticmethod
    def _hours_spinbox(parent) -> QDoubleSpinBox:
        widget = QDoubleSpinBox(parent)
        widget.setRange(0.0, 100000.0)
        widget.setDecimals(2)
        widget.setSingleStep(1.0)
        return widget

    def _load_values(self, values: Mapping[str, object]) -> None:
        self.lower_enabled.setChecked(bool(values["lower_enabled"]))
        self.lower_value.setValue(float(values["lower_value"]))
        self.lower_operator.setCurrentText(
            "≥" if bool(values["lower_inclusive"]) else ">"
        )
        self.upper_enabled.setChecked(bool(values["upper_enabled"]))
        self.upper_value.setValue(float(values["upper_value"]))
        self.upper_operator.setCurrentText(
            "≤" if bool(values["upper_inclusive"]) else "<"
        )
        self.minimum_duration.setValue(float(values["minimum_duration_hours"]))
        self.trim_start.setValue(float(values["trim_start_hours"]))
        self.trim_end.setValue(float(values["trim_end_hours"]))
        policy_index = self.missing_policy.findData(str(values["missing_policy"]))
        self.missing_policy.setCurrentIndex(max(0, policy_index))

    def _connect_updates(self) -> None:
        for checkbox in (self.lower_enabled, self.upper_enabled):
            checkbox.toggled.connect(self._schedule_preview)
            checkbox.toggled.connect(self._update_enabled_state)
        for combo in (
            self.lower_operator,
            self.upper_operator,
            self.missing_policy,
        ):
            combo.currentIndexChanged.connect(self._schedule_preview)
        for spinbox in (
            self.lower_value,
            self.upper_value,
            self.minimum_duration,
            self.trim_start,
            self.trim_end,
        ):
            spinbox.valueChanged.connect(self._schedule_preview)

    def _schedule_preview(self, *_args) -> None:
        if hasattr(self, "_preview_timer"):
            self._preview_timer.start()

    def _update_enabled_state(self, *_args) -> None:
        self.lower_operator.setEnabled(self.lower_enabled.isChecked())
        self.lower_value.setEnabled(self.lower_enabled.isChecked())
        self.upper_operator.setEnabled(self.upper_enabled.isChecked())
        self.upper_value.setEnabled(self.upper_enabled.isChecked())

    def current_params(self) -> dict[str, object]:
        return {
            "lower_enabled": self.lower_enabled.isChecked(),
            "lower_value": self.lower_value.value(),
            "lower_inclusive": self.lower_operator.currentText() == "≥",
            "upper_enabled": self.upper_enabled.isChecked(),
            "upper_value": self.upper_value.value(),
            "upper_inclusive": self.upper_operator.currentText() == "≤",
            "minimum_duration_hours": self.minimum_duration.value(),
            "trim_start_hours": self.trim_start.value(),
            "trim_end_hours": self.trim_end.value(),
            "missing_policy": self.missing_policy.currentData(),
        }

    def _update_preview(self) -> None:
        try:
            values = normalize_kp_selection_params(self.current_params())
            result = select_kp_samples(self._index, self._kp_data, values)
        except (TypeError, ValueError) as exc:
            self.result = None
            self.condition_label.setText("参数无效")
            self.summary_label.setText(str(exc))
            self.timeline.set_result(None)
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
            return

        self.params = values
        self.result = result
        total = len(self._index)
        selected = result.selected_samples
        missing = total - result.known_samples
        selected_percent = 0.0 if total == 0 else selected * 100.0 / total
        missing_percent = 0.0 if total == 0 else missing * 100.0 / total
        self.condition_label.setText(f"实际条件：{format_kp_condition(values)}")
        self.summary_label.setText(
            f"符合条件：{len(result.raw_intervals)} 个连续区间，"
            f"{result.raw_duration_hours:g} 小时；"
            f"边界及短区间处理后：{len(result.selected_intervals)} 个区间，"
            f"{result.selected_duration_hours:g} 小时；"
            f"最终保留 {selected_percent:.1f}%，Kp 缺失 {missing_percent:.1f}%。"
        )
        self.timeline.set_result(result)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)

    def _coverage_text(self) -> str:
        if self._kp_data is None or self._kp_data.empty:
            return "Kp 覆盖：无可用数据"
        if "Kp_datetime" not in self._kp_data:
            return "Kp 覆盖：缺少 Kp_datetime 列"
        timestamps = pd.to_datetime(
            self._kp_data["Kp_datetime"], errors="coerce", utc=True,
        ).dropna()
        if timestamps.empty:
            return "Kp 覆盖：无有效时间"
        coverage_start = timestamps.min() - pd.Timedelta(hours=3)
        return (
            "Kp 覆盖："
            f"{timestamps.min().strftime('%Y-%m-%d %H:%M')} ～ "
            f"{timestamps.max().strftime('%Y-%m-%d %H:%M')} UTC；"
            "每条记录作用于其时间戳之前的 3 小时。"
        )

    def accept(self) -> None:
        self._preview_timer.stop()
        self._update_preview()
        if self.result is None:
            return
        if self.result.selected_samples == 0:
            answer = QMessageBox.question(
                self,
                "没有保留样点",
                "当前参数不会保留任何样点，继续后所有通道都将变为 NaN。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        super().accept()
