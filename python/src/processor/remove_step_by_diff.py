from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui
from PySide6.QtCore import QRegularExpression, Qt, Signal
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.em_data import Channel
from processor.remove_spike import RegionSelectionPlotWidget
from processor.step_algorithms import (
    DEFAULT_MAX_STEPS,
    remove_diff_steps_by_count,
    remove_diff_steps_by_threshold,
    zero_diff_indices,
)
from processor.step_plot_widget import StepPlotWidget
from utils.series import dti_to_numpy


class RemoveStepByDiff(QWidget):
    result_signal = Signal(str, np.ndarray)
    DEFAULT_MAX_STEPS = DEFAULT_MAX_STEPS

    def __init__(self, channel: Channel, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.ch = channel
        self.ch_name = channel.name
        self.x_data = dti_to_numpy(channel.datetime_index())
        y = channel.cts
        self.nan_index = y[y.isna()].index.tolist()
        interpolated = y.interpolate(method="linear")
        self.y_data = interpolated.to_numpy(dtype=np.float64)
        self.origin_y_data = self.y_data.copy()
        self.diff_y_data = np.diff(self.y_data)
        self.history: list[np.ndarray] = []
        self.selected_diff_indices: list[int] = []

        self.signal_plot_widget = StepPlotWidget()
        self.plot_widget = self.signal_plot_widget
        self.signal_plot_item = self.signal_plot_widget.getPlotItem()
        self.diff_plot_widget = RegionSelectionPlotWidget()
        self.diff_plot_widget.set_selection_mode("select")

        self.threshold_frame = None
        self.multi_step_frame = None
        self.manual_frame = None
        self.action_frame = None

        float_validator = QRegularExpressionValidator(QRegularExpression(r"^\d*\.?\d+$"))
        int_validator = QRegularExpressionValidator(QRegularExpression(r"^\d+$"))

        self.threshold_btn = QPushButton("按阈值去除")
        self.threshold_edit = QLineEdit()
        self.threshold_edit.setValidator(float_validator)
        self.one_step_btn = QPushButton("按次数去除")
        self.one_step_times = QLineEdit()
        self.one_step_times.setValidator(int_validator)
        self.add_region_btn = QPushButton("添加连续区域")
        self.remove_region_btn = QPushButton("移除最后区域")
        self.clear_region_btn = QPushButton("清空选择")
        self.zero_selected_btn = QPushButton("选中差分置零")
        self.selection_status_label = QLabel("当前选中: 0 个差分点")
        self.undo_btn = QPushButton("撤销")
        self.revert_btn = QPushButton("还原原始数据")
        self.confirm_btn = QPushButton("确认")

        max_abs_diff = np.max(np.abs(self.diff_y_data)) if self.diff_y_data.size else 0.0
        self.threshold_edit.setText(f"{max_abs_diff:.4f}")
        self.one_step_times.setText("5")

        self._init_ui()
        self._connect_signal()
        self._init_plot()

    def _init_ui(self):
        instruction_label = QLabel(
            "差分去阶跃: 可以按阈值/次数自动将一阶差分置零，也可以像 remove_spike 一样框选差分点后手动置零。\n"
            "下图支持双击添加连续区域，右键删除区域，Shift/Alt 可约束缩放轴。"
        )

        self.signal_plot_widget.getAxis("left").setStyle(tickLength=0)
        self.signal_plot_widget.getAxis("bottom").setStyle(tickLength=0)
        self.diff_plot_widget.getAxis("left").setStyle(tickLength=0)
        self.diff_plot_widget.getAxis("bottom").setStyle(tickLength=0)
        self.diff_plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.plain_text = QTextEdit()
        self.plain_text.setReadOnly(True)
        self.plain_text.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.plain_text.setMinimumWidth(260)
        self.plain_text.setMaximumWidth(320)

        self.threshold_frame = QFrame(self)
        self.threshold_frame.setFrameShape(QFrame.Shape.StyledPanel)
        threshold_layout = QHBoxLayout(self.threshold_frame)
        threshold_layout.setContentsMargins(8, 6, 8, 6)
        threshold_layout.addWidget(QLabel("阈值差分去阶跃"))
        threshold_layout.addSpacing(8)
        threshold_layout.addWidget(QLabel("差分阈值："))
        threshold_layout.addWidget(self.threshold_edit)
        threshold_layout.addWidget(self.threshold_btn)

        self.multi_step_frame = QFrame(self)
        self.multi_step_frame.setFrameShape(QFrame.Shape.StyledPanel)
        multi_layout = QHBoxLayout(self.multi_step_frame)
        multi_layout.setContentsMargins(8, 6, 8, 6)
        multi_layout.addWidget(QLabel("按次数去除"))
        multi_layout.addSpacing(8)
        multi_layout.addWidget(QLabel("差分次数："))
        multi_layout.addWidget(self.one_step_times)
        multi_layout.addWidget(self.one_step_btn)
        multi_layout.addStretch(1)

        self.manual_frame = QFrame(self)
        self.manual_frame.setFrameShape(QFrame.Shape.StyledPanel)
        manual_layout = QHBoxLayout(self.manual_frame)
        manual_layout.setContentsMargins(8, 6, 8, 6)
        manual_layout.addWidget(QLabel("手动差分框选"))
        manual_layout.addSpacing(8)
        manual_layout.addWidget(self.add_region_btn)
        manual_layout.addWidget(self.remove_region_btn)
        manual_layout.addWidget(self.clear_region_btn)
        manual_layout.addWidget(self.zero_selected_btn)
        manual_layout.addWidget(self.selection_status_label)
        manual_layout.addStretch(1)

        self.action_frame = QFrame(self)
        self.action_frame.setFrameShape(QFrame.Shape.StyledPanel)
        action_layout = QHBoxLayout(self.action_frame)
        action_layout.setContentsMargins(8, 6, 8, 6)
        action_layout.addWidget(QLabel("操作"))
        action_layout.addSpacing(8)
        action_layout.addWidget(self.undo_btn)
        action_layout.addWidget(self.revert_btn)
        action_layout.addWidget(self.confirm_btn)
        action_layout.addStretch(1)

        plot_column = QVBoxLayout()
        plot_column.addWidget(self.signal_plot_widget, 3)
        plot_column.addWidget(self.diff_plot_widget, 2)

        control_layout = QVBoxLayout()
        control_layout.addWidget(self.threshold_frame)
        control_layout.addWidget(self.multi_step_frame)
        control_layout.addWidget(self.manual_frame)
        control_layout.addWidget(self.action_frame)

        layout = QGridLayout(self)
        layout.addWidget(instruction_label, 0, 0, 1, 2)
        layout.addWidget(self.plain_text, 1, 0)
        layout.addLayout(plot_column, 1, 1)
        layout.addLayout(control_layout, 2, 0, 1, 2)

    def _connect_signal(self):
        self.threshold_btn.clicked.connect(self._remove_by_threshold)
        self.one_step_btn.clicked.connect(self._remove_by_count)
        self.add_region_btn.clicked.connect(self.diff_plot_widget.add_region)
        self.remove_region_btn.clicked.connect(self.diff_plot_widget.remove_selected_region)
        self.clear_region_btn.clicked.connect(self._clear_diff_selection)
        self.zero_selected_btn.clicked.connect(self._remove_selected_diff_indices)
        self.diff_plot_widget.selection_changed.connect(self._on_selection_changed)
        self.undo_btn.clicked.connect(self._undo)
        self.revert_btn.clicked.connect(self._revert)
        self.confirm_btn.clicked.connect(self._confirm_and_close)

    def _init_plot(self):
        y_centered = self.y_data - np.nanmean(self.y_data)
        origin_centered = self.origin_y_data - np.nanmean(self.origin_y_data)
        self.origin_line = pg.PlotCurveItem(
            self.x_data,
            origin_centered,
            pen=pg.mkPen(120, 120, 120, width=1, style=Qt.PenStyle.DashLine),
            name="原始序列",
        )
        self.y_line = pg.PlotCurveItem(
            self.x_data,
            y_centered,
            pen=pg.mkPen(255, 0, 0, width=1),
            name="当前序列",
        )
        legend = self.signal_plot_item.addLegend()
        legend.anchor((1, 0), (1, 0))
        legend.setBrush(pg.mkBrush(255, 255, 255, 255))
        legend.setPen(pg.mkPen("k", width=1))
        self.signal_plot_item.showGrid(x=True, y=True, alpha=0.7)
        self.signal_plot_item.addItem(self.origin_line)
        self.signal_plot_item.addItem(self.y_line)

        self._refresh_diff_plot()
        self._show_top_diffs()

    def showEvent(self, ev: QtGui.QShowEvent) -> None:
        super().showEvent(ev)
        self.activateWindow()
        self.signal_plot_widget.setFocus(QtCore.Qt.FocusReason.ActiveWindowFocusReason)

    def _log(self, msg: str):
        self.plain_text.append(msg)

    def _show_top_diffs(self):
        abs_diff = np.abs(self.diff_y_data)
        top_n = min(10, len(abs_diff))
        if top_n <= 0:
            self._log("===== 一阶差分为空 =====")
            self._log("")
            return
        top_indices = np.argpartition(abs_diff, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(abs_diff[top_indices])[::-1]]
        self._log(f"===== 一阶差分(绝对值) Top {top_n} =====")
        for rank, idx in enumerate(top_indices, 1):
            ts = pd.Timestamp(self.x_data[idx], unit="s")
            self._log(f"  {rank}. 位置={idx}, 时间={ts}, |差分|={abs_diff[idx]:.4f}")
        self._log("")

    def _save_history(self):
        self.history.append(self.y_data.copy())

    def _refresh_diff_plot(self):
        if self.x_data.size <= 1:
            return
        self.diff_plot_widget.set_data(self.x_data[1:], self.diff_y_data)
        self.diff_plot_widget.set_selection_mode("select")
        self.selected_diff_indices = []
        self.selection_status_label.setText("当前选中: 0 个差分点")

    def _update_plot(self):
        y_centered = self.y_data - np.nanmean(self.y_data)
        origin_centered = self.origin_y_data - np.nanmean(self.origin_y_data)
        self.origin_line.setData(self.x_data, origin_centered)
        self.y_line.setData(self.x_data, y_centered)
        self._refresh_diff_plot()

    def _read_positive_int(self, edit: QLineEdit, label: str) -> Optional[int]:
        text = edit.text().strip()
        if not text:
            self._log(f"{label} 不能为空")
            return None
        value = int(text)
        if value <= 0:
            self._log(f"{label} 必须大于 0")
            return None
        return value

    def _on_selection_changed(self, selected_indices: list[int]):
        self.selected_diff_indices = list(selected_indices)
        self.selection_status_label.setText(f"当前选中: {len(selected_indices)} 个差分点")

    def _clear_diff_selection(self):
        self.diff_plot_widget.clear_all_selections()
        self.selected_diff_indices = []
        self.selection_status_label.setText("当前选中: 0 个差分点")

    def _matlab_style_one_step(
        self,
        samples: np.ndarray,
        n_steps: int,
    ) -> tuple[np.ndarray, list[tuple[int, float]]]:
        return remove_diff_steps_by_count(samples, n_steps)

    def _diff_threshold_destep(
        self,
        samples: np.ndarray,
        min_offset: float,
        max_steps: int | None = None,
    ) -> tuple[np.ndarray, list[tuple[int, float]]]:
        limit = self.DEFAULT_MAX_STEPS if max_steps is None else int(max_steps)
        return remove_diff_steps_by_threshold(samples, min_offset, limit)

    def _remove_by_threshold(self):
        text = self.threshold_edit.text().strip()
        if not text:
            self._log("请输入差分阈值")
            return

        threshold = float(text)
        if threshold <= 0:
            self._log("阈值必须大于 0")
            return

        self._save_history()
        corrected, removed_steps = self._diff_threshold_destep(
            self.y_data,
            threshold,
            max_steps=self.DEFAULT_MAX_STEPS,
        )
        if not removed_steps:
            self.history.pop()
            self._log(f"阈值 {threshold:.4f} 未找到可去除的台阶")
            return

        self.y_data = corrected
        self.diff_y_data = np.diff(self.y_data)
        self._update_plot()
        self._log(
            f"阈值差分去阶跃完成，共去除 {len(removed_steps)} 个台阶"
            f" | 阈值={threshold:.4f} | 最多台阶={self.DEFAULT_MAX_STEPS}"
        )
        for step_index, step_value in removed_steps:
            self._log(f"  阶跃位置={step_index}, 幅度={step_value:.4f}")
        self._log("")
        self._show_top_diffs()

    def _remove_by_count(self):
        times = self._read_positive_int(self.one_step_times, "次数")
        if times is None:
            return

        self._save_history()
        corrected, removed_steps = self._matlab_style_one_step(self.y_data, times)
        if not removed_steps:
            self.history.pop()
            self._log("未找到可去除的台阶")
            return

        self.y_data = corrected
        self.diff_y_data = np.diff(self.y_data)
        self._update_plot()
        self._log(f"按次数去除完成，共去除 {len(removed_steps)} 个台阶")
        for step_number, (index, step_value) in enumerate(removed_steps, start=1):
            self._log(f"  去除第 {step_number} 个台阶: 位置={index}, 幅度={step_value:.4f}")
        self._log("")
        self._show_top_diffs()

    def _remove_selected_diff_indices(self):
        if not self.selected_diff_indices:
            self._log("请先在下方一阶差分图中框选要置零的差分点")
            return

        self._save_history()
        corrected, removed_steps = zero_diff_indices(self.y_data, self.selected_diff_indices)
        if not removed_steps:
            self.history.pop()
            self._log("选中的差分点没有可置零的台阶")
            return

        self.y_data = corrected
        self.diff_y_data = np.diff(self.y_data)
        self._update_plot()
        self._log(f"手动差分置零完成，共去除 {len(removed_steps)} 个台阶")
        for step_index, step_value in removed_steps:
            self._log(f"  手动去除: 位置={step_index}, 幅度={step_value:.4f}")
        self._log("")
        self._show_top_diffs()

    def _undo(self):
        if not self.history:
            self._log("没有可撤销的操作")
            return
        self.y_data = self.history.pop()
        self.diff_y_data = np.diff(self.y_data)
        self._update_plot()
        self._log("已撤销上一步操作")

    def _revert(self):
        self._save_history()
        self.y_data = self.origin_y_data.copy()
        self.diff_y_data = np.diff(self.y_data)
        self._update_plot()
        self._log("已还原到原始数据")

    def _confirm_and_close(self):
        result = self.y_data.copy()
        for idx in self.nan_index:
            if 0 <= idx < len(result):
                result[idx] = np.nan
        self._log(f"确认修改，返回 {len(result)} 个数据点")
        self.result_signal.emit(self.ch_name, result)
        self.hide()
        self.close()
        if self.parent() is not None:
            self.parent().close()
