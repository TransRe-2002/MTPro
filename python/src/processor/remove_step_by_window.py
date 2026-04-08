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
from processor.step_algorithms import DEFAULT_AVG_WINDOW, DEFAULT_MAX_STEPS, windowed_mean_destep
from processor.step_plot_widget import StepPlotWidget
from utils.series import dti_to_numpy


class RemoveStepByWindow(QWidget):
    result_signal = Signal(str, np.ndarray)
    DEFAULT_AVG_WINDOW = DEFAULT_AVG_WINDOW
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

        self.plot_widget = StepPlotWidget()
        self.plot_item = self.plot_widget.getPlotItem()
        self.threshold_frame = None
        self.action_frame = None

        self.de_step_btn = QPushButton("窗口均值去阶跃")
        self.de_step_threshold = QLineEdit()
        self.avg_window_edit = QLineEdit()
        float_validator = QRegularExpressionValidator(QRegularExpression(r"^\d*\.?\d+$"))
        int_validator = QRegularExpressionValidator(QRegularExpression(r"^\d+$"))
        self.de_step_threshold.setValidator(float_validator)
        self.avg_window_edit.setValidator(int_validator)
        self.undo_btn = QPushButton("撤销")
        self.revert_btn = QPushButton("还原原始数据")
        self.confirm_btn = QPushButton("确认")

        self.avg_window_edit.setText(str(self.DEFAULT_AVG_WINDOW))
        max_abs_diff = np.max(np.abs(self.diff_y_data)) if self.diff_y_data.size else 0.0
        self.de_step_threshold.setText(f"{max_abs_diff:.4f}")

        self._init_ui()
        self._connect_signal()
        self._init_plot()

    def _init_ui(self):
        instruction_label = QLabel(
            "窗口均值去阶跃: 先用阈值定位一阶差分峰值，再用局部窗口均值估计台阶幅度。\n"
            "Shift+鼠标滚轮/中键拖动仅操作 X 轴，Alt+鼠标滚轮/中键拖动仅操作 Y 轴。"
        )

        self.plot_widget.getAxis("left").setStyle(tickLength=0)
        self.plot_widget.getAxis("bottom").setStyle(tickLength=0)

        self.plain_text = QTextEdit()
        self.plain_text.setReadOnly(True)
        self.plain_text.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.plain_text.setMinimumWidth(260)
        self.plain_text.setMaximumWidth(320)

        self.threshold_frame = QFrame(self)
        self.threshold_frame.setFrameShape(QFrame.Shape.StyledPanel)
        threshold_layout = QHBoxLayout(self.threshold_frame)
        threshold_layout.setContentsMargins(8, 6, 8, 6)
        threshold_layout.addWidget(QLabel("窗口均值去阶跃"))
        threshold_layout.addSpacing(8)
        threshold_layout.addWidget(QLabel("差分阈值："))
        threshold_layout.addWidget(self.de_step_threshold)
        threshold_layout.addWidget(QLabel("平均窗口："))
        threshold_layout.addWidget(self.avg_window_edit)
        threshold_layout.addWidget(self.de_step_btn)

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

        control_layout = QVBoxLayout()
        control_layout.addWidget(self.threshold_frame)
        control_layout.addWidget(self.action_frame)

        layout = QGridLayout(self)
        layout.addWidget(instruction_label, 0, 0, 1, 2)
        layout.addWidget(self.plain_text, 1, 0)
        layout.addWidget(self.plot_widget, 1, 1)
        layout.addLayout(control_layout, 2, 0, 1, 2)

    def _connect_signal(self):
        self.de_step_btn.clicked.connect(self._de_step_by_threshold)
        self.undo_btn.clicked.connect(self._undo)
        self.revert_btn.clicked.connect(self._revert)
        self.confirm_btn.clicked.connect(self._confirm_and_close)

    def _init_plot(self):
        y_centered = self.y_data - np.nanmean(self.y_data)
        self.y_line = pg.PlotCurveItem(self.x_data, y_centered, pen=pg.mkPen(255, 0, 0, width=1), name="序列数值")
        self.dy_line = pg.PlotCurveItem(self.x_data[1:], self.diff_y_data, pen=pg.mkPen(0, 0, 255, width=1), name="序列一阶差分")
        legend = self.plot_item.addLegend()
        legend.anchor((1, 0), (1, 0))
        legend.setBrush(pg.mkBrush(255, 255, 255, 255))
        legend.setPen(pg.mkPen("k", width=1))
        self.plot_item.showGrid(x=True, y=True, alpha=0.7)
        self.plot_item.addItem(self.y_line)
        self.plot_item.addItem(self.dy_line)
        self._show_top_diffs()

    def showEvent(self, ev: QtGui.QShowEvent) -> None:
        super().showEvent(ev)
        self.activateWindow()
        self.plot_widget.setFocus(QtCore.Qt.FocusReason.ActiveWindowFocusReason)

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

    def _update_plot(self):
        y_centered = self.y_data - np.nanmean(self.y_data)
        self.y_line.setData(self.x_data, y_centered)
        self.dy_line.setData(self.x_data[1:], self.diff_y_data)

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

    def _windowed_mean_destep(
        self,
        samples: np.ndarray,
        min_offset: float,
        avg_window: int | None = None,
        max_steps: int | None = None,
    ) -> tuple[np.ndarray, list[tuple[int, float]]]:
        return windowed_mean_destep(samples, min_offset, avg_window, max_steps)

    def _de_step_by_threshold(self):
        text = self.de_step_threshold.text().strip()
        if not text:
            self._log("请输入差分阈值")
            return

        threshold = float(text)
        if threshold <= 0:
            self._log("阈值必须大于 0")
            return

        avg_window = self._read_positive_int(self.avg_window_edit, "平均窗口")
        if avg_window is None:
            return

        self._save_history()
        corrected, applied_steps = self._windowed_mean_destep(
            self.y_data,
            threshold,
            avg_window=avg_window,
            max_steps=self.DEFAULT_MAX_STEPS,
        )
        if not applied_steps:
            self.history.pop()
            self._log(f"阈值 {threshold:.4f} 未找到台阶")
            return

        self.y_data = corrected
        self.diff_y_data = np.diff(self.y_data)
        self._update_plot()
        self._log(
            f"窗口均值去阶跃完成，去除了 {len(applied_steps)} 个台阶"
            f" | 阈值={threshold:.4f} | 平均窗口={avg_window} | 最多台阶={self.DEFAULT_MAX_STEPS}"
        )
        for step_index, step_value in applied_steps:
            self._log(f"  阶跃位置={step_index}, 修正幅度={step_value:.4f}")
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
