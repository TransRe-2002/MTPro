from __future__ import annotations

"""窗口均值去阶跃工具窗口。

该工具适合这类情况：
- 一阶差分中能看到明显峰值
- 台阶前后大体比较平稳
- 可以用局部窗口均值差近似台阶高度

本文件主要负责界面交互，真正的台阶估计算法在 step_algorithms 中。
"""

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pandas as pd
from PySide6.QtCore import QRegularExpression
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
)
import pyqtgraph as pg


from processor._base_processor import BaseProcessorWidget
from processor.step_algorithms import (
    DEFAULT_AVG_WINDOW,
    DEFAULT_MAX_STEPS,
    WindowedStepCandidate,
    windowed_mean_destep_with_diagnostics,
)
from processor.step_plot_widget import StepPlotWidget
from utils.validators import read_positive_int


class RemoveStepByWindow(BaseProcessorWidget):
    """基于"前后窗口均值差"的自动去阶跃窗口。"""
    DEFAULT_MIN_OFFSET = 100.0
    DEFAULT_AVG_WINDOW = DEFAULT_AVG_WINDOW
    DEFAULT_MAX_STEPS = DEFAULT_MAX_STEPS

    def __init__(self, data_id: int, channel_name: str, parent=None):
        super().__init__(data_id, channel_name, parent)

        self.diff_y_data: Optional[np.ndarray] = None

        self.plot_widget = StepPlotWidget()
        self.plot_item = self.plot_widget.getPlotItem()
        self.threshold_frame = None
        self.action_frame = None

        self.de_step_btn = QPushButton("窗口均值去阶跃")
        self.de_step_threshold = QLineEdit()
        self.avg_window_edit = QLineEdit()
        self.max_steps_edit = QLineEdit()
        float_validator = QRegularExpressionValidator(QRegularExpression(r"^\d*\.?\d+$"))
        int_validator = QRegularExpressionValidator(QRegularExpression(r"^\d+$"))
        self.de_step_threshold.setValidator(float_validator)
        self.avg_window_edit.setValidator(int_validator)
        self.max_steps_edit.setValidator(int_validator)
        self.undo_btn = QPushButton("撤销")
        self.revert_btn = QPushButton("还原原始数据")
        self.confirm_btn = QPushButton("确认")

        self.plain_text = QTextEdit()
        self.plain_text.setReadOnly(True)
        self.plain_text.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.plain_text.setMinimumWidth(260)
        self.plain_text.setMaximumWidth(320)

        self.avg_window_edit.setText(str(self.DEFAULT_AVG_WINDOW))
        self.max_steps_edit.setText(str(self.DEFAULT_MAX_STEPS))
        self.de_step_threshold.setText(f"{self.DEFAULT_MIN_OFFSET:.4f}")

        self._init_ui()
        self._connect_signal()


    def _on_data_loaded(self):
        self.diff_y_data = np.diff(self.y_data)
        self._init_plot()

    def _init_ui(self):
        """构建日志区、主图和参数按钮区。"""
        instruction_label = QLabel(
            "窗口均值去阶跃: 先用阈值定位一阶差分峰值，再用局部窗口均值估计台阶幅度。\n"
            "Shift+鼠标滚轮/中键拖动仅操作 X 轴，Alt+鼠标滚轮/中键拖动仅操作 Y 轴。"
        )

        self.plot_widget.getAxis("left").setStyle(tickLength=0)
        self.plot_widget.getAxis("bottom").setStyle(tickLength=0)

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
        threshold_layout.addWidget(QLabel("最多台阶："))
        threshold_layout.addWidget(self.max_steps_edit)
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
        """连接按钮行为。"""
        self.de_step_btn.clicked.connect(self._de_step_by_threshold)
        self.undo_btn.clicked.connect(self._undo)
        self.revert_btn.clicked.connect(self._revert)
        self.confirm_btn.clicked.connect(self._confirm_and_close)

    def _init_plot(self):
        """初始化主图中的原始序列和一阶差分曲线。"""
        y_centered = self.y_data - np.nanmean(self.y_data)
        self.y_line = pg.PlotCurveItem(
            self.x_data, y_centered,
            pen=pg.mkPen(255, 0, 0, width=1), name="序列数值",
            autoDownsample=True, downsampleMethod='peak',
            clipToView=True, useOpenGL=True, connect='finite',
        )
        self.dy_line = pg.PlotCurveItem(
            self.x_data[1:], self.diff_y_data,
            pen=pg.mkPen(0, 0, 255, width=1), name="序列一阶差分",
            autoDownsample=True, downsampleMethod='peak',
            clipToView=True, useOpenGL=True, connect='finite',
        )
        legend = self.plot_item.addLegend()
        legend.anchor((1, 0), (1, 0))
        legend.setBrush(pg.mkBrush(255, 255, 255, 255))
        legend.setPen(pg.mkPen("k", width=1))
        self.plot_item.showGrid(x=True, y=True, alpha=0.7)
        self.plot_item.addItem(self.y_line)
        self.plot_item.addItem(self.dy_line)
        self._show_top_diffs()

    # ------------------------------------------------------------------
    # BaseProcessorWidget 接口
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        self.plain_text.append(msg)

    def _update_derived_data(self):
        self.diff_y_data = np.diff(self.y_data)

    def _update_plot(self):
        """数据更新后，刷新主图与差分图。"""
        y_centered = self.y_data - np.nanmean(self.y_data)
        self.y_line.setData(self.x_data, y_centered)
        self.dy_line.setData(self.x_data[1:], self.diff_y_data)

    def _show_top_diffs(self):
        """把一阶差分中最大的若干个位置打印到日志，便于快速判断阈值。"""
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

    # ------------------------------------------------------------------
    # 去阶跃操作
    # ------------------------------------------------------------------

    def _log_rejected_step_candidates(
        self,
        threshold: float,
        avg_window: int,
        max_steps: int,
        candidates: list[WindowedStepCandidate],
    ):
        """当找到候选点但最终没有应用时，把拒绝原因写到日志里。"""
        if not candidates:
            self._log(f"阈值 {threshold:.4f} 未找到差分候选")
            return

        self._log(
            f"找到 {len(candidates)} 个差分候选，但窗口均值确认后未应用台阶"
            f" | 阈值={threshold:.4f} | 平均窗口={avg_window} | 最多台阶={max_steps}"
        )
        for candidate in candidates[:10]:
            if candidate.step_value is None:
                step_text = "窗口样本不足"
                reason = "拒绝: 台阶前/后窗口为空"
            else:
                step_text = f"{candidate.step_value:.4f}"
                reason = f"拒绝: |窗口均值台阶|={abs(candidate.step_value):.4f} < 阈值"
            self._log(
                f"  候选位置={candidate.index}, |原始差分|={candidate.raw_diff:.4f}, "
                f"窗口均值台阶={step_text}, 前窗点数={candidate.before_count}, "
                f"后窗点数={candidate.after_count}, {reason}"
            )
        if len(candidates) > 10:
            self._log(f"  ... 其余 {len(candidates) - 10} 个候选省略")
        self._log("")

    def _de_step_by_threshold(self):
        """按阈值执行一次窗口均值去阶跃。"""
        text = self.de_step_threshold.text().strip()
        if not text:
            self._log("请输入差分阈值")
            return

        threshold = float(text)
        if threshold <= 0:
            self._log("阈值必须大于 0")
            return

        avg_window = read_positive_int(self.avg_window_edit, "平均窗口", self._log)
        if avg_window is None:
            return
        max_steps = read_positive_int(self.max_steps_edit, "最多台阶", self._log)
        if max_steps is None:
            return

        self._save_history()
        corrected, applied_steps, candidates = windowed_mean_destep_with_diagnostics(
            self.y_data,
            threshold,
            avg_window=avg_window,
            max_steps=max_steps,
        )
        if not applied_steps:
            self.history.pop()
            self._log_rejected_step_candidates(threshold, avg_window, max_steps, candidates)
            return

        self.y_data = corrected
        self.diff_y_data = np.diff(self.y_data)
        self._update_plot()
        self._log(
            f"窗口均值去阶跃完成，去除了 {len(applied_steps)} 个台阶"
            f" | 阈值={threshold:.4f} | 平均窗口={avg_window} | 最多台阶={max_steps}"
        )
        for step_index, step_value in applied_steps:
            self._log(f"  阶跃位置={step_index}, 修正幅度={step_value:.4f}")
        self._log("")
        self._show_top_diffs()
