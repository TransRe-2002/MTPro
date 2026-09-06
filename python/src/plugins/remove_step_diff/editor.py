from __future__ import annotations

"""差分去阶跃工具窗口。

这个工具直接在一阶差分层面工作：
- 自动模式可以按阈值或按次数把差分尖峰置零
- 手动模式可以在差分图上框选后置零

随后通过重建累积和得到去阶跃后的原序列。
"""

import os
from typing import Optional

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pandas as pd
from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtGui import QKeySequence, QRegularExpressionValidator, QShortcut
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
)
import pyqtgraph as pg


from plugins.common.base_processor import BaseProcessorWidget
from plugins.remove_region.editor import RegionPlotWidget as RegionSelectionPlotWidget
from plugins.common.step_algorithms import (
    DEFAULT_MAX_STEPS,
    remove_diff_steps_by_count,
    remove_diff_steps_by_threshold,
    zero_diff_indices,
)
from plugins.common.plotting import StepPlotWidget
from utils.theme import subtle_pen, transparent_legend_brush
from utils.validators import read_positive_int


class RemoveStepByDiff(BaseProcessorWidget):
    """基于一阶差分置零的去阶跃窗口。"""
    DEFAULT_MAX_STEPS = DEFAULT_MAX_STEPS

    def __init__(self, source, channel_name: str, parent=None):
        super().__init__(source, channel_name, parent)

        self.diff_y_data: Optional[np.ndarray] = None
        self.selected_diff_indices: list[int] = []
        self._syncing_x_range = False

        self.signal_plot_widget = StepPlotWidget()
        self.plot_widget = self.signal_plot_widget
        self.signal_plot_item = self.signal_plot_widget.getPlotItem()
        self.diff_plot_widget = RegionSelectionPlotWidget()
        self.diff_plot_widget.set_selection_mode("select")
        self.plot_splitter = None

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
        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.revert_btn = QPushButton("还原原始数据")
        self.confirm_btn = QPushButton("确认")

        self.plain_text = QTextEdit()
        self.plain_text.setReadOnly(True)
        self.plain_text.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.plain_text.setMinimumWidth(260)
        self.plain_text.setMaximumWidth(320)

        self.one_step_times.setText("5")

        self._init_ui()
        self._connect_signal()
        self._load_input_data()


    def _on_data_loaded(self):
        self.diff_y_data = np.diff(self.y_data)
        max_abs_diff = np.max(np.abs(self.diff_y_data)) if self.diff_y_data.size else 0.0
        self.threshold_edit.setText(f"{max_abs_diff:.4f}")
        self._init_plot()

    def _init_ui(self):
        """搭建主序列图、差分图和控制区。"""
        instruction_label = QLabel(
            "差分去阶跃: 可以按阈值/次数自动将一阶差分置零，也可以像 remove_spike 一样框选差分点后手动置零。\n"
            "下图支持双击添加连续区域，右键删除区域，Shift/Alt 可约束缩放轴。"
        )

        self.signal_plot_widget.getAxis("left").setStyle(tickLength=0)
        self.signal_plot_widget.getAxis("bottom").setStyle(tickLength=0)
        self.diff_plot_widget.getAxis("left").setStyle(tickLength=0)
        self.diff_plot_widget.getAxis("bottom").setStyle(tickLength=0)
        self.diff_plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._apply_diff_plot_grid()

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

        self.plot_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.plot_splitter.setChildrenCollapsible(False)
        self.plot_splitter.setOpaqueResize(False)
        self.plot_splitter.addWidget(self.signal_plot_widget)
        self.plot_splitter.addWidget(self.diff_plot_widget)
        self.plot_splitter.setStretchFactor(0, 3)
        self.plot_splitter.setStretchFactor(1, 2)
        self.plot_splitter.setSizes([600, 400])

        processing_frame_layout = QHBoxLayout()
        processing_frame_layout.addWidget(self.threshold_frame)
        processing_frame_layout.addWidget(self.multi_step_frame)
        processing_frame_layout.addWidget(self.manual_frame, 1)

        control_layout = QVBoxLayout()
        control_layout.addLayout(processing_frame_layout)
        control_layout.addWidget(self.action_frame)

        layout = QGridLayout(self)
        layout.addWidget(instruction_label, 0, 0, 1, 2)
        layout.addWidget(self.plain_text, 1, 0)
        layout.addWidget(self.plot_splitter, 1, 1)
        layout.addLayout(control_layout, 2, 0, 1, 2)

    def _connect_signal(self):
        """连接自动去除、手动框选和撤销等操作。"""
        self.threshold_btn.clicked.connect(self._remove_by_threshold)
        self.one_step_btn.clicked.connect(self._remove_by_count)
        self.add_region_btn.clicked.connect(self.diff_plot_widget.add_region)
        self.remove_region_btn.clicked.connect(self.diff_plot_widget.remove_selected_region)
        self.clear_region_btn.clicked.connect(self._clear_diff_selection)
        self.zero_selected_btn.clicked.connect(self._remove_selected_diff_indices)
        self.diff_plot_widget.selection_changed.connect(self._on_selection_changed)
        self.undo_btn.clicked.connect(self._undo)
        self.undo_shortcut.activated.connect(self._undo)
        self.revert_btn.clicked.connect(self._revert)
        self.confirm_btn.clicked.connect(self._confirm_and_close)
        self.signal_plot_item.vb.sigXRangeChanged.connect(self._sync_diff_plot_x_range)
        self.diff_plot_widget.getViewBox().sigXRangeChanged.connect(self._sync_signal_plot_x_range)

    def _init_plot(self):
        """初始化上方序列图和下方差分图。"""
        y_centered = self.y_data - np.nanmean(self.y_data)
        origin_centered = self.origin_y_data - np.nanmean(self.origin_y_data)
        curve_opts = dict(
            autoDownsample=True, downsampleMethod='peak',
            clipToView=True, useOpenGL=True, connect='finite',
        )
        self.origin_line = StepPlotWidget.make_curve(
            self.x_data, origin_centered,
            pen=pg.mkPen(120, 120, 120, width=1, style=Qt.PenStyle.DashLine),
            name="原始序列", **curve_opts,
        )
        self.y_line = StepPlotWidget.make_curve(
            self.x_data, y_centered,
            pen=pg.mkPen(255, 0, 0, width=1),
            name="当前序列", **curve_opts,
        )
        legend = self.signal_plot_item.addLegend()
        legend.anchor((1, 0), (1, 0))
        legend.setBrush(transparent_legend_brush(self))
        legend.setPen(subtle_pen(self))
        self.signal_plot_item.showGrid(x=True, y=True, alpha=0.7)
        self.signal_plot_item.addItem(self.origin_line)
        self.signal_plot_item.addItem(self.y_line)

        self._refresh_diff_plot()
        self._apply_shared_x_range(self.signal_plot_widget, self.diff_plot_widget)
        self._show_top_diffs()

    def _apply_diff_plot_grid(self):
        grid_pen = pg.mkPen(color=(100, 100, 100, 180), width=1, style=Qt.PenStyle.DashLine)
        plot_item = self.diff_plot_widget.getPlotItem()
        plot_item.showGrid(x=True, y=True, alpha=0.7)
        plot_item.getAxis("bottom").setPen(grid_pen)
        plot_item.getAxis("left").setPen(grid_pen)

    # ------------------------------------------------------------------
    # BaseProcessorWidget 接口
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        self.plain_text.append(msg)

    def _update_derived_data(self):
        self.diff_y_data = np.diff(self.y_data)

    def _update_plot(self):
        """当 y_data 更新后，同时更新主图和差分图。"""
        y_centered = self.y_data - np.nanmean(self.y_data)
        origin_centered = self.origin_y_data - np.nanmean(self.origin_y_data)
        self.origin_line.setData(self.x_data, origin_centered)
        self.y_line.setData(self.x_data, y_centered)
        self._refresh_diff_plot()

    # ------------------------------------------------------------------
    # X 轴同步（双图）
    # ------------------------------------------------------------------

    def _apply_shared_x_range(self, source_plot: pg.PlotWidget, target_plot: pg.PlotWidget) -> None:
        """同步上下两张图的 X 轴范围。"""
        if self._syncing_x_range:
            return
        source_view = source_plot.getViewBox()
        target_view = target_plot.getViewBox()
        x_min, x_max = source_view.viewRange()[0]
        self._syncing_x_range = True
        try:
            target_view.setXRange(float(x_min), float(x_max), padding=0.0)
        finally:
            self._syncing_x_range = False

    def _sync_diff_plot_x_range(self, _view_box, _x_range) -> None:
        self._apply_shared_x_range(self.signal_plot_widget, self.diff_plot_widget)

    def _sync_signal_plot_x_range(self, _view_box, _x_range) -> None:
        self._apply_shared_x_range(self.diff_plot_widget, self.signal_plot_widget)

    # ------------------------------------------------------------------
    # 差分图和数据刷新
    # ------------------------------------------------------------------

    def _refresh_diff_plot(self):
        """重绘下方差分图，并清空旧的选择状态。"""
        if self.x_data.size <= 1:
            return
        self.diff_plot_widget.set_data(self.x_data[1:], self.diff_y_data)
        self._apply_diff_plot_grid()
        self.diff_plot_widget.set_selection_mode("select")
        self._apply_shared_x_range(self.signal_plot_widget, self.diff_plot_widget)
        self.selected_diff_indices = []
        self.selection_status_label.setText("当前选中: 0 个差分点")

    def _show_top_diffs(self):
        """把最大的若干个差分值打印到日志，帮助设阈值。"""
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

    def _on_selection_changed(self, selected_indices: list[int]):
        """接收差分图返回的选中差分点索引。"""
        self.selected_diff_indices = list(selected_indices)
        self.selection_status_label.setText(f"当前选中: {len(selected_indices)} 个差分点")

    def _clear_diff_selection(self):
        self.diff_plot_widget.clear_all_selections()
        self.selected_diff_indices = []
        self.selection_status_label.setText("当前选中: 0 个差分点")

    # ------------------------------------------------------------------
    # 去阶跃操作
    # ------------------------------------------------------------------

    def _remove_by_threshold(self):
        """按阈值去除若干个最显著的差分台阶。"""
        text = self.threshold_edit.text().strip()
        if not text:
            self._log("请输入差分阈值")
            return

        threshold = float(text)
        if threshold <= 0:
            self._log("阈值必须大于 0")
            return

        self._save_history()
        corrected, removed_steps = remove_diff_steps_by_threshold(
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
        """按指定次数去除最强的若干个差分台阶。"""
        times = read_positive_int(self.one_step_times, "次数", self._log)
        if times is None:
            return

        self._save_history()
        corrected, removed_steps = remove_diff_steps_by_count(self.y_data, times)
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
        """把用户手工框选的差分点置零。"""
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
