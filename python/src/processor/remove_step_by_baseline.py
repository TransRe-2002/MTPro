from __future__ import annotations

"""鲁棒基线去阶跃工具窗口。

这个工具适合"台阶不是瞬时跳变，而是跨多个采样点平滑过渡"的情况。
处理思路是：
1. 用户先框选一个阶跃过渡区域。
2. 在该区域左右各取一定上下文。
3. 用"低阶基线 + 饱和过渡项"做鲁棒拟合。
4. 只减去拟合得到的阶跃校正项，保留整体趋势背景。

真正的 robust 拟合算法在 step_algorithms.robust_baseline_destep 中。
"""

import argparse
import os
from typing import Optional

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
)
import pyqtgraph as pg

from core.em_data import EMData
from processor._base_processor import BaseProcessorWidget
from processor._demo_utils import DemoEMData, run_standalone_demo
from processor._region_mixin import RegionManagementMixin
from processor.step_algorithms import RobustBaselineStepResult, robust_baseline_destep
from processor.step_plot_widget import StepPlotWidget
from utils.theme import subtle_pen, transparent_legend_brush


class RemoveStepByBaseline(BaseProcessorWidget, RegionManagementMixin):
    """基于鲁棒基线拟合的去阶跃窗口。"""

    def __init__(self, data_id: int, channel_name: str, parent=None):
        super().__init__(data_id, channel_name, parent)

        self.last_result: Optional[RobustBaselineStepResult] = None
        self.selected_indices: list[int] = []
        self.log_messages: list[str] = []

        self.signal_plot_widget = StepPlotWidget()
        self.plot_widget = self.signal_plot_widget
        self.signal_plot_item = self.signal_plot_widget.getPlotItem()
        self.correction_plot_widget = StepPlotWidget()
        self.correction_plot_item = self.correction_plot_widget.getPlotItem()
        self.region_item: Optional[pg.LinearRegionItem] = None
        self.plot_splitter = None

        self.param_frame = None
        self.region_frame = None
        self.action_frame = None

        self.base_degree_spin = QSpinBox()
        self.transition_degree_spin = QSpinBox()
        self.context_spin = QSpinBox()
        self.loss_combo = QComboBox()
        self.preview_btn = QPushButton("预览拟合")
        self.apply_btn = QPushButton("应用校正")
        self.add_region_btn = QPushButton("添加连续区域")
        self.remove_region_btn = QPushButton("移除最后区域")
        self.clear_region_btn = QPushButton("清空选择")
        self.selection_status_label = QLabel("当前选中: 0 个采样点")
        self.undo_btn = QPushButton("撤销")
        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.revert_btn = QPushButton("还原原始数据")
        self.confirm_btn = QPushButton("确认")

        self._init_ui()
        self._connect_signal()
        self._load_input_data()


    def _on_data_loaded(self):
        self._init_plot()

    def _init_ui(self):
        """构建说明文字、参数区、区域选择区和双图布局。"""
        instruction_label = QLabel(
            "鲁棒基线去阶跃: 先框选连续阶跃过渡区，再用基线多项式 + 饱和过渡项做 robust 拟合，"
            "最后只减去阶跃校正项。\n"
            "适合经过多个采样点连续上升/下降、但整体趋势仍然单调的阶跃。"
            "基线阶数建议只用 0 或 1；过渡阶数可适当提高以拟合弯曲过渡，但会自动加平滑约束抑制细碎震荡。"
        )

        self.signal_plot_widget.getAxis("left").setStyle(tickLength=0)
        self.signal_plot_widget.getAxis("bottom").setStyle(tickLength=0)
        self.correction_plot_widget.getAxis("left").setStyle(tickLength=0)
        self.correction_plot_widget.getAxis("bottom").setStyle(tickLength=0)

        self.base_degree_spin.setRange(0, 1)
        self.base_degree_spin.setValue(1)
        self.transition_degree_spin.setRange(1, 10)
        self.transition_degree_spin.setValue(1)
        self.context_spin.setRange(10, 2000)
        self.context_spin.setSingleStep(10)
        self.context_spin.setValue(100)
        self.loss_combo.addItems(["linear", "soft_l1", "huber", "cauchy", "arctan"])
        self.loss_combo.setCurrentText("soft_l1")

        self.param_frame = QFrame(self)
        self.param_frame.setFrameShape(QFrame.Shape.StyledPanel)
        param_layout = QHBoxLayout(self.param_frame)
        param_layout.setContentsMargins(8, 6, 8, 6)
        param_layout.addWidget(QLabel("鲁棒基线去阶跃"))
        param_layout.addSpacing(8)
        param_layout.addWidget(QLabel("基线阶数："))
        param_layout.addWidget(self.base_degree_spin)
        param_layout.addWidget(QLabel("过渡阶数："))
        param_layout.addWidget(self.transition_degree_spin)
        param_layout.addWidget(QLabel("上下文点数："))
        param_layout.addWidget(self.context_spin)
        param_layout.addWidget(QLabel("损失函数："))
        param_layout.addWidget(self.loss_combo)
        param_layout.addWidget(self.preview_btn)
        param_layout.addWidget(self.apply_btn)

        self.region_frame = QFrame(self)
        self.region_frame.setFrameShape(QFrame.Shape.StyledPanel)
        region_layout = QHBoxLayout(self.region_frame)
        region_layout.setContentsMargins(8, 6, 8, 6)
        region_layout.addWidget(QLabel("区域选择"))
        region_layout.addSpacing(8)
        region_layout.addWidget(self.add_region_btn)
        region_layout.addWidget(self.remove_region_btn)
        region_layout.addWidget(self.clear_region_btn)
        region_layout.addWidget(self.selection_status_label)
        region_layout.addStretch(1)

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
        self.plot_splitter.addWidget(self.correction_plot_widget)
        self.plot_splitter.setStretchFactor(0, 4)
        self.plot_splitter.setStretchFactor(1, 2)
        self.plot_splitter.setSizes([620, 260])

        control_layout = QVBoxLayout()
        control_layout.addWidget(self.param_frame)
        control_layout.addWidget(self.region_frame)
        control_layout.addWidget(self.action_frame)

        layout = QGridLayout(self)
        layout.addWidget(instruction_label, 0, 0, 1, 2)
        layout.addWidget(self.plot_splitter, 1, 0, 1, 2)
        layout.addLayout(control_layout, 2, 0, 1, 2)

    def _connect_signal(self):
        """连接预览、应用、区域选择和撤销等交互行为。"""
        self.preview_btn.clicked.connect(self._preview_fit)
        self.apply_btn.clicked.connect(self._apply_correction)
        self.add_region_btn.clicked.connect(self._add_region)
        self.remove_region_btn.clicked.connect(self._remove_region)
        self.clear_region_btn.clicked.connect(self._clear_selection)
        self.undo_btn.clicked.connect(self._undo)
        self.undo_shortcut.activated.connect(self._undo)
        self.revert_btn.clicked.connect(self._revert)
        self.confirm_btn.clicked.connect(self._confirm_and_close)
        self.signal_plot_widget.double_click_callback = self._add_or_move_region_from_x
        self.signal_plot_widget.right_click_callback = self._handle_plot_right_click

    def _init_plot(self):
        """初始化主图中的原始/当前/基线/校正后曲线，以及下方校正项曲线。"""
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
            pen=pg.mkPen(210, 70, 70, width=1),
            name="当前序列", **curve_opts,
        )
        self.baseline_line = StepPlotWidget.make_curve(
            self.x_data, y_centered,
            pen=pg.mkPen(60, 150, 60, width=1),
            name="拟合基线", **curve_opts,
        )
        self.corrected_line = StepPlotWidget.make_curve(
            self.x_data, y_centered,
            pen=pg.mkPen(40, 90, 200, width=1),
            name="校正后序列", **curve_opts,
        )
        legend = self.signal_plot_item.addLegend()
        legend.anchor((1, 0), (1, 0))
        legend.setBrush(transparent_legend_brush(self))
        legend.setPen(subtle_pen(self))
        self.signal_plot_item.showGrid(x=True, y=True, alpha=0.7)
        self.signal_plot_item.addItem(self.origin_line)
        self.signal_plot_item.addItem(self.y_line)
        self.signal_plot_item.addItem(self.baseline_line)
        self.signal_plot_item.addItem(self.corrected_line)

        self.correction_line = StepPlotWidget.make_curve(
            self.x_data, np.zeros_like(self.y_data),
            pen=pg.mkPen(130, 60, 170, width=1),
            name="校正项", **curve_opts,
        )
        correction_legend = self.correction_plot_item.addLegend()
        correction_legend.anchor((1, 0), (1, 0))
        correction_legend.setBrush(transparent_legend_brush(self))
        correction_legend.setPen(subtle_pen(self))
        self.correction_plot_item.showGrid(x=True, y=True, alpha=0.7)
        self.correction_plot_item.addItem(self.correction_line)
        self.correction_plot_item.addItem(
            pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen(color=(80, 80, 80), style=Qt.PenStyle.DotLine))
        )

    # ------------------------------------------------------------------
    # BaseProcessorWidget 接口
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        self.log_messages.append(msg)

    def _reset_result_state(self):
        self.last_result = None

    def _update_plot(self):
        """根据当前数据和最近一次拟合结果刷新上下两张图。"""
        reference_level = np.nanmean(self.origin_y_data)
        y_centered = self.y_data - reference_level
        origin_centered = self.origin_y_data - reference_level
        self.origin_line.setData(self.x_data, origin_centered)
        self.y_line.setData(self.x_data, y_centered)

        if self.last_result is None:
            self.baseline_line.setData(self.x_data, y_centered)
            self.corrected_line.setData(self.x_data, y_centered)
            self.correction_line.setData(self.x_data, np.zeros_like(self.y_data))
            return

        baseline_centered = self.last_result.baseline - reference_level
        corrected_centered = self.last_result.corrected - reference_level
        self.baseline_line.setData(self.x_data, baseline_centered)
        self.corrected_line.setData(self.x_data, corrected_centered)
        self.correction_line.setData(self.x_data, self.last_result.correction)

    # ------------------------------------------------------------------
    # 区域选择（读）和业务逻辑
    # ------------------------------------------------------------------

    def _read_region(self) -> Optional[tuple[int, int]]:
        if not self.selected_indices:
            self._log("请先在上方主图中添加并调整连续阶跃区域")
            return None
        return int(min(self.selected_indices)), int(max(self.selected_indices))

    def _clear_selection(self):
        self._remove_region()

    def _read_fit_args(self) -> Optional[dict]:
        """在 UI 线程读取拟合参数（含切片数据），返回纯数据 dict。"""
        region = self._read_region()
        if region is None:
            return None
        return {
            "y": self.y_data,
            "region": region,
            "baseline_degree": int(self.base_degree_spin.value()),
            "transition_degree": int(self.transition_degree_spin.value()),
            "context_points": int(self.context_spin.value()),
            "loss": self.loss_combo.currentText(),
        }

    @staticmethod
    def _fit_compute(args: dict) -> RobustBaselineStepResult:
        """纯计算：在 compute 池里执行，不碰任何 Qt 对象。"""
        y = args["y"]
        region = args["region"]
        kwargs = {k: v for k, v in args.items() if k not in ("y", "region")}
        return robust_baseline_destep(y, region, **kwargs)

    def _preview_fit(self):
        args = self._read_fit_args()
        if args is None:
            return
        self._run_heavy(
            lambda: self._fit_compute(args),
            self._finish_preview,
            "鲁棒基线去阶跃（预览）",
        )

    def _finish_preview(self, result):
        if result is None:
            return
        self.last_result = result
        self._update_plot()
        self._log_preview_summary(result)

    def _log_preview_summary(self, result: RobustBaselineStepResult):
        left, right = result.region
        before_left = np.nanmean(self.y_data[max(0, left - 40):left]) if left > 0 else self.y_data[left]
        before_right = np.nanmean(self.y_data[right + 1:min(self.y_data.size, right + 41)]) if right + 1 < self.y_data.size else self.y_data[right]
        after_left = np.nanmean(result.corrected[max(0, left - 40):left]) if left > 0 else result.corrected[left]
        after_right = np.nanmean(result.corrected[right + 1:min(result.corrected.size, right + 41)]) if right + 1 < result.corrected.size else result.corrected[right]
        before_jump = float(before_right - before_left)
        after_jump = float(after_right - after_left)

        self._log(
            f"鲁棒基线预览完成 | region=[{left}, {right}] | "
            f"fit window=[{result.fit_window[0]}, {result.fit_window[1]}] | "
            f"step={result.step_height:.4f} | jump before={before_jump:.4f} | jump after={after_jump:.4f}"
        )
        self._log(
            f"  基线阶数={self.base_degree_spin.value()} | 过渡阶数={self.transition_degree_spin.value()} | "
            f"上下文点数={self.context_spin.value()} | loss={self.loss_combo.currentText()} | "
            f"success={result.success} | cost={result.cost:.6f}"
        )
        self._log("")

    def _apply_correction(self):
        args = self._read_fit_args()
        if args is None:
            return
        self._run_heavy(
            lambda: self._fit_compute(args),
            self._finish_apply,
            "鲁棒基线去阶跃（应用）",
        )

    def _finish_apply(self, result):
        if result is None:
            return
        self._save_history()
        self.y_data = result.corrected.copy()
        self.last_result = result
        self._update_plot()
        self._log_preview_summary(result)
        self._log("已将鲁棒基线去阶跃结果应用到当前通道")
        self._log("")


# ------------------------------------------------------------------
# Standalone demo
# ------------------------------------------------------------------

def _build_demo_channel() -> tuple[EMData, str]:
    npts = 500_000
    dt = pd.Timedelta(seconds=5)
    index = pd.date_range("2024-05-01 00:00:00", periods=npts, freq=dt)
    x = np.arange(npts, dtype=np.float64)
    baseline = 4.0 + 0.00002 * x + 0.18 * np.sin(x / 5000.0)
    left, right = 210_000, 210_120
    step_height = -7.5
    transition = np.zeros(npts, dtype=np.float64)
    ramp_x = np.arange(left, right + 1, dtype=np.float64)
    ramp = step_height * (ramp_x - left) / float(right - left)
    ramp += 0.25 * np.sin(np.linspace(0.0, 5.0 * np.pi, ramp_x.size))
    transition[left:right + 1] = ramp
    transition[right + 1:] = step_height
    noise = 0.03 * np.random.default_rng(7).standard_normal(npts)
    samples = baseline + transition + noise
    series = pd.Series(samples, index=index, name="Ex1")
    demo = DemoEMData("remove-step-baseline-demo", "/tmp/remove_step_by_baseline_demo.mat", "Ex1", series)
    return demo, "Ex1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone preview for the robust baseline step-removal widget.",
    )
    parser.add_argument("--mat", help="可选：加载真实 MAT 文件进行预览")
    parser.add_argument("--channel", help="可选：指定 MAT 文件中的通道名；不指定则使用第一个通道")
    args = parser.parse_args()

    run_standalone_demo(RemoveStepByBaseline, "鲁棒基线去阶跃演示", args.mat, args.channel, _build_demo_channel)


if __name__ == "__main__":
    main()
