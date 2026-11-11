"""Interactive preview for secular-variation baseline subtraction."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from base.qt_bridge import run as run_in_pool
from core.algorithms.secular_variation import SecularVariationResult, remove_local_secular_variation_series, remove_secular_variation_series
from plugin_api import complete_params
from plugins.common.plotting import StepPlotWidget
from utils.series import dti_to_numpy
from utils.theme import subtle_pen, transparent_legend_brush


class SecularVariationEditor(QtWidgets.QWidget):
    """Preview the estimated baseline before sending the transform to Pipeline."""

    def __init__(self, host, block, params: Mapping[str, object], parent=None):
        super().__init__(parent)
        self._host = host
        self._block = block
        self._source = block.series
        self._preview: SecularVariationResult | None = None
        self._preview_generation = 0
        self._x_values = _time_values(block.series.index)

        values = complete_params(host.plugin.info, params)
        self._region = values.get("_pipeline_step_range")
        self.window_spin = QtWidgets.QSpinBox(self)
        self.window_spin.setRange(1, 100_000_000)
        self.window_spin.setValue(int(values["smoothing_window_points"]))
        self.spacing_spin = QtWidgets.QSpinBox(self)
        self.spacing_spin.setRange(1, 100_000_000)
        self.spacing_spin.setValue(int(values["anchor_spacing_points"]))
        self.valid_fraction_spin = QtWidgets.QDoubleSpinBox(self)
        self.valid_fraction_spin.setRange(0.0, 1.0)
        self.valid_fraction_spin.setDecimals(3)
        self.valid_fraction_spin.setSingleStep(0.05)
        self.valid_fraction_spin.setValue(float(values["min_valid_fraction"]))

        self.preview_button = QtWidgets.QPushButton("预览平滑背景", self)
        self.block_view_button = QtWidgets.QPushButton("查看区间", self)
        self.full_view_button = QtWidgets.QPushButton("查看全序列", self)
        self.apply_button = QtWidgets.QPushButton("应用背景相减", self)
        self.apply_button.setEnabled(False)
        self.status_label = QtWidgets.QLabel("调整参数后预览；不会填补缺失值。", self)
        self.status_label.setWordWrap(True)
        self._init_plot()
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        note = QtWidgets.QLabel(
            "去除缓慢背景变化（SV）：在固定间隔的锚点处计算局部均值，使用原 TS.m 的 7 点"
            "平滑核后以三次样条连成背景曲线。上图显示原始序列与背景，下图显示相减预览。"
            "蓝色区域为背景估计区间；区间两侧延续边界校正量。可切换查看区间或全序列。"
            "窗口和间隔均以采样点计。" if self._region is not None else
            "去除缓慢背景变化（SV）：在固定间隔的锚点处计算局部均值，使用原 TS.m 的 7 点"
            "平滑核后以三次样条连成背景曲线。上图显示原始序列与背景，下图显示相减预览。",
            self,
        )
        note.setWordWrap(True)
        form = QtWidgets.QFormLayout()
        form.addRow("平滑窗口（点）:", self.window_spin)
        form.addRow("锚点间隔（点）:", self.spacing_spin)
        form.addRow("锚点最小有效比例:", self.valid_fraction_spin)

        controls = QtWidgets.QHBoxLayout()
        controls.addLayout(form)
        controls.addStretch(1)
        if self._region is not None:
            controls.addWidget(self.block_view_button)
            controls.addWidget(self.full_view_button)
        controls.addWidget(self.preview_button)
        controls.addWidget(self.apply_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(note)
        layout.addWidget(self.status_label)
        layout.addWidget(self.source_plot, 1)
        layout.addWidget(self.cleaned_plot, 1)
        layout.addLayout(controls)

    def _connect_signals(self) -> None:
        self.preview_button.clicked.connect(self._start_preview)
        self.apply_button.clicked.connect(self._apply)
        self.block_view_button.clicked.connect(self._show_block_range)
        self.full_view_button.clicked.connect(self._show_full_range)
        for control in (self.window_spin, self.spacing_spin, self.valid_fraction_spin):
            control.valueChanged.connect(self._invalidate_preview)

    def _init_plot(self) -> None:
        source_values = self._source.to_numpy(dtype=np.float64, copy=True)
        self.source_plot = StepPlotWidget(self)
        self.cleaned_plot = StepPlotWidget(self)
        for plot in (self.source_plot, self.cleaned_plot):
            item = plot.getPlotItem()
            item.showGrid(x=True, y=True, alpha=0.35)
            legend = item.addLegend()
            legend.anchor((1, 0), (1, 0))
            legend.setBrush(transparent_legend_brush(plot))
            legend.setPen(subtle_pen(plot))

        self.source_curve = StepPlotWidget.make_curve(
            self._x_values,
            source_values,
            pen=pg.mkPen(100, 100, 100, width=1),
            name="原始序列",
        )
        self.baseline_curve = StepPlotWidget.make_curve(
            [], [], pen=pg.mkPen(220, 120, 0, width=2), name="SV 背景"
        )
        self.cleaned_curve = StepPlotWidget.make_curve(
            [], [], pen=pg.mkPen(0, 114, 178, width=1), name="相减预览"
        )
        self.source_plot.addItem(self.source_curve)
        self.source_plot.addItem(self.baseline_curve)
        self.cleaned_plot.addItem(self.cleaned_curve)
        self.cleaned_plot.setXLink(self.source_plot)
        if self._region is not None:
            start, end = map(int, self._region)
            if 0 <= start <= end < len(self._x_values):
                for plot in (self.source_plot, self.cleaned_plot):
                    area = pg.LinearRegionItem(
                        values=(self._x_values[start], self._x_values[end]),
                        movable=False, brush=pg.mkBrush(70, 130, 220, 30),
                        pen=pg.mkPen(70, 130, 220, 130),
                    )
                    area.setZValue(-10)
                    plot.addItem(area, ignoreBounds=True)
                    plot.getViewBox().setAutoVisible(y=True)
                self._show_block_range()

    def _show_block_range(self) -> None:
        if self._region is None:
            return
        start, end = map(int, self._region)
        if not (0 <= start <= end < len(self._x_values)):
            return
        left, right = float(self._x_values[start]), float(self._x_values[end])
        if left == right:
            left, right = left - 0.5, right + 0.5
        self.source_plot.setXRange(left, right, padding=0.1)

    def _show_full_range(self) -> None:
        if len(self._x_values) < 2:
            return
        self.source_plot.setXRange(float(self._x_values[0]), float(self._x_values[-1]), padding=0.02)

    def _params(self) -> dict[str, int | float]:
        params = {
            "smoothing_window_points": int(self.window_spin.value()),
            "anchor_spacing_points": int(self.spacing_spin.value()),
            "min_valid_fraction": float(self.valid_fraction_spin.value()),
        }
        if self._region is not None:
            params.update({"region": list(self._region), "effect_both_sides": True})
        return params

    def _invalidate_preview(self, _value=None) -> None:
        self._preview = None
        self._preview_generation += 1
        self.apply_button.setEnabled(False)
        self.baseline_curve.setData([], [])
        self.cleaned_curve.setData([], [])
        self.status_label.setText("参数已变化，请重新预览。")

    def _start_preview(self) -> None:
        params = self._params()
        self._preview_generation += 1
        generation = self._preview_generation
        source = self._source
        self.preview_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.status_label.setText("正在估计平滑背景…")

        def compute():
            if params.get("effect_both_sides"):
                options = {key: value for key, value in params.items() if key not in ("region", "effect_both_sides")}
                return remove_local_secular_variation_series(source, tuple(params["region"]), **options)
            return remove_secular_variation_series(source, **params)

        def complete(result) -> None:
            self.preview_button.setEnabled(True)
            if generation != self._preview_generation:
                self.status_label.setText("参数已变化，请重新预览。")
                return
            self._preview = result
            self._show_preview(result)

        def failed(exc) -> None:
            self.preview_button.setEnabled(True)
            if generation != self._preview_generation:
                self.status_label.setText("参数已变化，请重新预览。")
            else:
                self.status_label.setText(f"预览失败：{exc}")

        run_in_pool("compute", compute, on_done=complete, on_error=failed, owner=self)

    def _show_preview(self, result: SecularVariationResult) -> None:
        self.baseline_curve.setData(
            self._x_values,
            result.estimated_baseline.to_numpy(dtype=np.float64),
        )
        self.cleaned_curve.setData(
            self._x_values,
            result.cleaned.to_numpy(dtype=np.float64),
        )
        self.apply_button.setEnabled(result.status == "applied")
        if result.status == "applied":
            skipped = (
                f"；{result.skipped_segments} 个有效数据段的锚点不足，保持原样"
                if result.skipped_segments else ""
            )
            self.status_label.setText(
                f"预览完成：{result.valid_anchor_count}/{result.anchor_count} 个锚点有效；"
                f"处理 {result.processed_segments} 个有效数据段{skipped}；原始 NaN 已保留。"
            )
        else:
            self.status_label.setText(
                "可用锚点少于两个，未应用相减。请缩短间隔/窗口或选择更长的有效数据段。"
            )

    def _apply(self) -> None:
        if self._preview is None or self._preview.status != "applied":
            return
        self.apply_button.setEnabled(False)
        self.status_label.setText("正在应用背景相减…")
        self._host.apply_params(self._params())


def _time_values(index) -> np.ndarray:
    if hasattr(index, "asi8"):
        return dti_to_numpy(index)
    return np.arange(len(index), dtype=np.float64)
