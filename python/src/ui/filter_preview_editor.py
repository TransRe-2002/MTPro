"""Reusable original/preview/difference editor for lightweight filters."""
from __future__ import annotations

from typing import Any
from collections.abc import Mapping

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from base.plugin_runtime import run_plugin_process
from plugin_api import ChannelBlock, ParamSpec, complete_params
from plugins.common.display import decimate_for_display
from plugins.common.plotting import StepPlotWidget
from utils.qt_bridge import run as run_in_pool
from utils.theme import subtle_pen, transparent_legend_brush


class FilterPreviewEditor(QWidget):
    """Schema-driven filter editor with deterministic asynchronous preview."""

    PREVIEW_DELAY_MS = 250

    def __init__(
        self,
        host,
        block: ChannelBlock,
        params: Mapping[str, Any],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._host = host
        self._block = block
        self._editors: dict[str, tuple[QWidget, ParamSpec]] = {}
        self._preview_generation = 0
        self.preview_result = None
        self.preview_params: dict[str, Any] | None = None

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(self.PREVIEW_DELAY_MS)
        self._preview_timer.timeout.connect(self._start_preview)

        values = complete_params(host.plugin.info, params)
        self._build_ui(values)
        QTimer.singleShot(0, self._start_preview)

    def _build_ui(self, values: Mapping[str, Any]) -> None:
        info = self._host.plugin.info
        root = QVBoxLayout(self)

        if info.description:
            description = QLabel(info.description, self)
            description.setWordWrap(True)
            root.addWidget(description)

        form = QFormLayout()
        for spec in info.params_schema:
            editor = self._create_field(spec, values.get(spec.name, spec.default))
            if editor is None:
                continue
            form.addRow(spec.label, editor)
            self._editors[spec.name] = (editor, spec)
            self._connect_field(editor)
        root.addLayout(form)

        self.status_label = QLabel("正在计算初始预览…", self)
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.comparison_plot = StepPlotWidget(self)
        self.comparison_plot.setObjectName("filterComparisonPlot")
        self.comparison_plot.setLabel("left", "幅值")
        comparison_legend = self.comparison_plot.addLegend()
        comparison_legend.setBrush(transparent_legend_brush(self.comparison_plot))
        comparison_legend.setPen(subtle_pen(self.comparison_plot))
        self.original_curve = StepPlotWidget.make_curve(
            [],
            [],
            pen=pg.mkPen("#8a8f98", width=1),
            name="Block 输入",
            decimate=False,
        )
        self.preview_curve = StepPlotWidget.make_curve(
            [],
            [],
            pen=pg.mkPen("#2f80ed", width=2),
            name="参数预览",
            decimate=False,
        )
        self.comparison_plot.addItem(self.original_curve)
        self.comparison_plot.addItem(self.preview_curve)

        self.difference_plot = StepPlotWidget(self)
        self.difference_plot.setObjectName("filterDifferencePlot")
        self.difference_plot.setLabel("left", "输入 − 预览")
        self.difference_plot.setXLink(self.comparison_plot)
        self.difference_curve = StepPlotWidget.make_curve(
            [],
            [],
            pen=pg.mkPen("#e67e22", width=1.5),
            decimate=False,
        )
        self.difference_plot.addItem(self.difference_curve)

        plots = QSplitter(Qt.Orientation.Vertical, self)
        plots.addWidget(self.comparison_plot)
        plots.addWidget(self.difference_plot)
        plots.setStretchFactor(0, 2)
        plots.setStretchFactor(1, 1)
        root.addWidget(plots, 1)

        self.metrics_label = QLabel("等待预览统计…", self)
        self.metrics_label.setWordWrap(True)
        root.addWidget(self.metrics_label)

        actions = QHBoxLayout()
        actions.addWidget(QLabel("上图显示", self))
        self.display_mode_combo = QComboBox(self)
        self.display_mode_combo.addItems(
            ("叠加对比", "仅 Block 输入", "仅参数预览")
        )
        self.display_mode_combo.currentIndexChanged.connect(
            self._update_curve_visibility
        )
        actions.addWidget(self.display_mode_combo)
        actions.addStretch(1)
        self.preview_button = QPushButton("重新计算预览", self)
        self.preview_button.clicked.connect(self._start_preview)
        actions.addWidget(self.preview_button)
        self.apply_button = QPushButton("确认应用", self)
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._apply)
        actions.addWidget(self.apply_button)
        root.addLayout(actions)

    def _update_curve_visibility(self, _index=None) -> None:
        mode = self.display_mode_combo.currentText()
        self.original_curve.setVisible(mode != "仅参数预览")
        self.preview_curve.setVisible(mode != "仅 Block 输入")

    def _create_field(self, spec: ParamSpec, value: Any) -> QWidget | None:
        if spec.type is float:
            editor = QDoubleSpinBox(self)
            editor.setDecimals(8)
            editor.setRange(
                float(spec.minimum) if spec.minimum is not None else -1.0e100,
                float(spec.maximum) if spec.maximum is not None else 1.0e100,
            )
            editor.setValue(float(value))
            return editor
        if spec.type is int:
            editor = QSpinBox(self)
            editor.setRange(
                int(spec.minimum) if spec.minimum is not None else -2147483648,
                int(spec.maximum) if spec.maximum is not None else 2147483647,
            )
            editor.setValue(int(value))
            return editor
        if spec.type is bool:
            editor = QCheckBox(self)
            editor.setChecked(bool(value))
            return editor
        if spec.type is str and spec.options:
            editor = QComboBox(self)
            editor.addItems(spec.options)
            editor.setCurrentText(str(value))
            return editor
        if spec.type is str:
            editor = QLineEdit(str(value), self)
            return editor
        return None

    def _connect_field(self, editor: QWidget) -> None:
        if isinstance(editor, (QSpinBox, QDoubleSpinBox)):
            editor.valueChanged.connect(self._schedule_preview)
        elif isinstance(editor, QCheckBox):
            editor.toggled.connect(self._schedule_preview)
        elif isinstance(editor, QComboBox):
            editor.currentIndexChanged.connect(self._schedule_preview)
        elif isinstance(editor, QLineEdit):
            editor.textChanged.connect(self._schedule_preview)

    def params(self) -> dict[str, Any]:
        values = complete_params(self._host.plugin.info, {})
        for name, (editor, spec) in self._editors.items():
            if spec.type is float:
                values[name] = editor.value()
            elif spec.type is int:
                values[name] = editor.value()
            elif spec.type is bool:
                values[name] = editor.isChecked()
            elif spec.type is str and spec.options:
                values[name] = editor.currentText()
            elif spec.type is str:
                values[name] = editor.text()
        return values

    def _schedule_preview(self, _value=None) -> None:
        self._preview_generation += 1
        self.preview_result = None
        self.preview_params = None
        self.apply_button.setEnabled(False)
        self.status_label.setText("参数已变化，正在更新预览…")
        self._preview_timer.start()

    def _start_preview(self) -> None:
        self._preview_timer.stop()
        self._preview_generation += 1
        generation = self._preview_generation
        params = self.params()
        self.preview_result = None
        self.preview_params = None
        self.preview_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.status_label.setText("正在使用正式处理算法计算预览…")

        def compute():
            try:
                result, completed = run_plugin_process(
                    self._host.plugin,
                    self._block,
                    params,
                )
                return generation, result, completed, None
            except Exception as exc:
                return generation, None, None, exc

        run_in_pool(
            "compute",
            compute,
            on_done=self._on_preview_ready,
            owner=self,
        )

    def _on_preview_ready(self, payload) -> None:
        generation, result, params, error = payload
        if generation != self._preview_generation:
            return
        self.preview_button.setEnabled(True)
        if error is not None:
            self.status_label.setText(f"预览失败：{error}")
            self.metrics_label.setText("请调整参数后重试。")
            return

        self.preview_result = result
        self.preview_params = dict(params)
        self._update_plots(result.series.to_numpy(dtype=np.float64, copy=True))
        self.metrics_label.setText(self._metrics_text(result))
        self.status_label.setText("预览已更新；确认后会把同一参数写入 Pipeline。")
        self.apply_button.setEnabled(True)

    def _update_plots(self, preview: np.ndarray) -> None:
        source = self._block.series.to_numpy(dtype=np.float64, copy=True)
        difference = source - preview
        index = self._block.series.index
        if hasattr(index, "asi8"):
            x_values = np.asarray(index.asi8, dtype=np.float64) / 1.0e9
        else:
            x_values = np.arange(source.size, dtype=np.float64)

        x_source, y_source = decimate_for_display(x_values, source)
        x_preview, y_preview = decimate_for_display(x_values, preview)
        x_difference, y_difference = decimate_for_display(x_values, difference)
        self.original_curve.setData(x_source, y_source)
        self.preview_curve.setData(x_preview, y_preview)
        self.difference_curve.setData(x_difference, y_difference)
        self.comparison_plot.enableAutoRange()
        self.difference_plot.enableAutoRange()

    def _metrics_text(self, result) -> str:
        source = self._block.series.to_numpy(dtype=np.float64, copy=False)
        preview = result.series.to_numpy(dtype=np.float64, copy=False)
        difference = source - preview

        def rms(values: np.ndarray) -> float:
            finite = values[np.isfinite(values)]
            return float(np.sqrt(np.mean(finite * finite))) if finite.size else float("nan")

        def peak_to_peak(values: np.ndarray) -> float:
            finite = values[np.isfinite(values)]
            return float(np.ptp(finite)) if finite.size else float("nan")

        valid_count = int(np.count_nonzero(np.isfinite(preview)))
        diagnostics = ", ".join(
            f"{key}={len(value) if isinstance(value, (list, tuple)) else value}"
            for key, value in result.diagnostics.items()
        )
        text = (
            f"有效点 {valid_count}/{preview.size} | "
            f"RMS 输入 {rms(source):.6g} → 预览 {rms(preview):.6g} | "
            f"差值 RMS {rms(difference):.6g} | "
            f"峰峰值 {peak_to_peak(source):.6g} → {peak_to_peak(preview):.6g}"
        )
        return f"{text} | {diagnostics}" if diagnostics else text

    def _apply(self) -> None:
        if self.preview_result is None or self.preview_params != self.params():
            self._schedule_preview()
            return
        self.apply_button.setEnabled(False)
        self.status_label.setText("正在确认并写入 Pipeline…")
        self._host.apply_params(self.preview_params)
