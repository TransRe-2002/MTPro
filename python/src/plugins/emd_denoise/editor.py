"""Parameter editor for EMD selected-IMF reconstruction."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from base.plugin_runtime import run_plugin_process
from base.qt_bridge import run as run_in_pool
from plugin_api import complete_params
from plugins.common.plotting import StepPlotWidget
from utils.decimate import minmax_decimate
from utils.series import dti_to_numpy
from utils.theme import subtle_pen, transparent_legend_brush


class EMDDenoiseEditor(QtWidgets.QWidget):
    def __init__(self, host, params: Mapping[str, object], parent=None, *, block=None):
        super().__init__(parent)
        self._host = host
        self._block = block if block is not None else host.block
        self._generation = 0
        self._busy = False
        self._preview_result = None
        self._preview_params = None
        self._source = self._block.series.to_numpy(dtype=np.float64, copy=True)
        self._x_values = dti_to_numpy(self._block.series.index)
        values = complete_params(host.plugin.info, params)

        self.max_imfs_spin = QtWidgets.QSpinBox(self)
        self.max_imfs_spin.setRange(1, 32)
        self.max_imfs_spin.setValue(int(values["max_imfs"]))
        self.max_siftings_spin = QtWidgets.QSpinBox(self)
        self.max_siftings_spin.setRange(1, 10000)
        self.max_siftings_spin.setValue(int(values["max_siftings"]))
        self.tolerance_spin = QtWidgets.QDoubleSpinBox(self)
        self.tolerance_spin.setRange(1.0e-12, 1.0)
        self.tolerance_spin.setDecimals(12)
        self.tolerance_spin.setValue(float(values["sift_tolerance"]))
        self.min_segment_spin = QtWidgets.QSpinBox(self)
        self.min_segment_spin.setRange(4, 10_000_000)
        self.min_segment_spin.setValue(int(values["min_segment_points"]))
        self.keep_imfs_edit = QtWidgets.QLineEdit(str(values["keep_imfs"]), self)
        self.keep_residue_check = QtWidgets.QCheckBox(self)
        self.keep_residue_check.setChecked(bool(values["keep_residue"]))

        note = QtWidgets.QLabel(
            "IMF 0 是最先提取的高频分量。保留 IMF 可填 all 或 1,2,3；"
            "如按 LMT 的高频 IMF 抑制思路，可保留 1,2,... 并保留低频残差。"
            "两端 NaN 会裁掉；最多插值 10 个连续 NaN，较长缺口会分段处理；结果恢复原缺失位置。",
            self,
        )
        note.setWordWrap(True)
        self.comparison_plot = StepPlotWidget(self)
        self.comparison_plot.setLabel("left", "幅值")
        legend = self.comparison_plot.addLegend()
        legend.setBrush(transparent_legend_brush(self.comparison_plot))
        legend.setPen(subtle_pen(self.comparison_plot))
        self.input_curve = self.comparison_plot.plot(
            pen=pg.mkPen("#777f89", width=1), name="Block 输入", connect="finite",
        )
        self.output_curve = self.comparison_plot.plot(
            pen=pg.mkPen("#1672d4", width=1.5), name="EMD 重构预览", connect="finite",
        )
        self.difference_plot = StepPlotWidget(self)
        self.difference_plot.setLabel("left", "输入 − 预览")
        self.difference_plot.setXLink(self.comparison_plot)
        self.difference_curve = self.difference_plot.plot(
            pen=pg.mkPen("#dc7820", width=1), connect="finite",
        )
        self._set_curve(self.input_curve, self._source)
        plots = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)
        plots.addWidget(self.comparison_plot)
        plots.addWidget(self.difference_plot)
        plots.setStretchFactor(0, 2)
        plots.setStretchFactor(1, 1)

        form = QtWidgets.QGridLayout()
        fields = (
            ("最大 IMF 数:", self.max_imfs_spin),
            ("单 IMF 最大筛分次数:", self.max_siftings_spin),
            ("筛分相对变化阈值:", self.tolerance_spin),
            ("最短有效段点数:", self.min_segment_spin),
            ("保留 IMF:", self.keep_imfs_edit),
            ("保留低频残差:", self.keep_residue_check),
        )
        for position, (label, control) in enumerate(fields):
            row, column = divmod(position, 3)
            form.addWidget(QtWidgets.QLabel(label, self), row, column * 2)
            form.addWidget(control, row, column * 2 + 1)
            form.setColumnStretch(column * 2 + 1, 1)

        self.status_label = QtWidgets.QLabel("输入已加载。点击“计算预览”运行 EMD。", self)
        self.status_label.setWordWrap(True)
        self.preview_button = QtWidgets.QPushButton("计算预览", self)
        self.preview_button.clicked.connect(self._start_preview)
        self.apply_button = QtWidgets.QPushButton("确认应用", self)
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._apply)
        actions = QtWidgets.QHBoxLayout()
        actions.addWidget(self.status_label, 1)
        actions.addWidget(self.preview_button)
        actions.addWidget(self.apply_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(note)
        layout.addWidget(plots, 1)
        layout.addLayout(form)
        layout.addLayout(actions)

        for control in (self.max_imfs_spin, self.max_siftings_spin,
                        self.tolerance_spin, self.min_segment_spin):
            control.valueChanged.connect(self._invalidate_preview)
        self.keep_imfs_edit.textChanged.connect(self._invalidate_preview)
        self.keep_residue_check.toggled.connect(self._invalidate_preview)

    def _params(self) -> dict[str, object]:
        return {
            "max_imfs": int(self.max_imfs_spin.value()),
            "max_siftings": int(self.max_siftings_spin.value()),
            "sift_tolerance": float(self.tolerance_spin.value()),
            "min_segment_points": int(self.min_segment_spin.value()),
            "keep_imfs": self.keep_imfs_edit.text().strip() or "all",
            "keep_residue": bool(self.keep_residue_check.isChecked()),
        }

    def _set_curve(self, curve, values: np.ndarray) -> None:
        x, y = minmax_decimate(self._x_values, values, target_bins=5000)
        if len(values) > 10_000:
            gaps = np.flatnonzero(np.diff(np.r_[False, ~np.isfinite(values), False]))[::2]
            if gaps.size:
                _, first = np.unique(gaps * 5000 // len(values), return_index=True)
                gap_x = self._x_values[gaps[first]]
                x = np.concatenate((x, gap_x))
                y = np.concatenate((y, np.full(gap_x.size, np.nan)))
                order = np.argsort(x, kind="stable")
                x, y = x[order], y[order]
        curve.setData(x, y, connect="finite")

    def _invalidate_preview(self, *_args) -> None:
        self._generation += 1
        self._preview_result = None
        self._preview_params = None
        self.output_curve.clear()
        self.difference_curve.clear()
        self.apply_button.setEnabled(False)
        self.status_label.setText(
            "参数已变化；正在等待上一次计算结束。" if self._busy
            else "参数已变化，请重新计算预览。"
        )

    def _start_preview(self) -> None:
        if self._busy:
            return
        self._generation += 1
        generation = self._generation
        params = self._params()
        plugin, block = self._host.plugin, self._block
        self._busy = True
        self._preview_result = None
        self._preview_params = None
        self.output_curve.clear()
        self.difference_curve.clear()
        self.preview_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.status_label.setText("正在后台计算 EMD 预览，长数据可能需要一些时间…")

        def compute():
            result, completed = run_plugin_process(plugin, block, params)
            return generation, result, completed

        run_in_pool(
            "compute", compute, on_done=self._on_preview_ready,
            on_error=lambda exc: self._on_preview_failed(generation, exc), owner=self,
        )

    def _on_preview_ready(self, payload) -> None:
        generation, result, params = payload
        self._busy = False
        self.preview_button.setEnabled(True)
        if generation != self._generation:
            self.status_label.setText("参数已变化，请重新计算预览。")
            return
        segments = result.diagnostics.get("processed_segments", [])
        if not segments:
            self.status_label.setText("没有满足最短长度的有效数据段，未进行 EMD 处理。")
            return
        self._preview_result = result
        self._preview_params = dict(params)
        output = result.series.to_numpy(dtype=np.float64, copy=False)
        self._set_curve(self.output_curve, output)
        self._set_curve(self.difference_curve, self._source - output)
        self.comparison_plot.enableAutoRange()
        self.difference_plot.enableAutoRange()
        count = len(segments[0].get("imf_energies", []))
        skipped = len(result.diagnostics.get("skipped_segments", []))
        status = f"预览完成：首段分解出 {count} 个 IMF；已处理 {len(segments)} 段"
        if skipped:
            status += f"，{skipped} 个过短片段保持原样"
        status += "。"
        if params["keep_imfs"].strip().lower() in {"all", "*"} and params["keep_residue"]:
            status += "保留全部 IMF 和残差时，差值通常仅为数值误差。"
        self.status_label.setText(status)
        self.apply_button.setEnabled(True)

    def _on_preview_failed(self, generation: int, exc: object) -> None:
        self._busy = False
        self.preview_button.setEnabled(True)
        if generation != self._generation:
            self.status_label.setText("参数已变化，请重新计算预览。")
        else:
            self.status_label.setText(f"预览失败：{exc}")

    def _apply(self) -> None:
        if self._preview_result is None or self._preview_params != self._params():
            return
        result, params = self._preview_result, self._preview_params
        self.apply_button.setEnabled(False)
        accept = getattr(self._host, "accept_preview", None)
        if callable(accept):
            accept(result, params)
        else:
            self._host.apply_params(params)
