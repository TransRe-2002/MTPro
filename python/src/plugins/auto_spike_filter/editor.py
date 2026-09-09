"""Pipeline-slice preview editor for automatic short-pulse filtering."""
from __future__ import annotations


import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from core.algorithms.spike import AutoSpikeDetection, detect_auto_spikes
from plugins.common.base_processor import build_pkg_from_series
from plugins.common.display import decimate_for_display
from utils.qt_bridge import run as run_in_pool
from utils.theme import configure_pyqtgraph_theme, pg_background


configure_pyqtgraph_theme()


class AutoSpikeFilter(QtWidgets.QWidget):
    """预览并确认自动筛选出的短脉冲飞点。"""

    data_requested = QtCore.Signal(int, str)
    result_submitted = QtCore.Signal(int, str, object)
    result_signal = QtCore.Signal(str, object)

    def __init__(self, source: int | pd.Series, channel_name: str, parent=None):
        super().__init__(parent)
        if self.parent() is not None:
            self.setWindowFlag(QtCore.Qt.WindowType.Window)
        self.data_id = source if isinstance(source, int) else 0
        self.ch_name = channel_name
        self._series_source = source if isinstance(source, pd.Series) else None
        self._x_data: np.ndarray | None = None
        self._source: np.ndarray | None = None
        self._detection: AutoSpikeDetection | None = None
        self._preview_params: dict[str, float | int] | None = None
        self._curve = None
        self._candidates = None
        self._protected = None
        self._preview_generation = 0

        self._init_ui()
        self._connect_signals()
        if self._series_source is not None:
            self.load_series(channel_name, self._series_source)

    def wire_data_ready(self, data_ready_signal) -> None:
        data_ready_signal.connect(self._on_data_ready)

    def load_series(self, channel_name: str, series: pd.Series) -> None:
        self.ch_name = channel_name
        self._on_data_ready(self.data_id, channel_name, build_pkg_from_series(channel_name, series))

    def _on_data_ready(self, data_id: int, channel: str, pkg: dict) -> None:
        if data_id != self.data_id or channel != self.ch_name:
            return
        self._x_data = np.asarray(pkg["x"], dtype=np.float64)
        self._source = np.asarray(pkg["y"], dtype=np.float64)
        self._start_preview()

    def _init_ui(self) -> None:
        self.window_spin = QtWidgets.QSpinBox(self)
        self.window_spin.setRange(3, 10_000_000)
        self.window_spin.setValue(101)
        self.overlap_spin = QtWidgets.QDoubleSpinBox(self)
        self.overlap_spin.setRange(0.0, 90.0)
        self.overlap_spin.setDecimals(1)
        self.overlap_spin.setSuffix(" %")
        self.overlap_spin.setValue(50.0)
        self.threshold_spin = QtWidgets.QDoubleSpinBox(self)
        self.threshold_spin.setRange(0.001, 1000.0)
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setSuffix(" σ")
        self.threshold_spin.setValue(10.0)
        self.width_spin = QtWidgets.QSpinBox(self)
        self.width_spin.setRange(1, 1000)
        self.width_spin.setValue(5)

        form = QtWidgets.QFormLayout()
        form.addRow("窗口点数:", self.window_spin)
        form.addRow("窗口重叠:", self.overlap_spin)
        form.addRow("鲁棒阈值:", self.threshold_spin)
        form.addRow("最大脉冲宽度:", self.width_spin)

        self.preview_button = QtWidgets.QPushButton("重新筛选并预览", self)
        self.apply_button = QtWidgets.QPushButton("确认：将短脉冲置为 NaN", self)
        self.apply_button.setEnabled(False)
        self.status_label = QtWidgets.QLabel("等待加载数据…", self)
        self.status_label.setWordWrap(True)

        self.plot_widget = pg.PlotWidget(self, axisItems={"bottom": pg.DateAxisItem()})
        self.plot_widget.setBackground(pg_background(self.plot_widget))
        self.plot_widget.getViewBox().setMouseEnabled(x=True, y=True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)

        controls = QtWidgets.QHBoxLayout()
        controls.addLayout(form)
        controls.addStretch(1)
        controls.addWidget(self.preview_button)
        controls.addWidget(self.apply_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                "红点为将自动置为 NaN 的短脉冲；橙点为长异常段或潜在台阶，仅标记不删除。",
                self,
            )
        )
        layout.addWidget(self.status_label)
        layout.addWidget(self.plot_widget, 1)
        layout.addLayout(controls)

    def _connect_signals(self) -> None:
        self.preview_button.clicked.connect(self._start_preview)
        self.apply_button.clicked.connect(self._apply_detection)
        for control in (self.window_spin, self.overlap_spin, self.threshold_spin, self.width_spin):
            control.valueChanged.connect(self._invalidate_preview)

    def _params(self) -> dict[str, float | int]:
        return {
            "window_points": int(self.window_spin.value()),
            "overlap_percent": float(self.overlap_spin.value()),
            "threshold_sigma": float(self.threshold_spin.value()),
            "max_pulse_width": int(self.width_spin.value()),
        }

    def _invalidate_preview(self, _value=None) -> None:
        self.apply_button.setEnabled(False)

    def _start_preview(self) -> None:
        if self._source is None:
            return
        params = self._params()
        self._preview_generation += 1
        generation = self._preview_generation
        self.preview_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.status_label.setText("正在按重叠窗口筛选跳点…")

        def compute():
            detection = detect_auto_spikes(self._source.copy(), **params)
            return generation, params, detection

        run_in_pool(
            "compute",
            compute,
            on_done=self._on_preview_ready,
            on_error=self._on_preview_failed,
            owner=self,
        )

    def _on_preview_ready(self, payload) -> None:
        generation, params, detection = payload
        if generation != self._preview_generation:
            return
        self._detection = detection
        self._preview_params = params
        self.preview_button.setEnabled(True)
        self.apply_button.setEnabled(True)
        self._draw_preview(detection)
        self.status_label.setText(
            f"扫描 {detection.window_count} 个窗口：候选 {detection.candidate_indices.size} 点，"
            f"自动删除 {detection.removed_indices.size} 点；"
            f"保留长异常段 {len(detection.long_runs)} 个、潜在台阶/边界段 {len(detection.protected_runs)} 个。"
        )

    def apply_plugin_params(self, params: dict) -> None:
        """Restore a Pipeline block's settings before presenting its preview."""

        controls = (
            (self.window_spin, "window_points"),
            (self.overlap_spin, "overlap_percent"),
            (self.threshold_spin, "threshold_sigma"),
            (self.width_spin, "max_pulse_width"),
        )
        changed = False
        for control, name in controls:
            if name in params and control.value() != params[name]:
                blocker = QtCore.QSignalBlocker(control)
                control.setValue(params[name])
                del blocker
                changed = True
        if changed:
            self._invalidate_preview()
        if changed and self._source is not None:
            self._start_preview()

    def _on_preview_failed(self, exc: BaseException) -> None:
        self.preview_button.setEnabled(True)
        self.status_label.setText(f"自动筛选失败：{exc!r}")

    def _draw_preview(self, detection: AutoSpikeDetection) -> None:
        if self._source is None or self._x_data is None:
            return
        self.plot_widget.clear()
        x_display, y_display = decimate_for_display(self._x_data, self._source)
        self._curve = self.plot_widget.plot(
            x_display,
            y_display,
            pen=pg.mkPen("#4a90d9", width=1),
            connect="finite",
        )
        if detection.removed_indices.size:
            indices = detection.removed_indices
            self._candidates = pg.ScatterPlotItem(
                self._x_data[indices], self._source[indices],
                pen=pg.mkPen(None), brush=pg.mkBrush(230, 60, 60, 220),
                size=7, symbol="o",
            )
            self.plot_widget.addItem(self._candidates)
        protected = _expand_runs(detection.long_runs + detection.protected_runs)
        if protected.size:
            self._protected = pg.ScatterPlotItem(
                self._x_data[protected], self._source[protected],
                pen=pg.mkPen(None), brush=pg.mkBrush(240, 160, 30, 220),
                size=6, symbol="t",
            )
            self.plot_widget.addItem(self._protected)

    def _apply_detection(self) -> None:
        if self._source is None or self._detection is None:
            return
        if self._preview_params != self._params():
            self.status_label.setText("参数已经变更，请先重新筛选并预览。")
            self.apply_button.setEnabled(False)
            return
        result = self._source.copy()
        result[self._detection.removed_indices] = np.nan
        self.result_submitted.emit(self.data_id, self.ch_name, result)
        self.result_signal.emit(self.ch_name, result)


def _expand_runs(runs: tuple[tuple[int, int], ...]) -> np.ndarray:
    if not runs:
        return np.array([], dtype=np.intp)
    return np.concatenate([
        np.arange(start, stop + 1, dtype=np.intp)
        for start, stop in runs
    ])
