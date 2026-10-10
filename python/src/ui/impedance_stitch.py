"""Interactive period-boundary stitching for two impedance snapshots."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.impedance_result import ImpedanceResult
from core.impedance_stitch import stitch_banded_results
from emtf.io.zfile import emtf_phase_deg
from output_tools.shared.c_response import c_response_from_bands
from ui.theme import configure_matplotlib_canvas, hide_matplotlib_reset_button


class ImpedanceStitchWidget(QDialog):
    """Choose two sources and a draggable period boundary, then save a splice."""

    result_created = Signal(dict)

    def __init__(self, results: list[ImpedanceResult], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._results = [item for item in results if item.bands is not None]
        self._result_by_id = {item.snapshot_id: item for item in self._results}
        self._dragging_boundary = False
        self._boundary_was_adjusted = False

        self.setWindowTitle("阻抗周期拼接")
        self.setMinimumSize(920, 680)
        self.resize(1100, 780)

        self.short_combo = QComboBox(self)
        self.long_combo = QComboBox(self)
        for item in self._results:
            label = f"{item.name}（{item.method}）"
            self.short_combo.addItem(label, item.snapshot_id)
            self.long_combo.addItem(label, item.snapshot_id)
        if len(self._results) >= 2:
            short_index, long_index = self._initial_source_indices()
            self.short_combo.setCurrentIndex(short_index)
            self.long_combo.setCurrentIndex(long_index)

        self.boundary_spin = QDoubleSpinBox(self)
        self.boundary_spin.setDecimals(6)
        self.boundary_spin.setRange(0.000001, 1.0e12)
        self.boundary_spin.setSuffix(" s")
        self.boundary_spin.setKeyboardTracking(False)
        self.boundary_line = QLabel(
            "分界规则：虚线左侧取短周期来源，右侧取长周期来源；保留原始点和误差，不插值、不平均。",
            self,
        )

        self.view_combo = QComboBox(self)
        self.view_combo.addItem("视电阻率 / 相位", "rho_phase")
        self.view_combo.addItem("倾子", "tipper")
        self.view_combo.addItem("C 响应", "c_response")

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("短周期来源："))
        source_row.addWidget(self.short_combo, 2)
        source_row.addSpacing(8)
        source_row.addWidget(QLabel("长周期来源："))
        source_row.addWidget(self.long_combo, 2)
        source_row.addSpacing(8)
        source_row.addWidget(QLabel("周期分界："))
        source_row.addWidget(self.boundary_spin)

        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("判断视图："))
        view_row.addWidget(self.view_combo)
        view_row.addStretch(1)

        self.figure = Figure(figsize=(10, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        configure_matplotlib_canvas(self.canvas, self.figure, self)
        hide_matplotlib_reset_button(self.canvas)
        self.count_label = QLabel(self)

        self.create_button = QPushButton("生成拼接结果", self)
        self.create_button.setDefault(True)
        self.cancel_button = QPushButton("取消", self)
        self.create_button.clicked.connect(self._create_result)
        self.cancel_button.clicked.connect(self.reject)
        button_row = QHBoxLayout()
        button_row.addWidget(self.count_label, 1)
        button_row.addWidget(self.create_button)
        button_row.addWidget(self.cancel_button)

        layout = QVBoxLayout(self)
        layout.addLayout(source_row)
        layout.addLayout(view_row)
        layout.addWidget(self.boundary_line)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(button_row)

        self.view_combo.currentIndexChanged.connect(self._draw)
        self.short_combo.currentIndexChanged.connect(self._sources_changed)
        self.long_combo.currentIndexChanged.connect(self._sources_changed)
        self.boundary_spin.valueChanged.connect(self._boundary_changed)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        self._set_initial_boundary()
        self._draw()

    def _initial_source_indices(self) -> tuple[int, int]:
        medians = []
        for item in self._results:
            periods = np.asarray(item.bands.periods, dtype=float)
            valid = periods[np.isfinite(periods) & (periods > 0)]
            medians.append(float(np.median(valid)) if len(valid) else float("inf"))
        order = np.argsort(medians, kind="stable")
        return int(order[0]), int(order[-1])

    def _selected_sources(self) -> tuple[ImpedanceResult | None, ImpedanceResult | None]:
        return (
            self._result_by_id.get(str(self.short_combo.currentData())),
            self._result_by_id.get(str(self.long_combo.currentData())),
        )

    @staticmethod
    def _valid_periods(result: ImpedanceResult | None) -> np.ndarray:
        if result is None or result.bands is None:
            return np.asarray([], dtype=float)
        periods = np.asarray(result.bands.periods, dtype=float)
        return periods[np.isfinite(periods) & (periods > 0)]

    def _default_boundary(self) -> float:
        short_periods = self._valid_periods(self._selected_sources()[0])
        long_periods = self._valid_periods(self._selected_sources()[1])
        if not len(short_periods) or not len(long_periods):
            return 1.0
        low = max(float(short_periods.min()), float(long_periods.min()))
        high = min(float(short_periods.max()), float(long_periods.max()))
        if low <= high:
            return float(np.sqrt(low * high))
        short_edge = float(short_periods.max())
        long_edge = float(long_periods.min())
        if short_edge > long_edge:
            short_edge, long_edge = long_edge, short_edge
        return float(np.sqrt(short_edge * long_edge))

    def _set_initial_boundary(self) -> None:
        self.boundary_spin.blockSignals(True)
        self.boundary_spin.setValue(self._default_boundary())
        self.boundary_spin.blockSignals(False)

    def _sources_changed(self, _index: int = -1) -> None:
        if not self._boundary_was_adjusted:
            self._set_initial_boundary()
        self._draw()

    def _boundary_changed(self, _value: float) -> None:
        self._boundary_was_adjusted = True
        self._draw()

    @staticmethod
    def _array(bands, name: str, *, complex_values: bool = False) -> np.ndarray:
        value = getattr(bands, name, None)
        if value is None:
            return np.asarray([], dtype=np.complex128 if complex_values else np.float64)
        return np.asarray(value, dtype=np.complex128 if complex_values else np.float64)

    @staticmethod
    def _safe_errors(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return np.where(np.isfinite(values) & (values >= 0), values, np.nan)

    def _draw_points(
        self,
        axis,
        result: ImpedanceResult,
        source_index: int,
        periods: np.ndarray,
        values: np.ndarray,
        errors: np.ndarray,
        boundary: float,
        label: str,
        marker: str,
        colour: str,
    ) -> int:
        count = min(len(periods), len(values), len(errors))
        periods = periods[:count]
        values = values[:count]
        errors = self._safe_errors(errors[:count])
        source_mask = periods < boundary if source_index == 0 else periods >= boundary
        valid = (
            source_mask & np.isfinite(periods) & (periods > 0)
            & np.isfinite(values)
        )
        if not valid.any():
            return 0
        axis.errorbar(
            periods[valid], values[valid], yerr=errors[valid],
            fmt=marker, linestyle="none", color=colour,
            capsize=3, markersize=5, label=label,
        )
        return int(valid.sum())

    def _draw(self, _index: int = -1) -> None:
        # Clearing shared log axes can temporarily reset their limits to [0, 1].
        for axis in self.figure.axes:
            axis.set_xscale("linear")
            axis.set_yscale("linear")
        self.figure.clear()
        short, long = self._selected_sources()
        mode = str(self.view_combo.currentData())
        boundary = float(self.boundary_spin.value())

        if mode == "c_response":
            axes = self.figure.subplots(2, 2, sharex=True)
            axis_layout = axes.tolist()
            row_modes = ("real", "imag")
            pairs = (
                (("cxy", r"$C_{xy}$", "o", "red"), ("cyx", r"$C_{yx}$", "s", "blue")),
                (("cxx", r"$C_{xx}$", "o", "red"), ("cyy", r"$C_{yy}$", "s", "blue")),
            )
            names = {"cxx": "zxx", "cxy": "zxy", "cyx": "zyx", "cyy": "zyy"}
            for row, part in enumerate(row_modes):
                for column in range(2):
                    axis = axis_layout[row][column]
                    for source_index, result in enumerate((short, long)):
                        if result is None or result.bands is None:
                            continue
                        bands = result.bands
                        periods = np.asarray(bands.periods, dtype=float)
                        components = c_response_from_bands(bands)
                        omega = np.full(periods.shape, np.nan, dtype=float)
                        good_period = np.isfinite(periods) & (periods > 0)
                        omega[good_period] = 2 * np.pi / periods[good_period]
                        for name, display_name, marker, colour in pairs[column]:
                            value = components.get(name)
                            impedance_se = self._array(bands, f"{names[name]}_se")
                            count = min(len(periods), len(value) if value is not None else 0, len(impedance_se))
                            if not count:
                                continue
                            scale = np.full(count, np.nan, dtype=float)
                            valid_omega = np.isfinite(omega[:count]) & (omega[:count] > 0)
                            scale[valid_omega] = 1000.0 / omega[:count][valid_omega]
                            component_values = np.asarray(value[:count])
                            y = component_values.real if part == "real" else component_values.imag
                            yerr = np.abs(scale) * impedance_se[:count]
                            self._draw_points(
                                axis, result, source_index, periods[:count], y, yerr,
                                boundary, display_name, marker, colour,
                            )
            titles = (
                ("交叉分量实部", "对角分量实部"),
                ("交叉分量虚部", "对角分量虚部"),
            )
            for row in range(2):
                for column in range(2):
                    axis = axis_layout[row][column]
                    axis.set_title(titles[row][column])
                    axis.set_ylabel("C (m)")
                    axis.set_xlabel("周期 (s)")
        else:
            axes = self.figure.subplots(2, 1, sharex=True)
            axis_layout = [[axes[0]], [axes[1]]]
            if mode == "rho_phase":
                titles = ("视电阻率", "相位")
                ylabels = ("视电阻率 (Ω·m)", "相位 (°)")
            else:
                titles = ("倾子实部", "倾子虚部")
                ylabels = ("实部", "虚部")

            for source_index, result in enumerate((short, long)):
                if result is None or result.bands is None:
                    continue
                bands = result.bands
                periods = np.asarray(bands.periods, dtype=float)
                if mode == "rho_phase":
                    series = (
                        (0, self._array(bands, "rho_xy"), self._array(bands, "rho_xy_se"), r"$\rho_{xy}$", "o", "red"),
                        (0, self._array(bands, "rho_yx"), self._array(bands, "rho_yx_se"), r"$\rho_{yx}$", "s", "blue"),
                        (1, emtf_phase_deg(self._array(bands, "zxy", complex_values=True)), self._array(bands, "phi_xy_se"), r"$\phi_{xy}$", "o", "red"),
                        (1, emtf_phase_deg(self._array(bands, "zyx", complex_values=True)), self._array(bands, "phi_yx_se"), r"$\phi_{yx}$", "s", "blue"),
                    )
                else:
                    tzx = self._array(bands, "tzx", complex_values=True)
                    tzy = self._array(bands, "tzy", complex_values=True)
                    tzx_se = self._array(bands, "tzx_se")
                    tzy_se = self._array(bands, "tzy_se")
                    series = (
                        (0, tzx.real, tzx_se, r"$T_{zx}$", "o", "red"),
                        (0, tzy.real, tzy_se, r"$T_{zy}$", "s", "blue"),
                        (1, tzx.imag, tzx_se, r"$T_{zx}$", "o", "red"),
                        (1, tzy.imag, tzy_se, r"$T_{zy}$", "s", "blue"),
                    )
                for axis_index, values, errors, label, marker, colour in series:
                    self._draw_points(
                        axis_layout[axis_index][0], result, source_index,
                        periods, values, errors, boundary, label, marker, colour,
                    )

            for axis_index, axis_row in enumerate(axis_layout):
                axis = axis_row[0]
                axis.set_title(titles[axis_index])
                axis.set_ylabel(ylabels[axis_index])
                axis.set_xlabel("周期 (s)")

        for row in axis_layout:
            for axis in row:
                axis.set_xscale("log")
                axis.grid(True, alpha=0.3)
                axis.axvline(boundary, color="#333333", linestyle="--", linewidth=1.3)
                handles, labels = axis.get_legend_handles_labels()
                if handles:
                    unique = dict(zip(labels, handles))
                    axis.legend(unique.values(), unique.keys(), loc="best", fontsize=8, framealpha=0.6)
        self.figure.tight_layout()
        self.canvas.draw_idle()

        short_count = len(self._valid_periods(short)[self._valid_periods(short) < boundary])
        long_periods = self._valid_periods(long)
        long_count = len(long_periods[long_periods >= boundary])
        self.count_label.setText(
            f"分界左侧 {short_count} 个周期点 · 右侧 {long_count} 个周期点；"
            "拖动虚线可调整分界"
        )
        same_source = short is not None and long is not None and short.snapshot_id == long.snapshot_id
        self.create_button.setEnabled(
            short is not None and long is not None and not same_source
            and short_count > 0 and long_count > 0
        )

    def _on_press(self, event) -> None:
        if event.inaxes is None or event.x is None or event.xdata is None:
            return
        if getattr(event, "button", None) != 1:
            return
        boundary_x = event.inaxes.transData.transform((self.boundary_spin.value(), 0))[0]
        self._dragging_boundary = abs(float(event.x) - boundary_x) <= 10

    def _on_motion(self, event) -> None:
        if not self._dragging_boundary or event.xdata is None or event.xdata <= 0:
            return
        self.boundary_spin.setValue(float(event.xdata))

    def _on_release(self, _event) -> None:
        self._dragging_boundary = False

    def _create_result(self) -> None:
        short, long = self._selected_sources()
        if short is None or long is None or short.bands is None or long.bands is None:
            return
        try:
            bands = stitch_banded_results(
                short.bands,
                long.bands,
                self.boundary_spin.value(),
            )
        except ValueError as error:
            QMessageBox.warning(self, "无法拼接", str(error))
            return

        payload = {
            "method": "stitched impedance",
            "name": f"{short.name} + {long.name} 拼接",
            "bands": bands,
            "station_name": short.station_name or bands.station_name,
            "remote_station_name": (
                short.remote_station_name
                if short.remote_station_name == long.remote_station_name
                else ""
            ),
            "latitude": short.latitude,
            "longitude": short.longitude,
            "elevation": short.elevation,
            "parameters": {
                "stitch_boundary_period_s": self.boundary_spin.value(),
                "short_source_id": short.snapshot_id,
                "short_source_name": short.name,
                "long_source_id": long.snapshot_id,
                "long_source_name": long.name,
                "rule": "short_period_lt_boundary; long_period_gte_boundary",
                "interpolation": False,
                "averaging": False,
            },
        }
        self.result_created.emit(payload)
        self.accept()
