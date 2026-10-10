"""Single-channel Welch power-spectrum viewer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import welch
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ui.theme import configure_matplotlib_canvas


def _finite_runs(values: np.ndarray) -> list[np.ndarray]:
    finite = np.isfinite(values)
    edges = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    return [values[start:end] for start, end in edges.reshape(-1, 2) if end - start >= 8]


def compute_welch_psd(
    values,
    sample_rate_hz: float,
    nperseg: int = 4096,
    overlap_percent: int = 50,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute a gap-aware Welch PSD by averaging contiguous finite runs."""
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size < 8:
        raise ValueError("至少需要 8 个样点")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("采样率必须大于 0")

    requested = max(8, min(int(nperseg), samples.size))
    runs = _finite_runs(samples)
    eligible = [run for run in runs if len(run) >= requested]
    if not eligible:
        if not runs:
            raise ValueError("通道没有连续的有效数据")
        longest = max(runs, key=len)
        requested = min(requested, len(longest))
        eligible = [longest]

    weighted = None
    frequencies = None
    total_weight = 0
    for run in eligible:
        noverlap = min(requested - 1, requested * int(overlap_percent) // 100)
        freq, density = welch(
            run,
            fs=float(sample_rate_hz),
            window=window,
            nperseg=requested,
            noverlap=noverlap,
            detrend="linear",
            scaling="density",
        )
        weight = len(run)
        frequencies = freq
        weighted = density * weight if weighted is None else weighted + density * weight
        total_weight += weight

    density = weighted / total_weight
    valid = np.isfinite(frequencies) & (frequencies > 0) & np.isfinite(density) & (density > 0)
    diagnostics = {
        "total_samples": int(samples.size),
        "finite_samples": int(np.isfinite(samples).sum()),
        "runs_used": len(eligible),
        "nperseg": requested,
    }
    return frequencies[valid], density[valid], diagnostics


class EnergySpectrumWidget(QWidget):
    def __init__(self, values, sample_rate_hz: float, label: str, units: str = "", parent=None):
        super().__init__(parent)
        # 与“显示曲线”一致：保留父对象用于生命周期管理，但作为独立顶层窗口显示。
        if self.parent() is not None:
            self.setWindowFlag(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.values = np.asarray(values, dtype=np.float64).copy()
        self.sample_rate_hz = float(sample_rate_hz)
        self.channel_label = label
        self.units = units
        self.setWindowTitle(f"能量谱 - {label}")
        self.resize(980, 680)

        self.segment_spin = QSpinBox(self)
        self.segment_spin.setRange(8, max(8, min(len(self.values), 1_048_576)))
        self.segment_spin.setValue(min(4096, max(8, len(self.values))))
        self.segment_spin.setSingleStep(128)
        self.overlap_spin = QSpinBox(self)
        self.overlap_spin.setRange(0, 95)
        self.overlap_spin.setValue(50)
        self.window_combo = QComboBox(self)
        self.window_combo.addItems(["hann", "hamming", "blackman", "boxcar"])
        self.axis_combo = QComboBox(self)
        self.axis_combo.addItems(["频率 (Hz)", "周期 (s)"])
        self.axis_combo.currentIndexChanged.connect(self.refresh_plot)
        refresh = QPushButton("重新计算", self)
        refresh.clicked.connect(self.refresh_plot)
        export = QPushButton("导出图像…", self)
        export.clicked.connect(self._export_figure)

        controls = QHBoxLayout()
        form = QFormLayout()
        form.addRow("Welch 分段点数", self.segment_spin)
        form.addRow("重叠率 (%)", self.overlap_spin)
        controls.addLayout(form)
        form2 = QFormLayout()
        form2.addRow("窗函数", self.window_combo)
        form2.addRow("横轴", self.axis_combo)
        controls.addLayout(form2)
        controls.addStretch(1)
        controls.addWidget(refresh)
        controls.addWidget(export)

        self.figure = Figure(figsize=(8, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        configure_matplotlib_canvas(self.canvas, self.figure, self)
        self.status = QLabel(self)
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.status)
        self.refresh_plot()

    def refresh_plot(self) -> None:
        try:
            frequencies, density, info = compute_welch_psd(
                self.values,
                self.sample_rate_hz,
                self.segment_spin.value(),
                self.overlap_spin.value(),
                self.window_combo.currentText(),
            )
        except Exception as exc:
            self.figure.clear()
            self.status.setText(f"计算失败：{exc}")
            self.canvas.draw()
            return

        x = frequencies
        xlabel = "Frequency (Hz)"
        if self.axis_combo.currentIndex() == 1:
            x = 1.0 / frequencies
            order = np.argsort(x)
            x, density = x[order], density[order]
            xlabel = "Period (s)"
        self.figure.clear()
        axis = self.figure.subplots(1, 1)
        axis.loglog(x, density, color="#3b82f6", linewidth=1.0)
        axis.set_xlabel(xlabel)
        ylabel = f"PSD ({self.units}²/Hz)" if self.units else "PSD"
        axis.set_ylabel(ylabel)
        axis.set_title(f"{self.channel_label} — Welch 功率谱密度")
        axis.grid(True, which="both", alpha=0.25)
        configure_matplotlib_canvas(self.canvas, self.figure, self)
        self.canvas.draw()
        ratio = 100.0 * info["finite_samples"] / info["total_samples"]
        self.status.setText(
            f"采样率 {self.sample_rate_hz:g} Hz｜有效样点 {ratio:.2f}%｜"
            f"连续段 {info['runs_used']}｜Welch 分段 {info['nperseg']} 点"
        )

    def _export_figure(self) -> None:
        default_name = f"{self.channel_label.replace(' ', '_')}_psd.png"
        filename, _ = QFileDialog.getSaveFileName(self, "导出能量谱", default_name, "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if filename:
            self.figure.savefig(Path(filename), dpi=200, bbox_inches="tight")
