from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from scipy.optimize import least_squares

pg.setConfigOptions(antialias=True)
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


@dataclass
class RobustStepFitResult:
    corrected: np.ndarray
    correction: np.ndarray
    baseline: np.ndarray
    fit_window: tuple[int, int]
    region: tuple[int, int]
    baseline_coeffs: np.ndarray
    transition_coeffs: np.ndarray
    step_height: float
    success: bool
    cost: float
    message: str


def _normalize_index(index: np.ndarray, start: int, stop: int) -> np.ndarray:
    center = 0.5 * (start + stop)
    scale = max(1.0, 0.5 * (stop - start))
    return (index - center) / scale


def _saturated_transition(index: np.ndarray, left: int, right: int, degree: int) -> np.ndarray:
    if degree <= 0:
        raise ValueError("transition degree must be positive")

    if right <= left:
        t = (index >= left).astype(np.float64)
    else:
        t = np.zeros_like(index, dtype=np.float64)
        inside = (index >= left) & (index <= right)
        t[index > right] = 1.0
        t[inside] = (index[inside] - left) / float(right - left)

    cols = [t ** power for power in range(1, degree + 1)]
    return np.column_stack(cols)


def _poly_design(index: np.ndarray, start: int, stop: int, degree: int) -> np.ndarray:
    x = _normalize_index(index.astype(np.float64), start, stop)
    cols = [x ** power for power in range(degree + 1)]
    return np.column_stack(cols)


def robust_remove_step_by_region(
    samples: np.ndarray,
    region: tuple[int, int],
    *,
    baseline_degree: int = 1,
    transition_degree: int = 1,
    context_points: int = 200,
    loss: str = "soft_l1",
    f_scale: float | None = None,
) -> RobustStepFitResult:
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError("samples must be a 1D array")
    if data.size < 3:
        raise ValueError("samples is too short")

    left, right = sorted((int(region[0]), int(region[1])))
    left = max(0, left)
    right = min(data.size - 1, right)
    if left == right:
        right = min(data.size - 1, left + 1)

    fit_start = max(0, left - int(context_points))
    fit_stop = min(data.size, right + int(context_points) + 1)
    fit_index = np.arange(fit_start, fit_stop, dtype=np.int64)
    if fit_index.size < 3:
        raise ValueError("fit window is too short")

    finite_mask = np.isfinite(data[fit_start:fit_stop])
    if int(np.count_nonzero(finite_mask)) < baseline_degree + transition_degree + 2:
        raise ValueError("not enough finite samples in fit window")

    baseline_design = _poly_design(fit_index, fit_start, fit_stop - 1, baseline_degree)
    transition_design = _saturated_transition(fit_index, left, right, transition_degree)
    design = np.column_stack([baseline_design, transition_design])

    y_fit = data[fit_start:fit_stop][finite_mask]
    design_fit = design[finite_mask]
    initial, *_ = np.linalg.lstsq(design_fit, y_fit, rcond=None)

    if f_scale is None:
        diff = np.diff(y_fit)
        scale = np.median(np.abs(diff - np.median(diff))) * 1.4826 if diff.size else 0.0
        f_scale = max(scale, 1e-3)

    result = least_squares(
        lambda coeffs: design_fit @ coeffs - y_fit,
        initial,
        loss=loss,
        f_scale=float(f_scale),
    )

    coeffs = result.x
    n_base = baseline_design.shape[1]
    baseline_coeffs = coeffs[:n_base]
    transition_coeffs = coeffs[n_base:]

    all_index = np.arange(data.size, dtype=np.int64)
    baseline_all = _poly_design(all_index, fit_start, fit_stop - 1, baseline_degree) @ baseline_coeffs
    correction = _saturated_transition(all_index, left, right, transition_degree) @ transition_coeffs

    corrected = data.copy()
    finite_all = np.isfinite(corrected)
    corrected[finite_all] = corrected[finite_all] - correction[finite_all]

    return RobustStepFitResult(
        corrected=corrected,
        correction=correction,
        baseline=baseline_all,
        fit_window=(fit_start, fit_stop - 1),
        region=(left, right),
        baseline_coeffs=baseline_coeffs,
        transition_coeffs=transition_coeffs,
        step_height=float(np.sum(transition_coeffs)),
        success=bool(result.success),
        cost=float(result.cost),
        message=result.message,
    )


def build_demo_signal(
    *,
    seed: int = 7,
    npts: int = 800,
    step_height: float = 6.0,
    noise_scale: float = 0.08,
) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    rng = np.random.default_rng(seed)
    index = np.arange(npts, dtype=np.float64)

    baseline = 0.0012 * index + 0.15 * np.sin(index / 28.0)
    left, right = int(0.41 * npts), int(0.46 * npts)
    right = max(left + 2, right)

    transition = np.zeros(npts, dtype=np.float64)
    transition[right + 1:] = step_height
    ramp_index = np.arange(left, right + 1, dtype=np.float64)
    ramp = step_height * (ramp_index - left) / float(right - left)
    ramp += 0.18 * np.sin(np.linspace(0.0, 4.0 * np.pi, ramp_index.size))
    transition[left:right + 1] = ramp

    noise = noise_scale * rng.standard_normal(npts)
    signal = baseline + transition + noise
    return signal, (left, right), baseline


def summarize_jump(samples: np.ndarray, region: tuple[int, int], avg_window: int = 40) -> tuple[float, float]:
    left, right = region
    left_start = max(0, left - avg_window)
    right_stop = min(samples.size, right + 1 + avg_window)
    left_mean = float(np.nanmean(samples[left_start:left]))
    right_mean = float(np.nanmean(samples[right + 1:right_stop]))
    return left_mean, right_mean


def region_values_to_indices(x_data: np.ndarray, values: tuple[float, float]) -> tuple[int, int]:
    left_value, right_value = sorted(values)
    left = int(np.searchsorted(x_data, left_value, side="left"))
    right = int(np.searchsorted(x_data, right_value, side="right") - 1)
    left = max(0, min(left, x_data.size - 2))
    right = max(left + 1, min(right, x_data.size - 1))
    return left, right


class StepRemovalDemoWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Robust 去阶跃草稿演示")
        self.resize(1400, 900)

        self.samples: np.ndarray
        self.true_baseline: np.ndarray
        self.x_data: np.ndarray
        self.default_region: tuple[int, int]
        self.last_result: RobustStepFitResult | None = None

        self.plot_widget = pg.PlotWidget()
        self.correction_plot = pg.PlotWidget()
        self.region_item: pg.LinearRegionItem | None = None
        self.fit_window_item: pg.LinearRegionItem | None = None

        self.original_curve = None
        self.true_baseline_curve = None
        self.fitted_baseline_curve = None
        self.corrected_curve = None
        self.correction_curve = None

        self.base_degree_spin = QtWidgets.QSpinBox()
        self.transition_degree_spin = QtWidgets.QSpinBox()
        self.context_spin = QtWidgets.QSpinBox()
        self.seed_spin = QtWidgets.QSpinBox()
        self.step_height_spin = QtWidgets.QDoubleSpinBox()
        self.noise_spin = QtWidgets.QDoubleSpinBox()
        self.loss_combo = QtWidgets.QComboBox()
        self.auto_apply_check = QtWidgets.QCheckBox("自动刷新")
        self.apply_btn = QtWidgets.QPushButton("应用拟合")
        self.reset_region_btn = QtWidgets.QPushButton("重置 Region")
        self.regenerate_btn = QtWidgets.QPushButton("重生成数据")
        self.status_label = QtWidgets.QLabel()
        self.summary_text = QtWidgets.QPlainTextEdit()

        self._init_ui()
        self._connect_signals()
        self._load_demo_signal()

    def _init_ui(self) -> None:
        self.base_degree_spin.setRange(0, 4)
        self.base_degree_spin.setValue(1)

        self.transition_degree_spin.setRange(1, 4)
        self.transition_degree_spin.setValue(2)

        self.context_spin.setRange(10, 500)
        self.context_spin.setValue(120)
        self.context_spin.setSingleStep(10)

        self.seed_spin.setRange(0, 9999)
        self.seed_spin.setValue(7)

        self.step_height_spin.setRange(0.5, 20.0)
        self.step_height_spin.setDecimals(2)
        self.step_height_spin.setSingleStep(0.25)
        self.step_height_spin.setValue(6.0)

        self.noise_spin.setRange(0.0, 2.0)
        self.noise_spin.setDecimals(3)
        self.noise_spin.setSingleStep(0.01)
        self.noise_spin.setValue(0.08)

        self.loss_combo.addItems(["linear", "soft_l1", "huber", "cauchy", "arctan"])
        self.loss_combo.setCurrentText("soft_l1")

        self.auto_apply_check.setChecked(True)

        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumBlockCount(50)
        self.summary_text.setMinimumHeight(140)

        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.correction_plot.showGrid(x=True, y=True, alpha=0.25)
        self.correction_plot.setXLink(self.plot_widget)
        self.plot_widget.addLegend(offset=(10, 10))
        self.correction_plot.addLegend(offset=(10, 10))

        controls = QtWidgets.QGridLayout()
        controls.addWidget(QtWidgets.QLabel("基线阶数"), 0, 0)
        controls.addWidget(self.base_degree_spin, 0, 1)
        controls.addWidget(QtWidgets.QLabel("过渡阶数"), 0, 2)
        controls.addWidget(self.transition_degree_spin, 0, 3)
        controls.addWidget(QtWidgets.QLabel("上下文点数"), 0, 4)
        controls.addWidget(self.context_spin, 0, 5)
        controls.addWidget(QtWidgets.QLabel("损失函数"), 0, 6)
        controls.addWidget(self.loss_combo, 0, 7)
        controls.addWidget(self.auto_apply_check, 0, 8)

        controls.addWidget(QtWidgets.QLabel("随机种子"), 1, 0)
        controls.addWidget(self.seed_spin, 1, 1)
        controls.addWidget(QtWidgets.QLabel("台阶高度"), 1, 2)
        controls.addWidget(self.step_height_spin, 1, 3)
        controls.addWidget(QtWidgets.QLabel("噪声幅度"), 1, 4)
        controls.addWidget(self.noise_spin, 1, 5)
        controls.addWidget(self.regenerate_btn, 1, 6)
        controls.addWidget(self.reset_region_btn, 1, 7)
        controls.addWidget(self.apply_btn, 1, 8)

        plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        plot_splitter.addWidget(self.plot_widget)
        plot_splitter.addWidget(self.correction_plot)
        plot_splitter.setStretchFactor(0, 3)
        plot_splitter.setStretchFactor(1, 2)

        layout = QtWidgets.QVBoxLayout(self)
        instruction = QtWidgets.QLabel(
            "拖动红色 Region 指定阶跃过渡区，调整参数后预览 robust 基线校正。"
            "顶部显示原始/基线/校正后，底部显示拟合得到的校正项。"
        )
        layout.addWidget(instruction)
        layout.addLayout(controls)
        layout.addWidget(self.status_label)
        layout.addWidget(plot_splitter, 1)
        layout.addWidget(self.summary_text)

    def _connect_signals(self) -> None:
        self.apply_btn.clicked.connect(self.apply_fit)
        self.reset_region_btn.clicked.connect(self.reset_region)
        self.regenerate_btn.clicked.connect(self.regenerate_data)

        for widget in (
            self.base_degree_spin,
            self.transition_degree_spin,
            self.context_spin,
            self.loss_combo,
        ):
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentTextChanged.connect(self._maybe_auto_apply)
            else:
                widget.valueChanged.connect(self._maybe_auto_apply)

        self.seed_spin.valueChanged.connect(self._on_generation_parameter_changed)
        self.step_height_spin.valueChanged.connect(self._on_generation_parameter_changed)
        self.noise_spin.valueChanged.connect(self._on_generation_parameter_changed)

    def _on_generation_parameter_changed(self) -> None:
        if self.auto_apply_check.isChecked():
            self.regenerate_data()

    def _maybe_auto_apply(self) -> None:
        if self.auto_apply_check.isChecked():
            self.apply_fit()

    def _load_demo_signal(self) -> None:
        self.samples, self.default_region, self.true_baseline = build_demo_signal(
            seed=int(self.seed_spin.value()),
            step_height=float(self.step_height_spin.value()),
            noise_scale=float(self.noise_spin.value()),
        )
        self.x_data = np.arange(self.samples.size, dtype=np.float64)
        self.last_result = None

        self.plot_widget.clear()
        self.correction_plot.clear()
        self.plot_widget.addLegend(offset=(10, 10))
        self.correction_plot.addLegend(offset=(10, 10))

        self.fit_window_item = pg.LinearRegionItem(values=self.default_region, movable=False)
        self.fit_window_item.setBrush(pg.mkBrush(100, 100, 100, 35))
        for line in self.fit_window_item.lines:
            line.setPen(pg.mkPen(color=(140, 140, 140), width=1))
        self.fit_window_item.setZValue(-20)
        self.plot_widget.addItem(self.fit_window_item)

        self.region_item = pg.LinearRegionItem(values=self.default_region, movable=True)
        self.region_item.setBrush(pg.mkBrush(220, 70, 70, 55))
        for line in self.region_item.lines:
            line.setPen(pg.mkPen(color=(200, 50, 50), width=3))
        self.region_item.setBounds((0.0, float(self.x_data[-1])))
        self.region_item.setZValue(10)
        self.plot_widget.addItem(self.region_item)
        self.region_item.sigRegionChanged.connect(self._update_region_status)
        self.region_item.sigRegionChangeFinished.connect(self._maybe_auto_apply)

        self.original_curve = self.plot_widget.plot(
            self.x_data,
            self.samples,
            pen=pg.mkPen(color=(210, 70, 70), width=1.2),
            name="original",
        )
        self.true_baseline_curve = self.plot_widget.plot(
            self.x_data,
            self.true_baseline,
            pen=pg.mkPen(color=(120, 120, 120), width=1, style=QtCore.Qt.PenStyle.DashLine),
            name="true baseline",
        )
        self.fitted_baseline_curve = self.plot_widget.plot(
            self.x_data,
            self.samples,
            pen=pg.mkPen(color=(60, 150, 60), width=1.3),
            name="fitted baseline",
        )
        self.corrected_curve = self.plot_widget.plot(
            self.x_data,
            self.samples,
            pen=pg.mkPen(color=(40, 90, 200), width=1.5),
            name="corrected",
        )

        self.correction_curve = self.correction_plot.plot(
            self.x_data,
            np.zeros_like(self.samples),
            pen=pg.mkPen(color=(130, 60, 170), width=1.5),
            name="correction",
        )
        zero_line = pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen(color=(80, 80, 80), style=QtCore.Qt.PenStyle.DotLine))
        self.correction_plot.addItem(zero_line)

        self.plot_widget.setLabel("left", "value")
        self.plot_widget.setLabel("bottom", "sample index")
        self.correction_plot.setLabel("left", "correction")
        self.correction_plot.setLabel("bottom", "sample index")

        self._update_region_status()
        self.apply_fit()

    def _current_region(self) -> tuple[int, int]:
        if self.region_item is None:
            return self.default_region
        return region_values_to_indices(self.x_data, tuple(self.region_item.getRegion()))

    def _update_region_status(self) -> None:
        left, right = self._current_region()
        width = right - left + 1
        self.status_label.setText(f"当前 Region: [{left}, {right}] | 采样点数={width}")

    def regenerate_data(self) -> None:
        self._load_demo_signal()

    def reset_region(self) -> None:
        if self.region_item is None:
            return
        self.region_item.blockSignals(True)
        self.region_item.setRegion(self.default_region)
        self.region_item.blockSignals(False)
        self._update_region_status()
        self.apply_fit()

    def apply_fit(self) -> None:
        region = self._current_region()
        try:
            result = robust_remove_step_by_region(
                self.samples,
                region,
                baseline_degree=int(self.base_degree_spin.value()),
                transition_degree=int(self.transition_degree_spin.value()),
                context_points=int(self.context_spin.value()),
                loss=self.loss_combo.currentText(),
            )
        except Exception as exc:
            self.status_label.setText(f"拟合失败: {exc}")
            self.summary_text.setPlainText(f"拟合失败\n{exc}")
            return

        self.last_result = result

        if self.fit_window_item is not None:
            self.fit_window_item.setRegion(result.fit_window)

        self.fitted_baseline_curve.setData(self.x_data, result.baseline)
        self.corrected_curve.setData(self.x_data, result.corrected)
        self.correction_curve.setData(self.x_data, result.correction)

        before_left, before_right = summarize_jump(self.samples, result.region)
        after_left, after_right = summarize_jump(result.corrected, result.region)
        before_jump = before_right - before_left
        after_jump = after_right - after_left

        self.status_label.setText(
            f"当前 Region: [{result.region[0]}, {result.region[1]}] | "
            f"fit window=[{result.fit_window[0]}, {result.fit_window[1]}] | "
            f"step={result.step_height:.4f} | jump before={before_jump:.4f} | jump after={after_jump:.4f}"
        )

        summary = [
            "Robust 去阶跃草稿",
            f"region = {result.region}",
            f"fit_window = {result.fit_window}",
            f"baseline_degree = {self.base_degree_spin.value()}",
            f"transition_degree = {self.transition_degree_spin.value()}",
            f"context_points = {self.context_spin.value()}",
            f"loss = {self.loss_combo.currentText()}",
            f"success = {result.success}",
            f"message = {result.message}",
            f"step_height = {result.step_height:.6f}",
            f"jump_before = {before_jump:.6f}",
            f"jump_after = {after_jump:.6f}",
            f"cost = {result.cost:.6f}",
            f"baseline_coeffs = {np.array2string(result.baseline_coeffs, precision=4)}",
            f"transition_coeffs = {np.array2string(result.transition_coeffs, precision=4)}",
        ]
        self.summary_text.setPlainText("\n".join(summary))


def run_demo(print_only: bool = False) -> RobustStepFitResult:
    samples, region, _ = build_demo_signal()
    result = robust_remove_step_by_region(
        samples,
        region,
        baseline_degree=1,
        transition_degree=2,
        context_points=120,
        loss="soft_l1",
    )

    before_left, before_right = summarize_jump(samples, region)
    after_left, after_right = summarize_jump(result.corrected, region)

    print(f"region={region}, fit_window={result.fit_window}")
    print(f"robust_success={result.success}, step_height={result.step_height:.4f}, cost={result.cost:.4f}")
    print(f"jump_before={before_right - before_left:.4f}")
    print(f"jump_after={after_right - after_left:.4f}")

    if not print_only:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        widget = StepRemovalDemoWidget()
        widget.show()
        app.exec()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive robust step-removal draft demo using PySide6 + PyQtGraph.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="run the numeric demo only and print a short summary",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="launch the GUI and exit automatically after a short delay",
    )
    args = parser.parse_args()

    if args.print_only:
        run_demo(print_only=True)
        return

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    widget = StepRemovalDemoWidget()
    widget.show()

    if args.smoke_test:
        QtCore.QTimer.singleShot(300, app.quit)

    app.exec()


if __name__ == "__main__":
    main()
