import numpy as np
import pandas as pd
import xarray as xr
import pytest
import tomllib
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QSplitter

from core.em_data import EMData
from processor.remove_wavelet_denoise import RemoveWaveletDenoise
from processor.remove_step_by_diff import RemoveStepByDiff as RemoveStep
from processor.remove_step_by_baseline import RemoveStepByBaseline
from processor.remove_step_by_window import RemoveStepByWindow
from processor._modifier_constraint import apply_modifier_axis_constraint
from processor.step_algorithms import (
    remove_diff_steps_by_count,
    robust_baseline_destep,
    wavelet_local_depulse,
    wavelet_multiscale_denoise,
    windowed_mean_destep_with_diagnostics,
)
from processor.step_plot_widget import StepPlotWidget as RemoveStepPlotWidget


class DemoData(EMData):
    def __init__(self):
        super().__init__("/tmp/demo.mat")
        self.name = "demo"
        self.npts = 5
        self.start_time = pd.Timestamp("2024-01-01 00:00:00")
        self.end_time = pd.Timestamp("2024-01-01 00:00:20")
        self.dt = pd.Timedelta(seconds=5)
        self.datetime_index = pd.date_range(self.start_time, periods=5, freq="5s")
        self.kp_data = None
        series = pd.Series([1.0, 2.0, 3.0, 5.0, 8.0], index=self.datetime_index)
        self.data = xr.Dataset(coords={"time": self.datetime_index})
        self.data["Ex1"] = xr.DataArray(
            series.to_numpy(dtype=np.float64),
            dims=("time",),
            coords={"time": self.datetime_index},
        )

    def restore_data(self, ch: str):
        return None


class StepDemoData(EMData):
    def __init__(self):
        super().__init__("/tmp/step_demo.mat")
        self.name = "step-demo"
        self.npts = 6
        self.start_time = pd.Timestamp("2024-01-01 00:00:00")
        self.end_time = pd.Timestamp("2024-01-01 00:00:25")
        self.dt = pd.Timedelta(seconds=5)
        self.datetime_index = pd.date_range(self.start_time, periods=6, freq="5s")
        self.kp_data = None
        series = pd.Series([0.0, 0.0, 0.0, 10.0, 10.0, 10.0], index=self.datetime_index)
        self.data = xr.Dataset(coords={"time": self.datetime_index})
        self.data["Ex1"] = xr.DataArray(
            series.to_numpy(dtype=np.float64),
            dims=("time",),
            coords={"time": self.datetime_index},
        )

    def restore_data(self, ch: str):
        return None


class StepDemoDataWithNaN(EMData):
    def __init__(self):
        super().__init__("/tmp/step_demo_nan.mat")
        self.name = "step-demo-nan"
        self.npts = 6
        self.start_time = pd.Timestamp("2024-01-01 00:00:00")
        self.end_time = pd.Timestamp("2024-01-01 00:00:25")
        self.dt = pd.Timedelta(seconds=5)
        self.datetime_index = pd.date_range(self.start_time, periods=6, freq="5s")
        self.kp_data = None
        series = pd.Series([0.0, np.nan, 0.0, 10.0, 10.0, 10.0], index=self.datetime_index)
        self.data = xr.Dataset(coords={"time": self.datetime_index})
        self.data["Ex1"] = xr.DataArray(
            series.to_numpy(dtype=np.float64),
            dims=("time",),
            coords={"time": self.datetime_index},
        )

    def restore_data(self, ch: str):
        return None


class MultiStepDemoData(EMData):
    def __init__(self):
        super().__init__("/tmp/multi_step_demo.mat")
        self.name = "multi-step-demo"
        self.npts = 7
        self.start_time = pd.Timestamp("2024-01-01 00:00:00")
        self.end_time = pd.Timestamp("2024-01-01 00:00:30")
        self.dt = pd.Timedelta(seconds=5)
        self.datetime_index = pd.date_range(self.start_time, periods=7, freq="5s")
        self.kp_data = None
        series = pd.Series([0.0, 0.0, 0.0, 10.0, 10.0, 20.0, 20.0], index=self.datetime_index)
        self.data = xr.Dataset(coords={"time": self.datetime_index})
        self.data["Ex1"] = xr.DataArray(
            series.to_numpy(dtype=np.float64),
            dims=("time",),
            coords={"time": self.datetime_index},
        )

    def restore_data(self, ch: str):
        return None


class PulseDemoData(EMData):
    def __init__(self):
        super().__init__("/tmp/pulse_demo.mat")
        self.name = "pulse-demo"
        self.npts = 4096
        self.start_time = pd.Timestamp("2024-01-01 00:00:00")
        self.dt = pd.Timedelta(seconds=1)
        self.datetime_index = pd.date_range(self.start_time, periods=self.npts, freq="1s")
        self.end_time = self.datetime_index[-1]
        self.kp_data = None

        x = np.arange(self.npts, dtype=np.float64)
        baseline = 0.001 * x + 0.1 * np.sin(x / 80.0)
        pulse = np.zeros(self.npts, dtype=np.float64)
        start = 1800
        rise_width = 20
        decay_width = 160
        amplitude = 5.0
        rise = np.linspace(0.0, amplitude, rise_width, endpoint=False, dtype=np.float64)
        decay = amplitude * np.exp(-4.0 * np.linspace(0.0, 1.0, decay_width, endpoint=False, dtype=np.float64))
        pulse[start:start + rise_width] = rise
        pulse[start + rise_width:start + rise_width + decay_width] = decay
        series = baseline + pulse

        self.data = xr.Dataset(coords={"time": self.datetime_index})
        self.data["Ex1"] = xr.DataArray(
            series.astype(np.float64),
            dims=("time",),
            coords={"time": self.datetime_index},
        )

    def restore_data(self, ch: str):
        return None


def test_remove_step_plot_widget_uses_strong_focus_and_axis_constraints(qtbot):
    plot = RemoveStepPlotWidget()
    qtbot.addWidget(plot)

    assert plot.focusPolicy() == Qt.FocusPolicy.StrongFocus

    view_box = plot.getViewBox()
    apply_modifier_axis_constraint(view_box, Qt.KeyboardModifier.ShiftModifier)
    assert tuple(view_box.state["mouseEnabled"]) == (True, False)

    apply_modifier_axis_constraint(view_box, Qt.KeyboardModifier.AltModifier)
    assert tuple(view_box.state["mouseEnabled"]) == (False, True)

    apply_modifier_axis_constraint(view_box, Qt.KeyboardModifier.NoModifier)
    assert tuple(view_box.state["mouseEnabled"]) == (True, True)


def test_remove_step_sets_focus_to_plot_on_show(qtbot):
    data = DemoData()
    widget = RemoveStep(data, "Ex1")
    qtbot.addWidget(widget)

    widget.show()
    qtbot.wait(100)

    assert isinstance(widget.plot_widget, RemoveStepPlotWidget)
    assert widget.signal_plot_widget.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert widget.one_step_times.text() == "5"
    assert widget.plain_text.maximumWidth() == 320
    assert widget.threshold_frame is not None
    assert widget.multi_step_frame is not None
    assert widget.manual_frame is not None
    assert widget.action_frame is not None


def test_remove_step_processing_frames_are_horizontal(qtbot):
    data = DemoData()
    widget = RemoveStep(data, "Ex1")
    qtbot.addWidget(widget)

    control_layout = widget.layout().itemAtPosition(2, 0).layout()
    processing_frame_layout = control_layout.itemAt(0).layout()

    assert isinstance(processing_frame_layout, QHBoxLayout)
    assert processing_frame_layout.itemAt(0).widget() is widget.threshold_frame
    assert processing_frame_layout.itemAt(1).widget() is widget.multi_step_frame
    assert processing_frame_layout.itemAt(2).widget() is widget.manual_frame


def test_remove_step_plots_are_in_vertical_splitter(qtbot):
    data = DemoData()
    widget = RemoveStep(data, "Ex1")
    qtbot.addWidget(widget)

    assert isinstance(widget.plot_splitter, QSplitter)
    assert widget.plot_splitter.orientation() == Qt.Orientation.Vertical
    assert widget.plot_splitter.opaqueResize() is False
    assert widget.plot_splitter.childrenCollapsible() is False
    assert widget.plot_splitter.widget(0) is widget.signal_plot_widget
    assert widget.plot_splitter.widget(1) is widget.diff_plot_widget


def test_remove_step_plots_keep_x_range_in_sync_but_not_y_range(qtbot):
    data = DemoData()
    widget = RemoveStep(data, "Ex1")
    qtbot.addWidget(widget)
    widget.show()
    qtbot.wait(50)

    widget.signal_plot_item.vb.setRange(xRange=(widget.x_data[1], widget.x_data[3]), yRange=(-10.0, 10.0), padding=0)
    qtbot.wait(10)
    signal_x = widget.signal_plot_item.vb.viewRange()[0]
    diff_x = widget.diff_plot_widget.getViewBox().viewRange()[0]
    signal_y = widget.signal_plot_item.vb.viewRange()[1]
    diff_y = widget.diff_plot_widget.getViewBox().viewRange()[1]

    assert np.allclose(signal_x, diff_x)
    assert not np.allclose(signal_y, diff_y)

    widget.diff_plot_widget.getViewBox().setRange(
        xRange=(widget.x_data[2], widget.x_data[4]),
        yRange=(-1.0, 1.0),
        padding=0,
    )
    qtbot.wait(10)
    signal_x = widget.signal_plot_item.vb.viewRange()[0]
    diff_x = widget.diff_plot_widget.getViewBox().viewRange()[0]

    assert np.allclose(signal_x, diff_x)


def test_remove_step_window_threshold_uses_windowed_destep(qtbot):
    data = StepDemoData()
    widget = RemoveStepByWindow(data, "Ex1")
    qtbot.addWidget(widget)

    widget.de_step_threshold.setText("5")
    widget.avg_window_edit.setText("2")
    widget.max_steps_edit.setText("5")
    widget._de_step_by_threshold()
    # 重计算在 compute 池后台执行，等结果回到主线程
    qtbot.waitUntil(lambda: "去除了 1 个台阶" in widget.plain_text.toPlainText(), timeout=10000)

    np.testing.assert_allclose(widget.y_data, np.zeros(6), atol=1e-8)
    np.testing.assert_allclose(widget.diff_y_data, np.zeros(5), atol=1e-8)
    assert "去除了 1 个台阶" in widget.plain_text.toPlainText()


def test_remove_step_window_uses_fixed_default_threshold_like_matlab_destep(qtbot):
    data = StepDemoData()
    widget = RemoveStepByWindow(data, "Ex1")
    qtbot.addWidget(widget)

    assert widget.de_step_threshold.text() == f"{widget.DEFAULT_MIN_OFFSET:.4f}"


def test_remove_step_window_threshold_uses_default_max_steps_in_log(qtbot):
    data = MultiStepDemoData()
    widget = RemoveStepByWindow(data, "Ex1")
    qtbot.addWidget(widget)

    assert widget.max_steps_edit.text() == str(widget.DEFAULT_MAX_STEPS)
    widget.de_step_threshold.setText("5")
    widget.avg_window_edit.setText("2")
    widget._de_step_by_threshold()
    qtbot.waitUntil(
        lambda: f"最多台阶={widget.DEFAULT_MAX_STEPS}" in widget.plain_text.toPlainText(),
        timeout=10000,
    )

    assert f"最多台阶={widget.DEFAULT_MAX_STEPS}" in widget.plain_text.toPlainText()


def test_remove_step_window_user_max_steps_limit_is_applied(qtbot):
    data = MultiStepDemoData()
    widget = RemoveStepByWindow(data, "Ex1")
    qtbot.addWidget(widget)

    expected, applied_steps, _ = windowed_mean_destep_with_diagnostics(
        widget.y_data,
        min_offset=5.0,
        avg_window=2,
        max_steps=1,
    )

    widget.de_step_threshold.setText("5")
    widget.avg_window_edit.setText("2")
    widget.max_steps_edit.setText("1")
    widget._de_step_by_threshold()

    assert len(applied_steps) == 1
    np.testing.assert_allclose(widget.y_data, expected, atol=1e-8)
    assert "去除了 1 个台阶" in widget.plain_text.toPlainText()
    assert "最多台阶=1" in widget.plain_text.toPlainText()


def test_remove_step_one_step_matches_matlab_style_algorithm(qtbot):
    data = MultiStepDemoData()
    widget = RemoveStep(data, "Ex1")
    qtbot.addWidget(widget)

    corrected_one, removed_one = remove_diff_steps_by_count(widget.y_data, 1)
    corrected_two, removed_two = remove_diff_steps_by_count(widget.y_data, 2)

    np.testing.assert_allclose(corrected_one, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0]), atol=1e-8)
    np.testing.assert_allclose(corrected_two, np.zeros(7), atol=1e-8)
    assert len(removed_one) == 1
    assert len(removed_two) == 2


def test_remove_step_multi_button_uses_matlab_style_one_step(qtbot):
    data = MultiStepDemoData()
    widget = RemoveStep(data, "Ex1")
    qtbot.addWidget(widget)

    widget.one_step_times.setText("1")
    widget._remove_by_count()

    np.testing.assert_allclose(widget.y_data, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0]), atol=1e-8)
    assert "共去除 1 个台阶" in widget.plain_text.toPlainText()


def test_remove_step_window_destep_handles_large_series(qtbot):
    data = DemoData()
    widget = RemoveStepByWindow(data, "Ex1")
    qtbot.addWidget(widget)

    npts = 200_000
    samples = np.linspace(0.0, 1.0, npts, dtype=np.float64)
    samples += 0.05 * np.sin(np.linspace(0.0, 400.0 * np.pi, npts, dtype=np.float64))
    samples[npts // 2:] += 25.0

    corrected, applied_steps, _ = windowed_mean_destep_with_diagnostics(
        samples,
        min_offset=10.0,
        avg_window=1000,
        max_steps=5,
    )

    assert len(applied_steps) >= 1

    left_mean = float(np.mean(corrected[npts // 2 - 500:npts // 2]))
    right_mean = float(np.mean(corrected[npts // 2:npts // 2 + 500]))
    assert abs(right_mean - left_mean) < 1.0
    assert corrected.shape == samples.shape


def test_remove_step_one_step_handles_large_series(qtbot):
    data = DemoData()
    widget = RemoveStep(data, "Ex1")
    qtbot.addWidget(widget)

    npts = 200_000
    samples = np.linspace(0.0, 1.0, npts, dtype=np.float64)
    samples[npts // 3:] += 15.0
    samples[2 * npts // 3:] += 20.0

    corrected, removed_steps = remove_diff_steps_by_count(samples, 2)

    assert len(removed_steps) == 2
    left_mean = float(np.mean(corrected[npts // 3 - 500:npts // 3]))
    mid_mean = float(np.mean(corrected[npts // 3:npts // 3 + 500]))
    right_mean = float(np.mean(corrected[2 * npts // 3:2 * npts // 3 + 500]))
    assert abs(left_mean - mid_mean) < 1.0
    assert abs(mid_mean - right_mean) < 1.0


def test_remove_step_manual_zero_selected_diff_points(qtbot):
    data = MultiStepDemoData()
    widget = RemoveStep(data, "Ex1")
    qtbot.addWidget(widget)

    widget.selected_diff_indices = [2]
    widget._remove_selected_diff_indices()

    np.testing.assert_allclose(widget.y_data, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0]), atol=1e-8)
    assert "手动差分置零完成，共去除 1 个台阶" in widget.plain_text.toPlainText()


def test_remove_step_ctrl_z_undoes_last_diff_zeroing(qtbot):
    data = MultiStepDemoData()
    widget = RemoveStep(data, "Ex1")
    qtbot.addWidget(widget)

    original = widget.y_data.copy()
    widget.selected_diff_indices = [2]
    widget._remove_selected_diff_indices()
    widget.undo_shortcut.activated.emit()

    np.testing.assert_allclose(widget.y_data, original, atol=1e-8)
    assert "已撤销上一步操作" in widget.plain_text.toPlainText()


def test_robust_baseline_destep_reduces_continuous_step():
    npts = 800
    x = np.arange(npts, dtype=np.float64)
    baseline = 0.002 * x + 0.12 * np.sin(x / 30.0)
    left, right = 320, 360
    transition = np.zeros(npts, dtype=np.float64)
    ramp_x = np.arange(left, right + 1, dtype=np.float64)
    transition[left:right + 1] = 6.0 * (ramp_x - left) / float(right - left)
    transition[right + 1:] = 6.0
    samples = baseline + transition

    result = robust_baseline_destep(
        samples,
        (left, right),
        baseline_degree=2,
        transition_degree=8,
        context_points=120,
        loss="soft_l1",
    )

    before_left = float(np.mean(samples[left - 40:left]))
    before_right = float(np.mean(samples[right + 1:right + 41]))
    after_left = float(np.mean(result.corrected[left - 40:left]))
    after_right = float(np.mean(result.corrected[right + 1:right + 41]))

    assert result.step_height > 0.0
    assert abs(result.step_height - 6.0) < 2.0
    assert abs(after_right - after_left) < abs(before_right - before_left)
    assert result.success is True


def test_robust_baseline_destep_handles_downward_step_and_limits_baseline_to_fit_window():
    npts = 800
    x = np.arange(npts, dtype=np.float64)
    baseline = 0.001 * x + 0.08 * np.sin(x / 25.0)
    left, right = 300, 345
    transition = np.zeros(npts, dtype=np.float64)
    ramp_x = np.arange(left, right + 1, dtype=np.float64)
    transition[left:right + 1] = -5.5 * (ramp_x - left) / float(right - left)
    transition[right + 1:] = -5.5
    samples = baseline + transition

    result = robust_baseline_destep(
        samples,
        (left, right),
        baseline_degree=2,
        transition_degree=8,
        context_points=100,
        loss="soft_l1",
    )

    before_left = float(np.mean(samples[left - 30:left]))
    before_right = float(np.mean(samples[right + 1:right + 31]))
    after_left = float(np.mean(result.corrected[left - 30:left]))
    after_right = float(np.mean(result.corrected[right + 1:right + 31]))

    assert result.step_height < 0.0
    assert abs(after_right - after_left) < 1.0
    assert np.all(np.isnan(result.baseline[:result.fit_window[0]]))
    assert np.all(np.isnan(result.baseline[result.fit_window[1] + 1:]))


def test_remove_step_baseline_widget_uses_region_selection_and_higher_degree_limit(qtbot):
    data = StepDemoData()
    widget = RemoveStepByBaseline(data, "Ex1")
    qtbot.addWidget(widget)

    assert widget.base_degree_spin.maximum() == 1
    assert widget.transition_degree_spin.maximum() == 10
    assert not hasattr(widget, "plain_text")
    assert widget.log_messages == []
    assert widget.plot_splitter.opaqueResize() is False
    assert widget.plot_splitter.orientation() == Qt.Orientation.Vertical
    assert widget.plot_splitter.widget(0) is widget.signal_plot_widget
    assert widget.plot_splitter.widget(1) is widget.correction_plot_widget
    assert widget.region_item is None


def test_remove_step_baseline_widget_adds_single_region_on_main_plot(qtbot):
    data = StepDemoData()
    widget = RemoveStepByBaseline(data, "Ex1")
    qtbot.addWidget(widget)

    widget._add_region()
    first_region = widget.region_item
    widget._add_region()

    assert first_region is not None
    assert widget.region_item is first_region
    assert len(widget.selected_indices) > 0


def test_remove_step_baseline_right_click_handler_removes_region(qtbot):
    data = StepDemoData()
    widget = RemoveStepByBaseline(data, "Ex1")
    qtbot.addWidget(widget)

    widget._add_region()
    assert widget.region_item is not None
    left, right = widget.region_item.getRegion()

    removed = widget._handle_plot_right_click((left + right) * 0.5, 0.0)

    assert removed is True
    assert widget.region_item is None
    assert widget.selected_indices == []


def test_remove_step_baseline_region_uses_current_view_width_limit(qtbot):
    data = StepDemoData()
    widget = RemoveStepByBaseline(data, "Ex1")
    qtbot.addWidget(widget)

    widget.signal_plot_item.vb.setXRange(widget.x_data[1], widget.x_data[4], padding=0)
    view_min, view_max = widget.signal_plot_item.vb.viewRange()[0]
    visible_width = view_max - view_min
    widget._add_or_move_region_from_x((view_min + view_max) * 0.5)

    assert widget.region_item is not None
    left, right = widget.region_item.getRegion()
    assert left >= view_min
    assert right <= view_max
    assert (right - left) <= visible_width * 0.2 + 1e-9


def test_wavelet_local_depulse_reduces_pulse_tail():
    pytest.importorskip("pywt")

    data = PulseDemoData()
    samples = data.get_channel_series("Ex1").to_numpy(dtype=np.float64)
    region = (1805, 1985)
    result = wavelet_local_depulse(
        samples,
        region,
        wavelet_name="db4",
        level=6,
        threshold_scale=1.0,
        padding_points=128,
        mode="soft",
    )

    before_peak = float(np.max(samples[region[0]:region[1] + 1]))
    after_peak = float(np.max(result.corrected[region[0]:region[1] + 1]))
    tail_before = float(np.mean(samples[1940:1980]))
    tail_after = float(np.mean(result.corrected[1940:1980]))
    outside_before = samples[1500:1600]
    outside_after = result.corrected[1500:1600]

    assert result.corrected.shape == samples.shape
    assert result.changed_coefficients > 0
    assert after_peak < before_peak
    assert tail_after < tail_before
    assert np.max(np.abs(outside_after - outside_before)) < 0.5


def test_remove_pulse_wavelet_widget_uses_region_selection_and_two_plots(qtbot):
    data = PulseDemoData()
    widget = RemoveWaveletDenoise(data, "Ex1")
    qtbot.addWidget(widget)

    assert widget.level_spin.value() == 6
    assert not hasattr(widget, "plain_text")
    assert isinstance(widget.log_messages, list)
    assert widget.plot_splitter.opaqueResize() is False
    assert widget.plot_splitter.orientation() == Qt.Orientation.Vertical
    assert widget.plot_splitter.widget(0) is widget.signal_plot_widget
    assert widget.plot_splitter.widget(1) is widget.detail_plot_widget
    assert widget.region_item is None


def test_remove_pulse_wavelet_region_uses_current_view_width_limit(qtbot):
    data = PulseDemoData()
    widget = RemoveWaveletDenoise(data, "Ex1")
    qtbot.addWidget(widget)

    widget.signal_plot_item.vb.setXRange(widget.x_data[100], widget.x_data[300], padding=0)
    view_min, view_max = widget.signal_plot_item.vb.viewRange()[0]
    visible_width = view_max - view_min
    widget._add_or_move_region_from_x((view_min + view_max) * 0.5)

    assert widget.region_item is not None
    left, right = widget.region_item.getRegion()
    assert left >= view_min
    assert right <= view_max
    assert (right - left) <= visible_width * 0.2 + 1e-9


def test_remove_wavelet_denoise_right_click_handler_removes_region(qtbot):
    data = PulseDemoData()
    widget = RemoveWaveletDenoise(data, "Ex1")
    qtbot.addWidget(widget)

    widget._add_region()
    assert widget.region_item is not None
    left, right = widget.region_item.getRegion()

    removed = widget._handle_plot_right_click((left + right) * 0.5, 0.0)

    assert removed is True
    assert widget.region_item is None
    assert widget.selected_indices == []


def test_remove_step_baseline_logs_to_memory_buffer(qtbot):
    data = StepDemoData()
    widget = RemoveStepByBaseline(data, "Ex1")
    qtbot.addWidget(widget)

    widget._log("baseline message")

    assert widget.log_messages[-1] == "baseline message"


def test_remove_step_baseline_confirm_restores_nan_positions(qtbot):
    data = StepDemoDataWithNaN()
    widget = RemoveStepByBaseline(data, "Ex1")
    qtbot.addWidget(widget)

    received = []
    widget.result_signal.connect(lambda ch, result: received.append((ch, result.copy())))
    widget._confirm_and_close()

    assert received
    channel_name, result = received[0]
    assert channel_name == "Ex1"
    assert np.isnan(result[1])


def test_robust_baseline_destep_caps_baseline_degree_and_supports_high_transition_degree():
    npts = 900
    x = np.arange(npts, dtype=np.float64)
    baseline = 0.0015 * x + 0.05 * np.sin(x / 35.0)
    left, right = 360, 430
    transition = np.zeros(npts, dtype=np.float64)
    t = np.linspace(0.0, 1.0, right - left + 1, dtype=np.float64)
    curved = 4.5 * (1.0 - np.exp(-4.0 * t))
    transition[left:right + 1] = curved
    transition[right + 1:] = curved[-1]
    samples = baseline + transition

    result = robust_baseline_destep(
        samples,
        (left, right),
        baseline_degree=9,
        transition_degree=10,
        context_points=120,
        loss="soft_l1",
    )

    assert result.baseline_coeffs.size == 2
    assert result.transition_coeffs.size == 10
    assert result.success is True


def test_remove_wavelet_denoise_logs_to_memory_buffer(qtbot):
    data = PulseDemoData()
    widget = RemoveWaveletDenoise(data, "Ex1")
    qtbot.addWidget(widget)

    widget._log("wavelet message")

    assert widget.log_messages[-1] == "wavelet message"


def test_remove_wavelet_denoise_full_scope_allows_processing_without_region(qtbot):
    data = PulseDemoData()
    widget = RemoveWaveletDenoise(data, "Ex1")
    qtbot.addWidget(widget)

    widget.scope_combo.setCurrentText("全通道")

    region = widget._read_region()

    assert region == (0, widget.y_data.size - 1)
    assert widget.add_region_btn.isEnabled() is False
    assert "全通道" in widget.selection_status_label.text()


def test_remove_wavelet_denoise_level_limit_tracks_selected_window(qtbot):
    pytest.importorskip("pywt")

    data = PulseDemoData()
    widget = RemoveWaveletDenoise(data, "Ex1")
    qtbot.addWidget(widget)

    initial_max = widget.level_spin.maximum()
    widget.signal_plot_item.vb.setXRange(widget.x_data[1800], widget.x_data[1820], padding=0)
    widget._add_or_move_region_from_x(float(widget.x_data[1810]))
    reduced_max = widget.level_spin.maximum()

    assert reduced_max <= initial_max
    assert reduced_max >= 1


def test_remove_wavelet_denoise_preset_updates_controls(qtbot):
    data = PulseDemoData()
    widget = RemoveWaveletDenoise(data, "Ex1")
    qtbot.addWidget(widget)

    widget.preset_combo.setCurrentText("尖峰")

    assert widget.mode_combo.currentText() == "hard"
    assert widget.threshold_scale_spin.value() == 2.2
    assert widget.padding_spin.value() == 64


def test_remove_wavelet_denoise_preview_updates_energy_plot(qtbot):
    pytest.importorskip("pywt")

    data = PulseDemoData()
    widget = RemoveWaveletDenoise(data, "Ex1")
    qtbot.addWidget(widget)
    widget._add_region()

    widget._preview_fit()
    # 重计算在 compute 池后台执行，等结果回到主线程
    qtbot.waitUntil(lambda: widget.last_result is not None, timeout=10000)

    assert widget.last_result is not None
    assert widget.last_result.detail_energy_before.size >= 1
    qtbot.waitUntil(lambda: widget.local_left_line.isVisible(), timeout=5000)
    assert widget.local_left_line.isVisible()
    assert widget.local_right_line.isVisible()


def test_remove_wavelet_denoise_scale_gains_sets_all_levels(qtbot):
    data = PulseDemoData()
    widget = RemoveWaveletDenoise(data, "Ex1")
    qtbot.addWidget(widget)

    widget.mode_combo.setCurrentText("manual")
    widget.global_gain_spin.setValue(0.75)
    widget._scale_gains()

    assert widget.gain_spins
    assert all(abs(spin.value() - 0.75) < 1e-9 for spin in widget.gain_spins)


def test_remove_wavelet_denoise_preset_dict_roundtrip(qtbot):
    data = PulseDemoData()
    widget = RemoveWaveletDenoise(data, "Ex1")
    qtbot.addWidget(widget)

    widget.scope_combo.setCurrentText("全通道")
    widget.mode_combo.setCurrentText("manual")
    widget.wavelet_combo.setCurrentText("sym6")
    widget.level_spin.setValue(min(4, widget.level_spin.maximum()))
    widget.threshold_scale_spin.setValue(1.7)
    widget.padding_spin.setValue(320)
    widget.global_gain_spin.setValue(0.85)
    for i, spin in enumerate(widget.gain_spins):
        spin.setValue(0.5 + 0.1 * i)

    preset = widget._state_to_preset_dict()

    other = RemoveWaveletDenoise(data, "Ex1")
    qtbot.addWidget(other)
    other._apply_preset_dict(preset)

    assert other.scope_combo.currentText() == "全通道"
    assert other.mode_combo.currentText() == "manual"
    assert other.wavelet_combo.currentText() == "sym6"
    assert other.threshold_scale_spin.value() == 1.7
    assert other.padding_spin.value() == 320
    assert other.global_gain_spin.value() == 0.85
    assert len(other.gain_spins) == len(widget.gain_spins)
    for left, right in zip(widget.gain_spins, other.gain_spins):
        assert abs(left.value() - right.value()) < 1e-9


def test_remove_wavelet_denoise_preset_toml_roundtrip(qtbot):
    data = PulseDemoData()
    widget = RemoveWaveletDenoise(data, "Ex1")
    qtbot.addWidget(widget)

    widget.transform_combo.setCurrentText("Wavelet Packet")
    widget.scope_combo.setCurrentText("全通道")
    widget.mode_combo.setCurrentText("manual")
    widget.level_spin.setValue(min(3, widget.level_spin.maximum()))
    widget.wp_low_gain_spin.setValue(0.6)
    widget.wp_mid_gain_spin.setValue(1.0)
    widget.wp_high_gain_spin.setValue(1.4)
    for i, spin in enumerate(widget.gain_spins):
        spin.setValue(0.8 + 0.05 * i)

    preset = widget._state_to_preset_dict()
    toml_text = widget._preset_dict_to_toml(preset)
    loaded = tomllib.loads(toml_text)

    assert loaded["scope"] == "全通道"
    assert loaded["mode"] == "manual"
    assert loaded["level"] == widget.level_spin.value()
    assert loaded["transform_kind"] == "wp"
    assert loaded["wp_group_gains"]["low"] == 0.6
    assert len(loaded["detail_gains"]) == len(widget.gain_spins)


def test_wavelet_multiscale_manual_gain_changes_local_signal():
    pytest.importorskip("pywt")

    data = PulseDemoData()
    samples = data.get_channel_series("Ex1").to_numpy(dtype=np.float64)
    region = (1805, 1985)
    result = wavelet_multiscale_denoise(
        samples,
        region,
        wavelet_name="db4",
        level=4,
        threshold_scale=1.0,
        padding_points=128,
        mode="manual",
        detail_gains=[0.2, 0.5, 1.0, 1.0],
    )

    assert result.mode == "manual"
    assert result.corrected.shape == samples.shape
    assert result.changed_coefficients > 0
    assert np.any(np.abs(result.corrected[region[0]:region[1] + 1] - samples[region[0]:region[1] + 1]) > 1e-6)


def test_wavelet_packet_manual_gain_changes_local_signal():
    pytest.importorskip("pywt")

    data = PulseDemoData()
    samples = data.get_channel_series("Ex1").to_numpy(dtype=np.float64)
    region = (1805, 1985)
    result = wavelet_multiscale_denoise(
        samples,
        region,
        transform_kind="wp",
        wavelet_name="db4",
        level=3,
        sample_rate_hz=1.0,
        threshold_scale=1.0,
        padding_points=128,
        mode="manual",
        detail_gains=[0.5] * 8,
    )

    assert result.transform_kind == "wp"
    assert len(result.band_labels) == 8
    assert "Hz" in result.band_labels[0]
    assert result.corrected.shape == samples.shape
    assert result.changed_coefficients > 0


def test_wavelet_dwt_frequency_labels_present():
    pytest.importorskip("pywt")

    data = PulseDemoData()
    samples = data.get_channel_series("Ex1").to_numpy(dtype=np.float64)
    region = (1805, 1985)
    result = wavelet_multiscale_denoise(
        samples,
        region,
        transform_kind="dwt",
        wavelet_name="db4",
        level=3,
        sample_rate_hz=1.0,
        threshold_scale=1.0,
        padding_points=128,
        mode="soft",
    )

    assert result.transform_kind == "dwt"
    assert len(result.band_labels) == 3
    assert all("Hz" in label for label in result.band_labels)


def test_wavelet_denoise_local_window_uses_tighter_padding():
    pytest.importorskip("pywt")

    data = PulseDemoData()
    samples = data.get_channel_series("Ex1").to_numpy(dtype=np.float64)
    region = (1805, 1985)
    result = wavelet_multiscale_denoise(
        samples,
        region,
        transform_kind="dwt",
        wavelet_name="db4",
        level=3,
        sample_rate_hz=1.0,
        threshold_scale=1.0,
        padding_points=64,
        mode="soft",
    )

    local_start, local_stop = result.local_window
    region_width = region[1] - region[0] + 1
    assert (local_stop - local_start + 1) <= region_width + 2 * max(64, int(np.ceil(region_width * 0.5)))


def test_remove_wavelet_denoise_wavelet_packet_updates_band_count(qtbot):
    data = PulseDemoData()
    widget = RemoveWaveletDenoise(data, "Ex1")
    qtbot.addWidget(widget)

    widget.transform_combo.setCurrentText("Wavelet Packet")
    widget.level_spin.setValue(3)

    assert len(widget.gain_spins) == 8


def test_remove_wavelet_denoise_gain_change_switches_to_manual(qtbot):
    data = PulseDemoData()
    widget = RemoveWaveletDenoise(data, "Ex1")
    qtbot.addWidget(widget)

    assert widget.mode_combo.currentText() != "manual"
    widget.gain_spins[0].setValue(0.8)

    assert widget.mode_combo.currentText() == "manual"


def test_remove_wavelet_denoise_wavelet_packet_group_gains_apply(qtbot):
    data = PulseDemoData()
    widget = RemoveWaveletDenoise(data, "Ex1")
    qtbot.addWidget(widget)

    widget.transform_combo.setCurrentText("Wavelet Packet")
    widget.mode_combo.setCurrentText("manual")
    widget.level_spin.setValue(3)
    widget.wp_low_gain_spin.setValue(0.5)
    widget.wp_mid_gain_spin.setValue(1.0)
    widget.wp_high_gain_spin.setValue(1.5)

    widget._apply_wp_groups()

    values = [spin.value() for spin in widget.gain_spins]
    assert len(values) == 8
    assert values[0] == 0.5
    assert values[-1] == 1.5
    assert "Hz" in widget.wp_low_label.text()
    assert "Hz" in widget.wp_mid_label.text()
    assert "Hz" in widget.wp_high_label.text()




def test_robust_baseline_destep_handles_large_series():
    npts = 500_000
    x = np.arange(npts, dtype=np.float64)
    baseline = 4.0 + 0.00002 * x + 0.15 * np.sin(x / 4000.0)
    left, right = 210_000, 210_120
    transition = np.zeros(npts, dtype=np.float64)
    ramp_x = np.arange(left, right + 1, dtype=np.float64)
    transition[left:right + 1] = -7.0 * (ramp_x - left) / float(right - left)
    transition[right + 1:] = -7.0
    samples = baseline + transition

    result = robust_baseline_destep(
        samples,
        (left, right),
        baseline_degree=1,
        transition_degree=1,
        context_points=100,
        loss="soft_l1",
    )

    before_left = float(np.mean(samples[left - 50:left]))
    before_right = float(np.mean(samples[right + 1:right + 51]))
    after_left = float(np.mean(result.corrected[left - 50:left]))
    after_right = float(np.mean(result.corrected[right + 1:right + 51]))

    assert result.corrected.shape == samples.shape
    assert result.step_height < 0.0
    assert abs(after_right - after_left) < 1.0
