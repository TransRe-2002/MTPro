import numpy as np
import pandas as pd
from PySide6.QtCore import Qt

from core.em_data import Channel, EMData
from processor.remove_step import RemoveStep, RemoveStepPlotWidget
from processor.remove_step_by_window import RemoveStepByWindow


class DemoData(EMData):
    def __init__(self):
        super().__init__("/tmp/demo.mat")
        self.name = "demo"
        self.npts = 5
        self.start_time = pd.Timestamp("2024-01-01 00:00:00")
        self.end_time = pd.Timestamp("2024-01-01 00:00:20")
        self.dt = pd.Timedelta(seconds=5)
        self.datetime_index = pd.date_range(self.start_time, periods=5, freq="5s")
        self.chid = ["Ex1"]
        self.kp_data = None
        series = pd.Series([1.0, 2.0, 3.0, 5.0, 8.0], index=self.datetime_index)
        self.data["Ex1"] = Channel("Ex1", series, self)

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
        self.chid = ["Ex1"]
        self.kp_data = None
        series = pd.Series([0.0, 0.0, 0.0, 10.0, 10.0, 10.0], index=self.datetime_index)
        self.data["Ex1"] = Channel("Ex1", series, self)

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
        self.chid = ["Ex1"]
        self.kp_data = None
        series = pd.Series([0.0, 0.0, 0.0, 10.0, 10.0, 20.0, 20.0], index=self.datetime_index)
        self.data["Ex1"] = Channel("Ex1", series, self)

    def restore_data(self, ch: str):
        return None


def test_remove_step_plot_widget_uses_strong_focus_and_axis_constraints(qtbot):
    plot = RemoveStepPlotWidget()
    qtbot.addWidget(plot)

    assert plot.focusPolicy() == Qt.FocusPolicy.StrongFocus

    view_box = plot.getViewBox()
    plot._apply_modifier_axis_constraint(Qt.KeyboardModifier.ShiftModifier)
    assert tuple(view_box.state["mouseEnabled"]) == (True, False)

    plot._apply_modifier_axis_constraint(Qt.KeyboardModifier.AltModifier)
    assert tuple(view_box.state["mouseEnabled"]) == (False, True)

    plot._apply_modifier_axis_constraint(Qt.KeyboardModifier.NoModifier)
    assert tuple(view_box.state["mouseEnabled"]) == (True, True)


def test_remove_step_sets_focus_to_plot_on_show(qtbot):
    data = DemoData()
    widget = RemoveStep(data.data["Ex1"])
    qtbot.addWidget(widget)

    widget.show()
    qtbot.wait(50)

    assert isinstance(widget.plot_widget, RemoveStepPlotWidget)
    assert widget.signal_plot_widget.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert widget.one_step_times.text() == "5"
    assert widget.plain_text.maximumWidth() == 320
    assert widget.threshold_frame is not None
    assert widget.multi_step_frame is not None
    assert widget.manual_frame is not None
    assert widget.action_frame is not None


def test_remove_step_window_threshold_uses_windowed_destep(qtbot):
    data = StepDemoData()
    widget = RemoveStepByWindow(data.data["Ex1"])
    qtbot.addWidget(widget)

    widget.de_step_threshold.setText("5")
    widget.avg_window_edit.setText("2")
    widget._de_step_by_threshold()

    np.testing.assert_allclose(widget.y_data, np.zeros(6), atol=1e-8)
    np.testing.assert_allclose(widget.diff_y_data, np.zeros(5), atol=1e-8)
    assert "去除了 1 个台阶" in widget.plain_text.toPlainText()


def test_remove_step_window_threshold_uses_default_max_steps_in_log(qtbot):
    data = MultiStepDemoData()
    widget = RemoveStepByWindow(data.data["Ex1"])
    qtbot.addWidget(widget)

    widget.de_step_threshold.setText("5")
    widget.avg_window_edit.setText("2")
    widget._de_step_by_threshold()

    assert f"最多台阶={widget.DEFAULT_MAX_STEPS}" in widget.plain_text.toPlainText()


def test_remove_step_one_step_matches_matlab_style_algorithm(qtbot):
    data = MultiStepDemoData()
    widget = RemoveStep(data.data["Ex1"])
    qtbot.addWidget(widget)

    corrected_one, removed_one = widget._matlab_style_one_step(widget.y_data, 1)
    corrected_two, removed_two = widget._matlab_style_one_step(widget.y_data, 2)

    np.testing.assert_allclose(corrected_one, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0]), atol=1e-8)
    np.testing.assert_allclose(corrected_two, np.zeros(7), atol=1e-8)
    assert len(removed_one) == 1
    assert len(removed_two) == 2


def test_remove_step_multi_button_uses_matlab_style_one_step(qtbot):
    data = MultiStepDemoData()
    widget = RemoveStep(data.data["Ex1"])
    qtbot.addWidget(widget)

    widget.one_step_times.setText("1")
    widget._remove_by_count()

    np.testing.assert_allclose(widget.y_data, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0]), atol=1e-8)
    assert "共去除 1 个台阶" in widget.plain_text.toPlainText()


def test_remove_step_window_destep_handles_large_series(qtbot):
    data = DemoData()
    widget = RemoveStepByWindow(data.data["Ex1"])
    qtbot.addWidget(widget)

    npts = 200_000
    samples = np.linspace(0.0, 1.0, npts, dtype=np.float64)
    samples += 0.05 * np.sin(np.linspace(0.0, 400.0 * np.pi, npts, dtype=np.float64))
    samples[npts // 2:] += 25.0

    corrected, applied_steps = widget._windowed_mean_destep(
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
    widget = RemoveStep(data.data["Ex1"])
    qtbot.addWidget(widget)

    npts = 200_000
    samples = np.linspace(0.0, 1.0, npts, dtype=np.float64)
    samples[npts // 3:] += 15.0
    samples[2 * npts // 3:] += 20.0

    corrected, removed_steps = widget._matlab_style_one_step(samples, 2)

    assert len(removed_steps) == 2
    left_mean = float(np.mean(corrected[npts // 3 - 500:npts // 3]))
    mid_mean = float(np.mean(corrected[npts // 3:npts // 3 + 500]))
    right_mean = float(np.mean(corrected[2 * npts // 3:2 * npts // 3 + 500]))
    assert abs(left_mean - mid_mean) < 1.0
    assert abs(mid_mean - right_mean) < 1.0


def test_remove_step_manual_zero_selected_diff_points(qtbot):
    data = MultiStepDemoData()
    widget = RemoveStep(data.data["Ex1"])
    qtbot.addWidget(widget)

    widget.selected_diff_indices = [2]
    widget._remove_selected_diff_indices()

    np.testing.assert_allclose(widget.y_data, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0]), atol=1e-8)
    assert "手动差分置零完成，共去除 1 个台阶" in widget.plain_text.toPlainText()
