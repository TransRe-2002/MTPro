import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication
from unittest.mock import patch

from core.em_data import Channel, EMData
from ui.data_process import DataProcessLegacyWidget, DataProcessWidget, ToolWindowState
from ui.data_process_pipeline import PipelineBlock


class DemoData(EMData):
    def __init__(self, name: str, channels: list[str]):
        super().__init__("/tmp/demo.mat")
        self.name = name
        self.npts = 3
        self.start_time = pd.Timestamp("2024-01-01 00:00:00")
        self.end_time = pd.Timestamp("2024-01-01 00:00:10")
        self.dt = pd.Timedelta(seconds=5)
        self.datetime_index = pd.date_range(self.start_time, periods=3, freq="5s")
        self.chid = list(channels)
        self.kp_data = None
        for ch_name in channels:
            series = pd.Series([1.0, 2.0, 3.0], index=self.datetime_index)
            self.data[ch_name] = Channel(ch_name, series, self)

    def restore_data(self, ch: str):
        return None


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_data_process_widget_preserves_selected_channel_and_windows():
    _app()
    widget = DataProcessLegacyWidget()
    data = DemoData("demo", ["Ex1", "Hy1"])

    widget.init_data(data, 1)
    widget.data_group["Hy1"].setChecked(True)

    first = widget.open_tool_window("remove step", "Hy1")
    second = widget.open_tool_window("remove step", "Hy1")

    assert first is second
    assert widget.current_channel().name == "Hy1"

    # Rebinding the same dataset should keep the selected channel and open tool window.
    widget.init_data(data, 1)

    assert widget.current_channel().name == "Hy1"
    assert widget.open_tool_window_states() == [
        ToolWindowState("remove step", "Hy1")
    ]

    widget.close()


def test_data_process_widget_closes_windows_when_data_cleared():
    _app()
    widget = DataProcessLegacyWidget()
    data = DemoData("demo", ["Ex1"])

    widget.init_data(data, 1)
    widget.open_tool_window("remove spike", "Ex1")
    assert len(widget.open_tool_window_states()) == 1

    widget.init_data(None, 0)

    assert widget.current_channel() is None
    assert len(widget.open_tool_window_states()) == 0

    widget.close()


def test_data_process_widget_reopens_remove_step_after_subwindow_close():
    app = _app()
    widget = DataProcessLegacyWidget()
    data = DemoData("demo", ["Ex1"])

    widget.show()
    widget.init_data(data, 1)

    first = widget.open_tool_window("remove step", "Ex1")
    assert first is not None
    first.show()
    app.processEvents()
    assert first.widget() is not None

    first.close()
    app.processEvents()
    app.sendPostedEvents(None, 0)
    app.processEvents()
    assert len(widget.open_tool_window_states()) == 0

    second = widget.open_tool_window("remove step", "Ex1")
    assert second is not None
    assert second is not first
    second.show()
    app.processEvents()
    assert second.widget() is not None
    assert second.widget().isVisible()

    widget.close()


def test_data_process_host_switches_modes_and_preserves_pipeline_state():
    _app()
    widget = DataProcessWidget()
    data = DemoData("demo", ["Ex1", "Hy1"])

    widget.init_data(data, 1)
    widget.set_mode("pipeline")

    pipeline = widget.pipeline_widget
    pipeline.selected_tool_id = "remove_step_diff"
    pipeline._on_add_block_requested("Ex1", pipeline.pipelines[0].layers[0].layer_id, 100)

    assert widget.current_mode() == "pipeline"
    assert pipeline.block_count() == 1

    widget.set_mode("classic")
    widget.set_mode("pipeline")

    assert widget.current_mode() == "pipeline"
    assert pipeline.block_count() == 1

    widget.close()


def test_pipeline_tool_tree_selection_and_clear():
    _app()
    widget = DataProcessWidget()
    data = DemoData("demo", ["Ex1"])

    widget.init_data(data, 1)
    widget.set_mode("pipeline")

    pipeline = widget.pipeline_widget
    pipeline._set_selected_tool("remove_step_diff")

    assert pipeline.selected_tool_id == "remove_step_diff"
    assert pipeline.tool_tree is not None
    assert pipeline.tool_tree.currentIndex().isValid()
    assert "remove step diff" in pipeline.info_text.toPlainText()

    pipeline._clear_selected_tool()

    assert pipeline.selected_tool_id is None
    assert not pipeline.tool_tree.currentIndex().isValid()

    widget.close()


def test_pipeline_drop_adds_block_and_updates_selected_tool():
    _app()
    widget = DataProcessWidget()
    data = DemoData("demo", ["Ex1"])

    widget.init_data(data, 1)
    widget.set_mode("pipeline")

    pipeline = widget.pipeline_widget
    layer = pipeline.pipelines[0].layers[0]

    pipeline._on_tool_dropped("Ex1", layer.layer_id, "remove_step_diff", 100)

    assert pipeline.selected_tool_id == "remove_step_diff"
    assert pipeline.block_count() == 1
    assert layer.blocks[0].tool_id == "remove_step_diff"

    widget.close()


def test_pipeline_preview_commit_rebases_and_marks_downstream_stale():
    _app()
    widget = DataProcessWidget()
    data = DemoData("demo", ["Ex1"])

    widget.init_data(data, 1)
    widget.set_mode("pipeline")

    pipeline_widget = widget.pipeline_widget
    channel_pipeline = pipeline_widget.pipelines[0]
    layer = channel_pipeline.layers[0]
    base = data.data["Ex1"].cts.to_numpy(dtype=np.float64, copy=True)

    first = PipelineBlock(
        tool_id="remove_spike",
        start=0,
        end=2,
        status="ready",
        input_snapshot=base.copy(),
        output_snapshot=base + 1,
    )
    second = PipelineBlock(
        tool_id="remove_step_diff",
        start=2,
        end=3,
        status="ready",
        input_snapshot=base + 1,
        output_snapshot=base + 3,
    )
    layer.blocks.extend([first, second])

    preview = pipeline_widget._collect_preview_results(update_status=True)
    np.testing.assert_allclose(preview["Ex1"], base + 3)
    assert first.status == "ready"
    assert second.status == "ready"

    first.output_snapshot = base + 10
    preview = pipeline_widget._collect_preview_results(update_status=True)
    np.testing.assert_allclose(preview["Ex1"], base + 10)
    assert first.status == "ready"
    assert second.status == "stale"

    changed = []
    pipeline_widget.change_finished_signal.connect(
        lambda data_id, channel: changed.append((data_id, channel))
    )
    with patch("ui.data_process_pipeline.QMessageBox.information"):
        pipeline_widget._commit_preview()

    np.testing.assert_allclose(
        data.data["Ex1"].cts.to_numpy(dtype=np.float64, copy=True),
        base + 10,
    )
    np.testing.assert_allclose(channel_pipeline.baseline_snapshot, base + 10)
    assert first.output_snapshot is None
    assert second.output_snapshot is None
    assert changed == [(1, "Ex1")]

    widget.close()
