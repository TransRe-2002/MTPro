import json

import numpy as np
import pandas as pd
import xarray as xr
from PySide6.QtCore import QByteArray, QMimeData, QPoint, Qt
from PySide6.QtWidgets import QApplication, QLabel
from unittest.mock import patch

from base.data_manager import DataManager
from core.em_data import EMData
from ui.data_process import DataProcessWidget, WorkspaceMdiArea
from ui.data_process_shared import DATA_TREE_SELECTION_MIME
from ui.data_process_pipeline import DataProcessPipelineWidget, PipelineBlock, TOOL_META
from ui.data_tree_view import DataTreeViewer


class DemoData(EMData):
    def __init__(
        self,
        name: str,
        channels: list[str],
        start: str = "2024-01-01 00:00:00",
        values: list[float] | None = None,
    ):
        super().__init__("/tmp/demo.mat")
        self.name = name
        samples = values or [1.0, 2.0, 3.0]
        self.npts = len(samples)
        self.start_time = pd.Timestamp(start)
        self.dt = pd.Timedelta(seconds=5)
        self.datetime_index = pd.date_range(self.start_time, periods=self.npts, freq="5s")
        self.end_time = self.datetime_index[-1]
        self.kp_data = None
        self.data = xr.Dataset(coords={"time": self.datetime_index})
        for ch_name in channels:
            self.data[ch_name] = xr.DataArray(
                np.asarray(samples, dtype=np.float64),
                dims=("time",),
                coords={"time": self.datetime_index},
            )

    def restore_data(self, ch: str):
        return None


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_data_process_host_creates_an_all_channel_pipeline_for_active_data():
    _app()
    widget = DataProcessWidget()
    data = DemoData("demo", ["Ex1", "Hy1"])

    widget.init_data(data, 1)
    assert len(widget.pipeline_panel._workspaces) == 1
    assert widget.pipeline_panel.workspace_tabs.count() == 2

    pipeline = widget.pipeline_widget
    assert pipeline is not None
    assert pipeline.channel_refs() == [(1, "Ex1"), (1, "Hy1")]

    widget.close()


def test_data_process_host_adds_selected_tool_to_pipeline_channel():
    _app()
    manager = DataManager()
    data = DemoData("demo", ["Ex1", "Hy1"])
    key = manager.add(data)
    widget = DataProcessWidget(manager)

    pipeline = widget.open_tool_for(key, "Hy1", "remove step")

    assert pipeline is not None
    channel_pipeline = pipeline._find_pipeline(key, "Hy1")
    assert channel_pipeline is not None
    assert channel_pipeline.layers[0].blocks[0].tool_id == "remove_step_diff"

    widget.close()


def test_data_process_has_one_pipeline_workspace_tab_set():
    _app()
    widget = DataProcessWidget()

    assert len(widget.pipeline_panel._workspaces) == 1
    assert widget.pipeline_panel.workspace_tabs.tabText(widget.pipeline_panel.workspace_tabs.count() - 1) == "+"
    assert widget.current_workspace() is widget.pipeline_panel.current_workspace()
    assert not hasattr(widget, "mode_stack")

    widget.close()


def test_data_process_creates_new_pipeline_workspace_tabs():
    _app()
    data = DemoData("demo", ["Ex1"])
    widget = DataProcessWidget()
    widget.init_data(data, 1)

    assert len(widget.pipeline_panel._workspaces) == 1
    assert widget.pipeline_panel.current_workspace() is not None

    first_workspace = widget.pipeline_panel.current_workspace()
    second_workspace = widget.pipeline_panel._create_workspace()

    assert len(widget.pipeline_panel._workspaces) == 2
    assert widget.pipeline_panel.workspace_tabs.count() == 3
    assert second_workspace is not None
    assert second_workspace is not first_workspace
    assert second_workspace.last_pipeline_window() is not None
    assert second_workspace.last_pipeline_window().channel_refs() == [(1, "Ex1")]

    widget.close()


def test_workspace_tab_close_keeps_single_workspace_and_hides_add_tab_close_button():
    _app()
    widget = DataProcessWidget()

    panel = widget.pipeline_panel
    panel._create_workspace()

    assert len(panel._workspaces) == 2
    assert panel.workspace_tabs.count() == 3

    panel._close_workspace(0)

    assert len(panel._workspaces) == 1
    assert panel.workspace_tabs.count() == 2
    assert panel.workspace_tabs.tabText(panel.workspace_tabs.count() - 1) == "+"

    tab_bar = panel.workspace_tabs.tabBar()
    add_index = panel.workspace_tabs.count() - 1
    assert tab_bar.tabButton(add_index, tab_bar.ButtonPosition.RightSide) is None or not tab_bar.tabButton(
        add_index, tab_bar.ButtonPosition.RightSide
    ).isVisible()

    panel._close_workspace(0)
    assert len(panel._workspaces) == 1
    assert panel.workspace_tabs.count() == 2

    widget.close()


def test_data_tree_drag_payload_uses_checked_channels_and_dataset_drag(qtbot):
    _app()
    manager = DataManager()
    key = manager.add(DemoData("demo", ["Ex1", "Hy1"]))
    viewer = DataTreeViewer(manager)
    qtbot.addWidget(viewer)

    dataset_index = viewer.model.index_for_key(key)
    ex_index = viewer.model.index_for_channel(key, "Ex1")
    hy_index = viewer.model.index_for_channel(key, "Hy1")

    viewer.tree_view.setCurrentIndex(ex_index)
    viewer.model.setData(ex_index, Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)
    viewer.model.setData(hy_index, Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)

    assert viewer.current_drag_payload() == {
        "kind": "channels",
        "refs": [
            {"data_id": key, "channel_name": "Ex1"},
            {"data_id": key, "channel_name": "Hy1"},
        ],
    }

    viewer.tree_view.setCurrentIndex(dataset_index)
    assert viewer.current_drag_payload() == {
        "kind": "dataset",
        "data_id": key,
    }


def test_workspace_creates_pipeline_window_and_clears_added_checked_channels(qtbot):
    _app()
    manager = DataManager()
    key = manager.add(DemoData("demo", ["Ex1", "Hy1"]))
    workspace = WorkspaceMdiArea(manager)
    workspace.init_data(manager.get(key), key)
    qtbot.addWidget(workspace)

    cleared = []
    workspace.set_clear_checked_channels(lambda refs: cleared.extend(refs))

    pipeline = workspace.create_pipeline_window([(key, "Ex1"), (key, "Hy1")])

    assert pipeline is not None
    assert pipeline.channel_refs() == [(key, "Ex1"), (key, "Hy1")]
    assert workspace.last_pipeline_window() is pipeline

    added, duplicated, failed = workspace.add_channels_to_pipeline_window(
        pipeline,
        [(key, "Ex1")],
    )

    assert added == []
    assert duplicated == [(key, "Ex1")]
    assert failed == []
    assert cleared == []
    workspace.close()


class DropStub:
    def __init__(self, payload: dict):
        self._mime = QMimeData()
        self._mime.setData(
            DATA_TREE_SELECTION_MIME,
            QByteArray(json.dumps(payload, ensure_ascii=True).encode("utf-8")),
        )
        self.accepted = False
        self.ignored = False

    def mimeData(self):
        return self._mime

    def position(self):
        return QPoint(20, 20)

    def acceptProposedAction(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


def test_workspace_drop_uses_mime_payload_and_clears_added_refs(qtbot):
    _app()
    manager = DataManager()
    first_key = manager.add(DemoData("first", ["Ex1"]))
    second_key = manager.add(DemoData("second", ["Hy1"]))
    workspace = WorkspaceMdiArea(manager)
    workspace.init_data(manager.get(first_key), first_key)
    workspace.set_selection_payload_resolver(lambda: {"kind": "channels", "refs": [{"data_id": first_key, "channel_name": "Ex1"}]})
    qtbot.addWidget(workspace)

    cleared = []
    workspace.set_clear_checked_channels(lambda refs: cleared.extend(refs))
    created_payloads = []

    class PipelineStub:
        def __init__(self, refs):
            self._refs = list(refs)

        def channel_refs(self):
            return list(self._refs)

    def fake_create_pipeline_window(channel_refs, position=None):
        del position
        created_payloads.append(list(channel_refs))
        return PipelineStub(channel_refs)

    workspace.create_pipeline_window = fake_create_pipeline_window
    event = DropStub(
        {
            "kind": "channels",
            "refs": [{"data_id": second_key, "channel_name": "Hy1"}],
        }
    )

    workspace._handle_data_selection_drop(event)

    assert event.accepted is True
    assert event.ignored is False
    assert created_payloads == [[(second_key, "Hy1")]]
    assert cleared == [(second_key, "Hy1")]


def test_pipeline_add_channels_rejects_non_overlapping_time_range(qtbot):
    _app()
    manager = DataManager()
    first_key = manager.add(DemoData("first", ["Ex1"], start="2024-01-01 00:00:00"))
    second_key = manager.add(DemoData("second", ["Ex1"], start="2024-01-02 00:00:00"))
    pipeline = DataProcessPipelineWidget(manager)
    pipeline.data_id = first_key
    pipeline.em_data = manager.get(first_key)
    qtbot.addWidget(pipeline)
    added, duplicated, failed = pipeline.add_channels([(first_key, "Ex1")])
    assert added == [(first_key, "Ex1")]

    added, duplicated, failed = pipeline.add_channels([(second_key, "Ex1")])

    assert added == []
    assert duplicated == []
    assert failed == [(second_key, "Ex1")]
    assert pipeline.channel_refs() == [(first_key, "Ex1")]
    pipeline.close()


def test_pipeline_common_time_window_preview_and_commit(qtbot):
    _app()
    manager = DataManager()
    first = DemoData("first", ["Ex1"], start="2024-01-01 00:00:00", values=[1.0, 2.0, 3.0, 4.0])
    second = DemoData("second", ["Hy1"], start="2024-01-01 00:00:05", values=[10.0, 20.0, 30.0])
    first_key = manager.add(first)
    second_key = manager.add(second)
    pipeline = DataProcessPipelineWidget(manager)
    pipeline.data_id = first_key
    pipeline.em_data = first
    qtbot.addWidget(pipeline)
    added, duplicated, failed = pipeline.add_channels([(first_key, "Ex1"), (second_key, "Hy1")])
    assert added == [(first_key, "Ex1"), (second_key, "Hy1")]
    assert duplicated == []
    assert failed == []

    assert pipeline._common_time_range() == (
        pd.Timestamp("2024-01-01 00:00:05"),
        pd.Timestamp("2024-01-01 00:00:15"),
    )
    np.testing.assert_allclose(pipeline._snapshot_for_channel(first_key, "Ex1"), [2.0, 3.0, 4.0])
    np.testing.assert_allclose(pipeline._snapshot_for_channel(second_key, "Hy1"), [10.0, 20.0, 30.0])

    first_pipeline = pipeline.pipelines[0]
    first_pipeline.layers[0].blocks.append(
        PipelineBlock(
            tool_id="remove_spike",
            start=0,
            end=3,
            status="ready",
            input_slice=np.array([2.0, 3.0, 4.0]),
            output_slice=np.array([12.0, 13.0, 14.0]),
        )
    )

    changed = []
    pipeline.change_finished_signal.connect(lambda data_id, channel: changed.append((data_id, channel)))
    with patch("ui.data_process_pipeline.QMessageBox.information"):
        pipeline._commit_preview()

    np.testing.assert_allclose(first.get_channel_series("Ex1").to_numpy(dtype=np.float64), [1.0, 12.0, 13.0, 14.0])
    np.testing.assert_allclose(second.get_channel_series("Hy1").to_numpy(dtype=np.float64), [10.0, 20.0, 30.0])
    assert changed == [(first_key, "Ex1")]
    pipeline.close()


def test_pipeline_normalizes_range_index_series_to_datetime_index(qtbot):
    _app()
    manager = DataManager()
    data = DemoData("demo", ["Ex1"])
    data.set_channel_series("Ex1", pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(0, 3)))
    key = manager.add(data)

    pipeline = DataProcessPipelineWidget(manager)
    pipeline.data_id = key
    pipeline.em_data = data
    qtbot.addWidget(pipeline)

    added, duplicated, failed = pipeline.add_channels([(key, "Ex1")])

    assert added == [(key, "Ex1")]
    assert duplicated == []
    assert failed == []
    assert isinstance(data.get_channel_series("Ex1").index, pd.DatetimeIndex)
    assert data.get_channel_series("Ex1").index.equals(data.datetime_index)
    np.testing.assert_allclose(pipeline._snapshot_for_channel(key, "Ex1"), [1.0, 2.0, 3.0])

    pipeline.close()


def test_pipeline_uses_shared_tool_drag_instead_of_a_second_tool_tree():
    _app()
    widget = DataProcessWidget()
    data = DemoData("demo", ["Ex1"])

    widget.init_data(data, 1)

    workspace = widget.pipeline_panel.current_workspace()
    assert workspace is not None
    pipeline = workspace.last_pipeline_window()
    assert pipeline is not None
    assert not hasattr(pipeline, "tool_tree")
    assert pipeline.pipeline_hint is not None
    assert "数据处理工具" in pipeline.pipeline_hint.text()

    widget.close()


def test_pipeline_drop_adds_block_from_shared_tool_payload():
    _app()
    widget = DataProcessWidget()
    data = DemoData("demo", ["Ex1"])

    widget.init_data(data, 1)

    workspace = widget.pipeline_panel.current_workspace()
    assert workspace is not None
    pipeline = workspace.last_pipeline_window()
    assert pipeline is not None
    layer = pipeline.pipelines[0].layers[0]

    pipeline._on_tool_dropped(1, "Ex1", layer.layer_id, "remove_step_diff", 100)

    assert pipeline.block_count() == 1
    assert layer.blocks[0].tool_id == "remove_step_diff"

    widget.close()


def test_pipeline_preview_commit_replays_downstream_edit_on_updated_upstream():
    _app()
    widget = DataProcessWidget()
    data = DemoData("demo", ["Ex1"])

    widget.init_data(data, 1)

    workspace = widget.pipeline_panel.current_workspace()
    assert workspace is not None
    pipeline_widget = workspace.last_pipeline_window()
    assert pipeline_widget is not None
    channel_pipeline = pipeline_widget.pipelines[0]
    layer = channel_pipeline.layers[0]
    base = data.get_channel_series("Ex1").to_numpy(dtype=np.float64, copy=True)

    first = PipelineBlock(
        tool_id="remove_spike",
        start=0,
        end=3,
        status="ready",
        input_slice=base.copy(),
        output_slice=base + 1,
    )
    second = PipelineBlock(
        tool_id="remove_step_diff",
        start=0,
        end=3,
        status="ready",
        input_slice=base + 1,
        output_slice=base + 3,
    )
    layer.blocks.extend([first, second])

    preview = pipeline_widget._collect_preview_results(update_status=True)
    np.testing.assert_allclose(preview[(1, "Ex1")], base + 3)
    assert first.status == "ready"
    assert second.status == "ready"
    pipeline_widget._refresh_all_pipeline_statuses()
    np.testing.assert_allclose(pipeline_widget._lane_widgets[0].samples, base + 3)

    first.output_slice = base + 10
    preview = pipeline_widget._collect_preview_results(update_status=True)
    np.testing.assert_allclose(preview[(1, "Ex1")], base + 12)
    assert first.status == "ready"
    assert second.status == "ready"
    np.testing.assert_allclose(second.input_slice, base + 10)
    np.testing.assert_allclose(second.output_slice, base + 12)
    pipeline_widget._refresh_all_pipeline_statuses()
    np.testing.assert_allclose(pipeline_widget._lane_widgets[0].samples, base + 12)

    changed = []
    pipeline_widget.change_finished_signal.connect(
        lambda data_id, channel: changed.append((data_id, channel))
    )
    with patch("ui.data_process_pipeline.QMessageBox.information"):
        pipeline_widget._commit_preview()

    np.testing.assert_allclose(
        data.get_channel_series("Ex1").to_numpy(dtype=np.float64, copy=True),
        base + 12,
    )
    np.testing.assert_allclose(channel_pipeline.baseline_snapshot, base + 12)
    assert first.output_slice is None
    assert second.output_slice is None
    assert changed == [(1, "Ex1")]

    widget.close()


def test_pipeline_replays_deletion_without_overwriting_unchanged_upstream_values():
    cached_input = np.array([1.0, 2.0, 3.0, 4.0])
    cached_output = np.array([1.0, np.nan, 5.0, 4.0])
    updated_upstream = np.array([10.0, 20.0, 30.0, 40.0])

    replayed = DataProcessPipelineWidget._replay_cached_edit(
        cached_input,
        cached_output,
        updated_upstream,
    )

    np.testing.assert_allclose(replayed, [10.0, np.nan, 32.0, 40.0], equal_nan=True)


def test_pipeline_only_lists_executable_channel_transform_tools():
    assert "remove_spike" in TOOL_META
    assert "remove_step_diff" in TOOL_META
    assert "robust_estimate" not in TOOL_META
    assert "remote_reference" not in TOOL_META


def test_pipeline_commit_refuses_unprocessed_blocks(qtbot):
    _app()
    manager = DataManager()
    data = DemoData("demo", ["Ex1"])
    key = manager.add(data)
    pipeline_widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(pipeline_widget)
    pipeline_widget.add_channels([(key, "Ex1")])
    pipeline_widget.pipelines[0].layers[0].blocks.append(
        PipelineBlock(tool_id="remove_spike", start=0, end=3)
    )
    before = data.get_channel_series("Ex1").copy()

    with patch("ui.data_process_pipeline.QMessageBox.warning") as warning:
        pipeline_widget._commit_preview()

    warning.assert_called_once()
    pd.testing.assert_series_equal(data.get_channel_series("Ex1"), before)
    assert pipeline_widget.pipelines[0].layers[0].blocks[0].status == "empty"
    pipeline_widget.close()
