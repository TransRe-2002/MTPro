import json

import numpy as np
import pandas as pd
import xarray as xr
from PySide6.QtCore import QByteArray, QMimeData, QPoint, Qt
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QTabWidget
from unittest.mock import patch

from base.data_manager import DataManager
from core.em_data import EMData
from ui.data_process import DataProcessWidget, WorkspaceMdiArea
from ui.models.processing_tools import DATA_TREE_SELECTION_MIME
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


def test_pipeline_toolbar_omits_preview_summary_button(qtbot):
    _app()
    manager = DataManager()
    data = DemoData("demo", ["Ex1"])
    data_id = manager.add(data)
    pipeline = DataProcessPipelineWidget(manager)
    pipeline.init_data(data, data_id)
    qtbot.addWidget(pipeline)

    button_texts = {button.text() for button in pipeline.findChildren(QPushButton)}
    assert "preview" not in button_texts
    assert "commit" in button_texts
    pipeline.close()


def test_data_process_host_stays_empty_when_active_data_changes():
    _app()
    widget = DataProcessWidget()
    data = DemoData("demo", ["Ex1", "Hy1"])

    widget.init_data(data, 1)
    assert widget.workspace.subWindowList() == []
    assert widget.pipeline_widget is None
    widget.init_data(DemoData("other", ["Hy1"]), 2)
    assert widget.workspace.subWindowList() == []
    assert widget.pipeline_widget is None

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


def test_data_process_restores_workspace_tabs():
    _app()
    widget = DataProcessWidget()

    assert isinstance(widget.workspace, WorkspaceMdiArea)
    assert widget.current_workspace() is widget.workspace
    tabs = widget.pipeline_panel.workspace_tabs
    assert isinstance(tabs, QTabWidget)
    assert [tabs.tabText(index) for index in range(tabs.count())] == ["工作区 1", "+"]
    assert tabs.currentWidget() is widget.workspace

    widget.close()


def test_workspace_plus_tab_starts_empty_and_preserves_other_workspace_state(qtbot):
    widget = DataProcessWidget()
    qtbot.addWidget(widget)
    widget.init_data(DemoData("demo", ["Ex1"]), 1)
    panel = widget.pipeline_panel
    tabs = panel.workspace_tabs
    first_workspace = widget.workspace
    first_pipeline = widget._create_pipeline_window()
    block = PipelineBlock(tool_id="remove_spike", start=0, end=3)
    first_pipeline.pipelines[0].layers[0].blocks.append(block)

    tabs.setCurrentIndex(tabs.count() - 1)

    assert tabs.count() == 3
    assert widget.workspace is not first_workspace
    assert widget.pipeline_widget is None
    assert widget.workspace.subWindowList() == []
    tabs.setCurrentWidget(first_workspace)
    assert widget.pipeline_widget is first_pipeline
    assert first_pipeline.pipelines[0].layers[0].blocks == [block]


def test_reordered_workspace_tabs_close_the_selected_page_and_keep_plus_last(qtbot):
    widget = DataProcessWidget()
    qtbot.addWidget(widget)
    widget.init_data(DemoData("demo", ["Ex1"]), 1)
    panel = widget.pipeline_panel
    tabs = panel.workspace_tabs
    first = widget.workspace
    second = panel._create_workspace()
    third = panel._create_workspace()
    tabs.tabBar().moveTab(tabs.indexOf(first), tabs.indexOf(third))
    tabs.tabBar().moveTab(tabs.count() - 1, 0)
    assert tabs.count() == 4
    assert tabs.tabText(tabs.count() - 1) == "+"

    tabs.setCurrentWidget(second)
    panel._close_workspace(tabs.indexOf(first))
    assert panel.workspaces() == [second, third]
    assert widget.workspace is second
    panel._close_workspace(tabs.indexOf(second))
    assert widget.workspace is third
    panel._close_workspace(tabs.indexOf(third))
    assert panel.workspaces() == [third]
    assert tabs.count() == 2


def test_workspace_context_and_data_cleanup_reach_inactive_tabs(qtbot):
    manager = DataManager()
    first_data = DemoData("one", ["Ex1"])
    second_data = DemoData("two", ["Hy1"])
    first_key, second_key = manager.add(first_data), manager.add(second_data)
    widget = DataProcessWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(first_data, first_key)
    widget._create_pipeline_window()
    panel = widget.pipeline_panel
    first = widget.workspace
    second = panel._create_workspace()
    widget._create_pipeline_window()
    resolver = lambda: (second_key, "Hy1")
    payload = lambda: {"kind": "dataset", "data_id": second_key}
    cleared = []
    widget.set_context_resolver(resolver)
    widget.set_selection_payload_resolver(payload)
    widget.set_clear_checked_channels(cleared.extend)
    widget.init_data(second_data, second_key)
    third = panel._create_workspace()
    widget._create_pipeline_window()
    for workspace in (first, second, third):
        assert workspace.data_id == second_key
        assert workspace._context_resolver is resolver
        assert workspace._selection_payload_resolver is payload
        workspace._clear_checked_channels([(second_key, "Hy1")])
    assert cleared == [(second_key, "Hy1")] * 3
    widget.close_windows_for_data(first_key)
    for workspace in panel.workspaces():
        for subwindow in workspace.subWindowList():
            assert all(key != first_key for key, _ in subwindow.widget().channel_refs())
    # Closing a workspace must not unload its source datasets.
    panel._close_workspace(panel.workspace_tabs.indexOf(second))
    assert manager.get(first_key) is first_data
    assert manager.get(second_key) is second_data


def test_data_process_creates_a_new_pipeline_mdi_subwindow(qtbot):
    _app()
    data = DemoData("demo", ["Ex1"])
    widget = DataProcessWidget()
    qtbot.addWidget(widget)
    widget.init_data(data, 1)

    assert widget.pipeline_widget is None
    qtbot.mouseClick(widget.new_pipeline_button, Qt.MouseButton.LeftButton)
    first_pipeline = widget.pipeline_widget
    qtbot.mouseClick(widget.new_pipeline_button, Qt.MouseButton.LeftButton)
    second_pipeline = widget.workspace.active_pipeline_window()

    assert first_pipeline is not None
    assert second_pipeline is not None
    assert second_pipeline is not first_pipeline
    assert len(widget.workspace.subWindowList()) == 2
    assert second_pipeline.channel_refs() == [(1, "Ex1")]

    widget.close()


def test_pipeline_can_hide_and_restore_a_channel_without_removing_its_blocks(qtbot):
    _app()
    manager = DataManager()
    data = DemoData("demo", ["Ex1", "Hy1"])
    key = manager.add(data)
    pipeline_widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(pipeline_widget)
    pipeline_widget.add_channels([(key, "Ex1"), (key, "Hy1")])

    hidden_pipeline = pipeline_widget._find_pipeline(key, "Hy1")
    assert hidden_pipeline is not None
    hidden_block = PipelineBlock(tool_id="remove_spike", start=0, end=3)
    hidden_pipeline.layers[0].blocks.append(hidden_block)

    pipeline_widget._set_channel_visible((key, "Hy1"), False)

    assert pipeline_widget.channel_refs() == [(key, "Ex1"), (key, "Hy1")]
    assert pipeline_widget.visible_channel_refs() == [(key, "Ex1")]
    assert [lane.pipeline.ref for lane in pipeline_widget._lane_widgets] == [(key, "Ex1")]
    assert hidden_pipeline.layers[0].blocks == [hidden_block]
    assert [action.text() for action in pipeline_widget.channel_visibility_menu.actions()] == [
        f"#{key} Ex1",
        f"#{key} Hy1",
    ]
    assert not pipeline_widget.channel_visibility_menu.actions()[1].isChecked()

    pipeline_widget._set_channel_visible((key, "Hy1"), True)

    assert pipeline_widget.visible_channel_refs() == [(key, "Ex1"), (key, "Hy1")]
    assert [lane.pipeline.ref for lane in pipeline_widget._lane_widgets] == [
        (key, "Ex1"),
        (key, "Hy1"),
    ]
    assert hidden_pipeline.layers[0].blocks == [hidden_block]
    pipeline_widget.close()


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


def test_empty_tab_populates_only_from_dropped_data_and_appends_channels(qtbot, monkeypatch):
    manager = DataManager()
    first_key = manager.add(DemoData("first", ["Ex1", "Hy1"]))
    second_key = manager.add(DemoData("second", ["Hy1"]))
    widget = DataProcessWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(manager.get(first_key), first_key)
    first_workspace = widget.workspace
    assert first_workspace.subWindowList() == []

    dataset_drop = DropStub({"kind": "dataset", "data_id": second_key})
    first_workspace.dropEvent(dataset_drop)
    assert dataset_drop.accepted
    first_pipeline = widget.pipeline_widget
    assert first_pipeline.channel_refs() == [(second_key, "Hy1")]

    second_workspace = widget.pipeline_panel._create_workspace()
    assert second_workspace.subWindowList() == []
    channel_drop = DropStub({
        "kind": "channels", "refs": [{"data_id": first_key, "channel_name": "Ex1"}],
    })
    second_workspace.dropEvent(channel_drop)
    assert channel_drop.accepted
    second_pipeline = widget.pipeline_widget
    assert second_pipeline.channel_refs() == [(first_key, "Ex1")]

    subwindow = second_workspace.pipeline_subwindow(second_pipeline)
    monkeypatch.setattr(second_workspace, "_subwindow_at", lambda _pos: subwindow)
    append_drop = DropStub({
        "kind": "channels", "refs": [{"data_id": first_key, "channel_name": "Hy1"}],
    })
    second_workspace.dropEvent(append_drop)
    assert append_drop.accepted
    assert second_pipeline.channel_refs() == [(first_key, "Ex1"), (first_key, "Hy1")]
    assert len(second_workspace.subWindowList()) == 1
    widget.init_data(manager.get(second_key), second_key)
    assert widget.pipeline_widget is second_pipeline
    assert len(second_workspace.subWindowList()) == 1
    widget.pipeline_panel.workspace_tabs.setCurrentWidget(first_workspace)
    assert widget.pipeline_widget is first_pipeline
    assert first_pipeline.channel_refs() == [(second_key, "Hy1")]


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
    dataset_changes = []
    pipeline.change_finished_signal.connect(lambda data_id, channel: changed.append((data_id, channel)))
    manager.data_changed.connect(
        lambda data_id, structural: dataset_changes.append((data_id, structural))
    )
    with patch("ui.data_process_pipeline.QMessageBox.information"):
        pipeline._commit_preview()

    np.testing.assert_allclose(first.get_channel_series("Ex1").to_numpy(dtype=np.float64), [1.0, 12.0, 13.0, 14.0])
    np.testing.assert_allclose(second.get_channel_series("Hy1").to_numpy(dtype=np.float64), [10.0, 20.0, 30.0])
    assert changed == [(first_key, "Ex1")]
    assert dataset_changes == [(first_key, False)]
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
    widget._create_pipeline_window()

    pipeline = widget.workspace.last_pipeline_window()
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
    widget._create_pipeline_window()

    pipeline = widget.workspace.last_pipeline_window()
    assert pipeline is not None
    layer = pipeline.pipelines[0].layers[0]

    pipeline._on_tool_dropped(1, "Ex1", layer.layer_id, "remove_step_diff", 100)

    assert pipeline.block_count() == 1
    assert layer.blocks[0].tool_id == "remove_step_diff"

    widget.close()


def test_manual_removal_blocks_start_ready_with_an_identity_preview():
    _app()
    widget = DataProcessWidget()
    data = DemoData("demo", ["Ex1"])
    widget.init_data(data, 1)
    widget._create_pipeline_window()

    pipeline_widget = widget.pipeline_widget
    assert pipeline_widget is not None
    channel_pipeline = pipeline_widget.pipelines[0]
    first_layer = channel_pipeline.layers[0]
    pipeline_widget._on_tool_dropped(1, "Ex1", first_layer.layer_id, "remove_spike", 0)

    second_layer = channel_pipeline.add_layer()
    pipeline_widget._on_tool_dropped(1, "Ex1", second_layer.layer_id, "remove_region", 0)

    blocks = [first_layer.blocks[0], second_layer.blocks[0]]
    for block in blocks:
        assert block.status == "ready"
        np.testing.assert_allclose(block.input_slice, block.output_slice)
    assert pipeline_widget._unready_blocks() == []

    widget.close()


def test_pipeline_preview_commit_replays_downstream_edit_on_updated_upstream():
    _app()
    widget = DataProcessWidget()
    data = DemoData("demo", ["Ex1"])

    widget.init_data(data, 1)
    widget._create_pipeline_window()

    pipeline_widget = widget.workspace.last_pipeline_window()
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
    assert not pipeline_widget.clear_blocks_after_commit_checkbox.isChecked()
    assert first.output_slice is None
    assert second.output_slice is None
    assert changed == [(1, "Ex1")]

    widget.close()


def test_pipeline_refresh_reuses_unchanged_preview_and_overview_lod(qtbot):
    _app()
    manager = DataManager()
    data = DemoData("demo", ["Ex1"])
    data_id = manager.add(data)
    pipeline_widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(pipeline_widget)
    pipeline_widget.init_data(data, data_id)

    lane = pipeline_widget._lane_widgets[0]
    first_preview = lane.samples
    first_pyramid = lane.overview_widget._pyramid

    pipeline_widget._refresh_all_pipeline_statuses()

    assert lane.samples is first_preview
    assert lane.overview_widget._pyramid is first_pyramid

    channel_pipeline = pipeline_widget.pipelines[0]
    base = channel_pipeline.baseline_snapshot
    block = PipelineBlock(
        tool_id="remove_spike",
        start=0,
        end=base.size,
        status="ready",
        input_slice=base.copy(),
        output_slice=base + 1,
    )
    channel_pipeline.layers[0].blocks.append(block)
    pipeline_widget._refresh_all_pipeline_statuses()
    changed_preview = lane.samples
    changed_pyramid = lane.overview_widget._pyramid

    assert changed_preview is not first_preview
    assert changed_pyramid is not first_pyramid
    np.testing.assert_allclose(changed_preview, base + 1)

    pipeline_widget._refresh_all_pipeline_statuses()

    assert lane.samples is changed_preview
    assert lane.overview_widget._pyramid is changed_pyramid
    pipeline_widget.close()


def test_pipeline_commit_clears_blocks_when_clear_option_is_enabled(qtbot):
    _app()
    manager = DataManager()
    data = DemoData("demo", ["Ex1"])
    key = manager.add(data)
    pipeline_widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(pipeline_widget)
    pipeline_widget.add_channels([(key, "Ex1")])
    pipeline_widget.clear_blocks_after_commit_checkbox.setChecked(True)

    layer = pipeline_widget.pipelines[0].layers[0]
    base = data.get_channel_series("Ex1").to_numpy(dtype=np.float64, copy=True)
    block = PipelineBlock(
        tool_id="remove_spike",
        start=0,
        end=base.size,
        status="ready",
        input_slice=base.copy(),
        output_slice=base + 1,
    )
    layer.blocks.append(block)

    with patch("ui.data_process_pipeline.QMessageBox.information"):
        pipeline_widget._commit_preview()

    assert layer.blocks == []
    pipeline_widget.close()


def test_pipeline_commit_never_closes_other_pipeline_mdi_windows(qtbot):
    _app()
    widget = DataProcessWidget()
    qtbot.addWidget(widget)
    data = DemoData("demo", ["Ex1"])
    widget.init_data(data, 1)

    first_pipeline = widget._create_pipeline_window()
    assert first_pipeline is not None
    second_pipeline = widget._create_pipeline_window()
    assert second_pipeline is not None

    base = data.get_channel_series("Ex1").to_numpy(dtype=np.float64, copy=True)
    first_pipeline.pipelines[0].layers[0].blocks.append(
        PipelineBlock(
            tool_id="remove_spike",
            start=0,
            end=base.size,
            status="ready",
            input_slice=base.copy(),
            output_slice=base + 1,
        )
    )
    first_pipeline.clear_blocks_after_commit_checkbox.setChecked(True)

    with patch("ui.data_process_pipeline.QMessageBox.information"):
        first_pipeline._commit_preview()

    assert len(widget.workspace.subWindowList()) == 2
    assert widget.workspace.pipeline_subwindow(first_pipeline) is not None
    assert widget.workspace.pipeline_subwindow(second_pipeline) is not None
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
