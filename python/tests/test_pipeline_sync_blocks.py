from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import xarray as xr
from PySide6.QtCore import QByteArray, QMimeData, QPoint, QPointF, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QApplication

from base.data_manager import DataManager
from core.em_data import EMData
from core.pipeline_model import PipelineBlock
from ui.data_process_shared import (
    PROCESSING_TOOL_MIME,
    PIPELINE_TOOL_MIME,
    create_processing_tool_model,
)
from ui.data_tree_view import ToolTreeView
from ui.data_process_pipeline import DataProcessPipelineWidget, pipeline_tool_id_from_mime


class _SyncData(EMData):
    def __init__(self, channels: list[str]) -> None:
        super().__init__("/tmp/pipeline-sync.mat")
        self.name = "pipeline sync"
        self.start_time = pd.Timestamp("2025-01-01 00:00:00", tz="Asia/Shanghai")
        self.datetime_index = pd.date_range(
            self.start_time,
            periods=6,
            freq="s",
        )
        self.end_time = self.datetime_index[-1]
        self.dt = pd.Timedelta(seconds=1)
        self.kp_data = None
        self.data = xr.Dataset(coords={"time": self.datetime_index})
        for offset, channel in enumerate(channels):
            self.data[channel] = xr.DataArray(
                np.arange(6, dtype=np.float64) + offset * 10,
                dims=("time",),
                coords={"time": self.datetime_index},
            )

    def restore_data(self, channel: str):
        del channel


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_pipeline_accepts_shared_processing_tool_drag_mime():
    shared_mime = QMimeData()
    shared_mime.setData(PROCESSING_TOOL_MIME, QByteArray(b"remove step diff"))
    assert pipeline_tool_id_from_mime(shared_mime) == "remove step diff"

    pipeline_mime = QMimeData()
    pipeline_mime.setData(PIPELINE_TOOL_MIME, QByteArray(b"remove_step_diff"))
    assert pipeline_tool_id_from_mime(pipeline_mime) == "remove_step_diff"


def test_shared_processing_tool_tree_emits_pipeline_drag_mime(monkeypatch):
    _app()
    captured = {}

    class _Drag:
        def __init__(self, parent):
            captured["parent"] = parent

        def setMimeData(self, mime_data):
            captured["mime_data"] = mime_data

        def exec(self, supported_actions, default_action):
            captured["actions"] = supported_actions, default_action

    monkeypatch.setattr("ui.data_tree_view.QDrag", _Drag)
    tree = ToolTreeView()
    model = create_processing_tool_model()
    tree.setModel(model)
    tree.setCurrentIndex(model.item(0).child(0).index())

    tree.startDrag(Qt.DropAction.CopyAction)

    mime_data = captured["mime_data"]
    assert mime_data.hasFormat(PROCESSING_TOOL_MIME)
    assert mime_data.hasFormat(PIPELINE_TOOL_MIME)
    assert pipeline_tool_id_from_mime(mime_data) == "remove region"
    tree.deleteLater()


def test_pipeline_track_adds_block_from_shared_processing_tool_drag(qtbot):
    _app()
    manager = DataManager()
    data = _SyncData(["Ex"])
    data_id = manager.add(data)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(data, data_id)
    track = widget._lane_widgets[0]._tracks[0]

    mime_data = QMimeData()
    mime_data.setData(PROCESSING_TOOL_MIME, QByteArray(b"remove step diff"))
    enter_event = QDragEnterEvent(
        QPoint(12, 8),
        Qt.DropAction.CopyAction,
        mime_data,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    track.dragEnterEvent(enter_event)
    assert enter_event.isAccepted()

    drop_event = QDropEvent(
        QPointF(12, 8),
        Qt.DropAction.CopyAction,
        mime_data,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    track.dropEvent(drop_event)

    assert drop_event.isAccepted()
    assert widget.block_count() == 1
    assert widget.pipelines[0].layers[0].blocks[0].tool_id == "remove_step_diff"
    widget.close()


def test_workspace_level_tool_drop_routes_to_pipeline_track(qtbot, monkeypatch):
    _app()
    manager = DataManager()
    data = _SyncData(["Ex"])
    data_id = manager.add(data)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(data, data_id)
    track = widget._lane_widgets[0]._tracks[0]
    monkeypatch.setattr(
        "ui.data_process_pipeline.QApplication.widgetAt",
        lambda _position: track,
    )

    assert widget.drop_tool_at_global_position("remove step diff", QPoint(12, 8))
    assert widget.block_count() == 1
    assert widget.pipelines[0].layers[0].blocks[0].tool_id == "remove_step_diff"
    widget.close()


def test_remove_spike_channel_sync_stages_independent_pipeline_blocks(qtbot):
    _app()
    manager = DataManager()
    data = _SyncData(["Ex", "Ey", "Hx"])
    data_id = manager.add(data)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(data, data_id)

    source = widget._find_pipeline(data_id, "Ex")
    assert source is not None
    source_block = PipelineBlock(
        tool_id="remove_spike",
        start=1,
        end=5,
        params={"indices": [1, 3]},
        status="configured",
    )
    source.layers[0].blocks.append(source_block)

    before = {
        channel: data.get_channel_series(channel).to_numpy(dtype=np.float64, copy=True)
        for channel in data.channel_names()
    }
    writes: list[tuple[int, str]] = []
    widget.change_finished_signal.connect(lambda key, channel: writes.append((key, channel)))

    widget._on_editor_channel_sync(source_block.block_id)

    assert writes == []
    for channel, values in before.items():
        np.testing.assert_allclose(
            data.get_channel_series(channel).to_numpy(dtype=np.float64),
            values,
        )

    for channel in ("Ey", "Hx"):
        target = widget._find_pipeline(data_id, channel)
        assert target is not None
        assert len(target.layers[0].blocks) == 1
        clone = target.layers[0].blocks[0]
        assert clone.block_id != source_block.block_id
        assert clone.tool_id == "remove_spike"
        assert (clone.start, clone.end) == (1, 5)
        assert clone.params == {"indices": [1, 3]}
        assert clone.status == "configured"
        assert clone.input_slice is None
        assert clone.output_slice is None

    widget.close()


def test_remove_spike_channel_sync_uses_current_pipeline_channels_only(qtbot):
    _app()
    manager = DataManager()
    first = _SyncData(["Ex", "Ey"])
    first_id = manager.add(first)
    second = _SyncData(["Hx"])
    second_id = manager.add(second)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.add_channels([(first_id, "Ex"), (first_id, "Ey"), (second_id, "Hx")])

    source = widget._find_pipeline(first_id, "Ex")
    assert source is not None
    source_block = PipelineBlock(
        tool_id="remove_spike",
        start=0,
        end=3,
        params={"indices": [1]},
        status="configured",
    )
    source.layers[0].blocks.append(source_block)

    widget._on_editor_channel_sync(source_block.block_id)

    same_dataset = widget._find_pipeline(first_id, "Ey")
    other_dataset = widget._find_pipeline(second_id, "Hx")
    assert same_dataset is not None and len(same_dataset.layers[0].blocks) == 1
    assert other_dataset is not None and other_dataset.layers[0].blocks == []
    widget.close()


def test_remove_region_cross_dataset_sync_stages_blocks_without_writes(qtbot):
    _app()
    manager = DataManager()
    first = _SyncData(["Ex"])
    first_id = manager.add(first)
    second = _SyncData(["Hx"])
    second_id = manager.add(second)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.add_channels([(first_id, "Ex"), (second_id, "Hx")])

    source = widget._find_pipeline(first_id, "Ex")
    assert source is not None
    source_block = PipelineBlock(
        tool_id="remove_region",
        start=1,
        end=5,
        params={
            "time_ranges": [[
                first.datetime_index[2].isoformat(),
                first.datetime_index[3].isoformat(),
            ]],
        },
        status="configured",
    )
    source.layers[0].blocks.append(source_block)
    before = second.get_channel_series("Hx").to_numpy(dtype=np.float64, copy=True)

    widget._on_editor_region_sync(source_block.block_id)

    target = widget._find_pipeline(second_id, "Hx")
    assert target is not None
    assert len(target.layers[0].blocks) == 1
    clone = target.layers[0].blocks[0]
    assert clone.tool_id == "remove_region"
    assert clone.params == source_block.params
    np.testing.assert_allclose(second.get_channel_series("Hx"), before)
    widget.close()
