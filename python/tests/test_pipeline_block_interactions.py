from __future__ import annotations

import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import xarray as xr
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication

from base.data_manager import DataManager
from core.em_data import EMData
from core.pipeline_model import PipelineBlock
from ui.data_process_pipeline import (
    PIPELINE_BLOCK_CLIPBOARD_MIME,
    DataProcessPipelineWidget,
    PipelineTrackWidget,
)
from ui.widgets.plugin_editor import PluginSliceEditor


class InteractionData(EMData):
    def __init__(self) -> None:
        super().__init__("/tmp/pipeline-block-interactions.mat")
        self.name = "pipeline block interactions"
        self.start_time = pd.Timestamp("2025-01-01", tz="Asia/Shanghai")
        self.datetime_index = pd.date_range(
            self.start_time, periods=12, freq=pd.Timedelta(seconds=1)
        )
        self.end_time = self.datetime_index[-1]
        self.dt = pd.Timedelta(seconds=1)
        self.kp_data = None
        self.data = xr.Dataset(coords={"time": self.datetime_index})
        for offset, channel in enumerate(("Ex", "Ey", "Hx")):
            self.data[channel] = xr.DataArray(
                np.arange(12, dtype=np.float64) + offset * 20,
                dims=("time",),
                coords={"time": self.datetime_index},
            )

    def restore_data(self, channel: str):
        del channel


def _widget(qtbot):
    app = QApplication.instance() or QApplication([])
    manager = DataManager()
    data = InteractionData()
    data_id = manager.add(data)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(data, data_id)
    return app, widget, data_id


def test_linked_editor_uses_clicked_channel_series_and_switches_member(qtbot):
    _app, widget, data_id = _widget(qtbot)
    source = widget._find_pipeline(data_id, "Ex")
    assert source is not None
    params = widget._default_params_for_tool("median_filter")
    params["window_points"] = 5
    source_block = PipelineBlock(
        tool_id="median_filter",
        start=1,
        end=10,
        params=params,
        status="configured",
    )
    source.layers[0].blocks.append(source_block)
    widget._set_block_linked(source_block.block_id, params, linked=True)
    group_id = source_block.settings_group_id

    widget._open_interactive_editor(source, source.layers[0], source_block)
    first_dialog = widget._group_editor_dialogs[group_id]
    first_editor = first_dialog.findChild(PluginSliceEditor)
    assert first_editor is not None
    assert first_editor._host is not None
    assert first_editor._host.block.channel == "Ex"
    np.testing.assert_allclose(
        first_editor._host.block.series.to_numpy(),
        np.arange(12, dtype=np.float64)[1:10],
    )

    target_pipeline, target_layer, target_block = next(
        member
        for member in widget._group_members(group_id)
        if member[0].channel == "Ey"
    )
    widget._open_interactive_editor(target_pipeline, target_layer, target_block)
    QApplication.processEvents()

    second_dialog = widget._group_editor_dialogs[group_id]
    second_editor = second_dialog.findChild(PluginSliceEditor)
    assert second_dialog is not first_dialog
    assert first_dialog not in widget._editor_dialogs
    assert second_editor is not None
    assert second_editor._host is not None
    assert second_editor._host.block.channel == "Ey"
    assert second_editor.params["window_points"] == 5
    np.testing.assert_allclose(
        second_editor._host.block.series.to_numpy(),
        np.arange(12, dtype=np.float64)[1:10] + 20,
    )
    second_dialog.close()
    widget.close()


def test_middle_click_requests_block_removal(qtbot):
    _app, widget, data_id = _widget(qtbot)
    pipeline = widget._find_pipeline(data_id, "Ex")
    assert pipeline is not None
    block = PipelineBlock(
        tool_id="median_filter",
        start=1,
        end=4,
        params=widget._default_params_for_tool("median_filter"),
        status="configured",
    )
    layer = pipeline.layers[0]
    layer.blocks.append(block)
    track = PipelineTrackWidget(pipeline, layer, 12, widget._snap, widget)
    qtbot.addWidget(track)
    track.set_zoom(6.0, pixels_per_point=20.0)
    track.show()

    removed = []
    track.block_removed.connect(removed.append)
    qtbot.mouseClick(
        track,
        Qt.MouseButton.MiddleButton,
        pos=QPoint(50, track.height() // 2),
    )
    assert removed == [block.block_id]
    widget.close()


def test_context_menu_and_independent_copy_paste(qtbot):
    app, widget, data_id = _widget(qtbot)
    app.clipboard().clear()
    source = widget._find_pipeline(data_id, "Ex")
    target = widget._find_pipeline(data_id, "Hx")
    assert source is not None
    assert target is not None
    params = widget._default_params_for_tool("remove_wavelet_denoise")
    params.update(
        {
            "wavelet": "db6",
            "threshold": 1.75,
            "region": [1, 3],
            "operations": [{"kind": "replace", "start": 0, "values": [1.0]}],
            "time_ranges": [["2025-01-01", "2025-01-02"]],
        }
    )
    source_block = PipelineBlock(
        tool_id="remove_wavelet_denoise",
        start=1,
        end=5,
        params=params,
        status="ready",
        input_slice=np.arange(4, dtype=np.float64),
        output_slice=np.arange(4, dtype=np.float64),
        settings_group_id="source-link-group",
    )
    source.layers[0].blocks.append(source_block)
    track = PipelineTrackWidget(source, source.layers[0], 12, widget._snap, widget)
    qtbot.addWidget(track)

    menu, actions = track._build_block_context_menu(source_block)
    assert set(actions) == {"settings", "unlink", "cut", "copy"}
    assert actions["settings"].text() == "参数设置" + chr(0x2026)
    assert actions["unlink"].text() == "解除当前 Block 联动"
    menu.deleteLater()

    assert widget._copy_block_to_clipboard(source_block.block_id)
    assert app.clipboard().mimeData().hasFormat(PIPELINE_BLOCK_CLIPBOARD_MIME)
    payload = widget._pipeline_block_from_clipboard()
    assert payload is not None
    assert payload["span_points"] == 4
    assert payload["params"]["wavelet"] == "db6"
    assert payload["params"]["threshold"] == 1.75
    assert not payload["params"].get("region")
    assert not payload["params"].get("operations")
    assert not payload["params"].get("time_ranges")

    menu, actions = track._build_block_context_menu(None)
    assert set(actions) == {"paste"}
    menu.deleteLater()

    with patch.object(widget, "_refresh_all_pipeline_statuses"):
        widget._paste_block_from_clipboard(
            data_id, "Hx", target.layers[0].layer_id, 6
        )

    assert len(target.layers[0].blocks) == 1
    pasted = target.layers[0].blocks[0]
    assert pasted.block_id != source_block.block_id
    assert pasted.tool_id == source_block.tool_id
    assert pasted.end - pasted.start == 4
    assert pasted.settings_group_id == ""
    assert pasted.input_slice is None
    assert pasted.output_slice is None
    assert pasted.params["wavelet"] == "db6"
    assert pasted.params["threshold"] == 1.75
    assert not pasted.params.get("region")
    assert not pasted.params.get("operations")
    assert not pasted.params.get("time_ranges")
    widget.close()


def test_cut_copies_then_removes_the_source_block(qtbot):
    app, widget, data_id = _widget(qtbot)
    app.clipboard().clear()
    pipeline = widget._find_pipeline(data_id, "Ex")
    assert pipeline is not None
    block = PipelineBlock(
        tool_id="median_filter",
        start=1,
        end=4,
        params=widget._default_params_for_tool("median_filter"),
        status="configured",
    )
    pipeline.layers[0].blocks.append(block)

    with patch.object(widget, "_refresh_all_pipeline_statuses"):
        widget._cut_block_to_clipboard(block.block_id)

    assert widget._find_block(block.block_id) is None
    assert app.clipboard().mimeData().hasFormat(PIPELINE_BLOCK_CLIPBOARD_MIME)
    widget.close()
