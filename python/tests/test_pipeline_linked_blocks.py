from __future__ import annotations

import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialogButtonBox,
    QMessageBox,
    QPushButton,
    QWidget,
)

from base.data_manager import DataManager
from core.em_data import EMData
from core.pipeline_model import PipelineBlock
from ui.data_process_pipeline import DataProcessPipelineWidget
from ui.data_process_shared import create_processing_tool_model
from ui.plugin_adapter import (
    PIPELINE_LINK_ALL_TEXT,
    PluginParamDialog,
    create_pipeline_link_all_checkbox,
)
from ui.tools import registry as tool_registry


class LinkedData(EMData):
    def __init__(
        self,
        channels: list[str],
        *,
        periods: int = 12,
        sample_seconds: int = 1,
    ) -> None:
        super().__init__("/tmp/pipeline-linked.mat")
        self.name = "pipeline linked"
        self.start_time = pd.Timestamp("2025-01-01", tz="Asia/Shanghai")
        self.datetime_index = pd.date_range(
            self.start_time,
            periods=periods,
            freq=pd.Timedelta(seconds=sample_seconds),
        )
        self.end_time = self.datetime_index[-1]
        self.dt = pd.Timedelta(seconds=sample_seconds)
        self.kp_data = None
        self.data = xr.Dataset(coords={"time": self.datetime_index})
        for offset, channel in enumerate(channels):
            self.data[channel] = xr.DataArray(
                np.arange(periods, dtype=np.float64) + offset * 20,
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


def _linked_widget(qtbot):
    _app()
    manager = DataManager()
    data = LinkedData(["Ex", "Ey", "Hx"])
    data_id = manager.add(data)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(data, data_id)
    return widget, data_id


def test_linked_group_creates_one_independent_block_per_channel(qtbot):
    widget, data_id = _linked_widget(qtbot)
    source = widget._find_pipeline(data_id, "Ex")
    assert source is not None
    params = widget._default_params_for_tool("remove_wavelet_denoise")
    params.update({
        "wavelet": "db6",
        "threshold": 1.75,
        "region": [1, 6],
        "operations": [{"kind": "replace", "start": 0, "values": [1.0]}],
    })
    source_block = PipelineBlock(
        tool_id="remove_wavelet_denoise",
        start=2,
        end=9,
        params=params,
        status="configured",
    )
    source.layers[0].blocks.append(source_block)

    widget._set_block_linked(source_block.block_id, params, linked=True)

    members = widget._group_members(source_block.settings_group_id)
    assert len(members) == 3
    assert len({block.block_id for _pipeline, _layer, block in members}) == 3
    assert {block.settings_group_id for _pipeline, _layer, block in members} == {
        source_block.settings_group_id
    }
    group = widget._settings_groups[source_block.settings_group_id]
    assert group.shared_params["wavelet"] == "db6"
    assert group.shared_params["threshold"] == 1.75
    assert "region" not in group.shared_params
    assert "operations" not in group.shared_params

    for pipeline, _layer, block in members:
        assert (block.start, block.end) == (2, 9)
        assert block.input_slice is None
        assert block.output_slice is None
        if pipeline.channel == "Ex":
            assert block.params["region"] == [1, 6]
            assert block.params["operations"]
        else:
            assert block.params["region"] == []
            assert block.params["operations"] == []
    widget.close()


def test_linked_group_setting_update_invalidates_every_member(qtbot):
    widget, data_id = _linked_widget(qtbot)
    source = widget._find_pipeline(data_id, "Ex")
    assert source is not None
    params = widget._default_params_for_tool("median_filter")
    params["window_points"] = 5
    source_block = PipelineBlock(
        tool_id="median_filter",
        start=1,
        end=10,
        params=params,
        status="ready",
        input_slice=np.arange(9, dtype=np.float64),
        output_slice=np.arange(9, dtype=np.float64),
    )
    source.layers[0].blocks.append(source_block)
    widget._set_block_linked(source_block.block_id, params, linked=True)

    updated = dict(params)
    updated["window_points"] = 9
    widget._set_block_linked(source_block.block_id, updated, linked=True)

    members = widget._group_members(source_block.settings_group_id)
    assert len(members) == 3
    for _pipeline, _layer, block in members:
        assert block.params["window_points"] == 9
        assert block.status == "configured"
        assert block.input_slice is None
        assert block.output_slice is None
    widget.close()


def test_unchecking_link_all_dissolves_the_whole_group(qtbot):
    widget, data_id = _linked_widget(qtbot)
    source = widget._find_pipeline(data_id, "Ex")
    assert source is not None
    params = widget._default_params_for_tool("median_filter")
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

    widget._set_block_linked(source_block.block_id, params, linked=False)

    assert source_block.settings_group_id == ""
    assert widget._group_members(group_id) == []
    assert group_id not in widget._settings_groups
    assert all(
        block.settings_group_id == ""
        for pipeline in widget.pipelines
        for layer in pipeline.layers
        for block in layer.blocks
    )
    widget.close()


def test_removing_reference_promotes_a_remaining_group_member(qtbot):
    widget, data_id = _linked_widget(qtbot)
    source = widget._find_pipeline(data_id, "Ex")
    assert source is not None
    params = widget._default_params_for_tool("median_filter")
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

    widget._on_block_removed(source_block.block_id)

    remaining = widget._group_members(group_id)
    assert len(remaining) == 2
    group = widget._settings_groups[group_id]
    assert group.reference_block_id in {
        block.block_id for _pipeline, _layer, block in remaining
    }
    assert group.shared_params["window_points"] == params["window_points"]
    widget.close()


def test_plugin_param_dialog_exposes_pipeline_link_checkbox(qtbot):
    _app()
    plugin = tool_registry.get_plugin("median_filter")
    assert plugin is not None
    dialog = PluginParamDialog(
        plugin.info,
        {},
        allow_link_all=True,
        link_all=True,
    )
    qtbot.addWidget(dialog)

    assert dialog.link_all_checkbox is not None
    assert dialog.link_all_checkbox.isChecked()
    assert dialog.link_all_checkbox.text() == PIPELINE_LINK_ALL_TEXT
    dialog.show()
    QApplication.processEvents()
    buttons = dialog.findChild(QDialogButtonBox)
    assert buttons is not None
    checkbox_pos = dialog.link_all_checkbox.mapTo(dialog, QPoint())
    buttons_pos = buttons.mapTo(dialog, QPoint())
    assert abs(checkbox_pos.y() - buttons_pos.y()) < buttons.height()
    assert checkbox_pos.x() < buttons_pos.x()
    dialog.close()


def test_automatic_pipeline_tools_share_the_all_channel_control(qtbot):
    _app()
    model = create_processing_tool_model()
    automatic_category = model.item(1)
    assert automatic_category is not None

    tool_ids = [
        automatic_category.child(row).data(Qt.ItemDataRole.UserRole + 1)
        for row in range(automatic_category.rowCount())
    ]
    assert tool_ids
    assert all(tool_registry.get_plugin(tool_id) is not None for tool_id in tool_ids)

    parent = QWidget()
    qtbot.addWidget(parent)
    checkbox = create_pipeline_link_all_checkbox(parent, checked=False)
    assert checkbox.text() == "应用到当前 Pipeline 的全部通道"
    assert not checkbox.isChecked()


def test_linked_groups_get_distinct_badges_and_group_selection(qtbot):
    widget, data_id = _linked_widget(qtbot)
    source = widget._find_pipeline(data_id, "Ex")
    assert source is not None

    first_params = widget._default_params_for_tool("median_filter")
    first = PipelineBlock(
        tool_id="median_filter",
        start=0,
        end=5,
        params=first_params,
        status="configured",
    )
    second_params = widget._default_params_for_tool("detrend")
    second = PipelineBlock(
        tool_id="detrend",
        start=6,
        end=11,
        params=second_params,
        status="configured",
    )
    source.layers[0].blocks.extend([first, second])

    widget._set_block_linked(first.block_id, first_params, linked=True)
    widget._set_block_linked(second.block_id, second_params, linked=True)

    first_group = widget._settings_groups[first.settings_group_id]
    second_group = widget._settings_groups[second.settings_group_id]
    assert first_group.display_label == "A"
    assert second_group.display_label == "B"
    assert widget._settings_group_visual(first.settings_group_id)[1] != (
        widget._settings_group_visual(second.settings_group_id)[1]
    )
    assert "联动组 A · 3 个通道" in widget._settings_group_tooltip(
        first.settings_group_id
    )

    widget.selected_block_id = first.block_id
    widget._refresh_selection()
    assert all(
        track.selected_group_id == first.settings_group_id
        for lane in widget._lane_widgets
        for track in lane._tracks
    )
    widget.close()


@pytest.mark.parametrize(
    ("tool_id", "confirmation_text"),
    (
        ("auto_spike_filter", "确认："),
        ("emd_denoise", "运行 EMD 并应用重构"),
        ("median_filter", "确认应用"),
    ),
)
def test_automatic_editor_places_link_checkbox_beside_confirmation(
    qtbot,
    tool_id,
    confirmation_text,
):
    widget, data_id = _linked_widget(qtbot)
    source = widget._find_pipeline(data_id, "Ex")
    assert source is not None
    params = widget._default_params_for_tool(tool_id)
    block = PipelineBlock(
        tool_id=tool_id,
        start=0,
        end=12,
        params=params,
        status="empty",
    )
    layer = source.layers[0]
    layer.blocks.append(block)

    widget._open_interactive_editor(source, layer, block)
    assert widget._editor_dialogs
    dialog = widget._editor_dialogs[-1]
    qtbot.addWidget(dialog)
    QApplication.processEvents()

    checkbox = next(
        item
        for item in dialog.findChildren(QCheckBox)
        if item.text() == PIPELINE_LINK_ALL_TEXT
    )
    confirmation = next(
        item
        for item in dialog.findChildren(QPushButton)
        if item.text().startswith(confirmation_text)
    )
    checkbox_pos = checkbox.mapTo(dialog, QPoint())
    confirmation_pos = confirmation.mapTo(dialog, QPoint())
    assert checkbox.isVisibleTo(dialog)
    assert abs(checkbox_pos.y() - confirmation_pos.y()) < confirmation.height()
    assert checkbox_pos.x() < confirmation_pos.x()
    dialog.close()
    widget.close()


def test_linked_group_scales_point_settings_for_target_sample_rate(qtbot):
    _app()
    manager = DataManager()
    slow = LinkedData(["Ex"], periods=6, sample_seconds=2)
    fast = LinkedData(["Hx"], periods=11, sample_seconds=1)
    slow_id = manager.add(slow)
    fast_id = manager.add(fast)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.add_channels([(slow_id, "Ex"), (fast_id, "Hx")])
    source = widget._find_pipeline(slow_id, "Ex")
    assert source is not None
    params = widget._default_params_for_tool("median_filter")
    params.update({"window_points": 5, "min_segment_points": 5})
    source_block = PipelineBlock(
        tool_id="median_filter",
        start=0,
        end=6,
        params=params,
        status="configured",
    )
    source.layers[0].blocks.append(source_block)

    widget._set_block_linked(source_block.block_id, params, linked=True)

    members = widget._group_members(source_block.settings_group_id)
    target = next(block for pipeline, _layer, block in members if pipeline.data_id == fast_id)
    assert (target.start, target.end) == (0, 11)
    assert target.params["window_points"] == 11
    assert target.params["min_segment_points"] == 10
    widget.close()


def test_linked_group_members_follow_geometry_during_drag_preview(qtbot):
    widget, data_id = _linked_widget(qtbot)
    source = widget._find_pipeline(data_id, "Ex")
    assert source is not None
    params = widget._default_params_for_tool("median_filter")
    source_block = PipelineBlock(
        tool_id="median_filter",
        start=1,
        end=8,
        params=params,
        status="configured",
    )
    source.layers[0].blocks.append(source_block)
    widget._set_block_linked(source_block.block_id, params, linked=True)
    members = widget._group_members(source_block.settings_group_id)
    for pipeline, _layer, block in members:
        pipeline.add_layer()
        block.status = "ready"
        block.input_slice = np.arange(7, dtype=np.float64)
        block.output_slice = np.arange(7, dtype=np.float64)

    source.layers[0].blocks.remove(source_block)
    source.layers[1].blocks.append(source_block)
    source_block.start = 3
    source_block.end = 10

    widget._rebuild_lanes()
    source_lane = next(
        lane for lane in widget._lane_widgets if lane.pipeline is source
    )
    source_track = next(
        track for track in source_lane._tracks if track.layer is source.layers[1]
    )
    source_track.block_geometry_preview_changed.emit(source_block.block_id)

    for pipeline, layer, block in widget._group_members(
        source_block.settings_group_id
    ):
        assert (block.start, block.end) == (3, 10)
        assert layer is pipeline.layers[1]
        assert block.status == "ready"
        assert block.input_slice is not None
        assert block.output_slice is not None
    widget.close()



def test_unlinking_one_block_preserves_the_remaining_linked_group(qtbot):
    widget, data_id = _linked_widget(qtbot)
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

    widget._on_block_unlink_requested(source_block.block_id)

    assert source_block.settings_group_id == ""
    remaining = widget._group_members(group_id)
    assert len(remaining) == 2
    assert all(block.settings_group_id == group_id for _p, _l, block in remaining)
    assert group_id in widget._settings_groups
    assert widget._settings_groups[group_id].reference_block_id in {
        block.block_id for _pipeline, _layer, block in remaining
    }
    widget.close()


def test_changed_linked_params_yes_detaches_only_current_block(qtbot):
    widget, data_id = _linked_widget(qtbot)
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
    updated = dict(params, window_points=9)

    with patch.object(
        QMessageBox,
        "question",
        return_value=QMessageBox.StandardButton.Yes,
    ) as question:
        detached = widget._on_plugin_params_applied(
            source_block.block_id,
            updated,
            linked=True,
        )

    assert detached
    assert question.call_count == 1
    assert source_block.settings_group_id == ""
    assert source_block.params["window_points"] == 9
    remaining = widget._group_members(group_id)
    assert len(remaining) == 2
    assert all(block.params["window_points"] == 5 for _p, _l, block in remaining)
    widget.close()


def test_changed_linked_params_no_keeps_group_and_updates_every_member(qtbot):
    widget, data_id = _linked_widget(qtbot)
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
    updated = dict(params, window_points=9)

    with patch.object(
        QMessageBox,
        "question",
        return_value=QMessageBox.StandardButton.No,
    ) as question:
        detached = widget._on_plugin_params_applied(
            source_block.block_id,
            updated,
            linked=False,
        )

    assert not detached
    assert question.call_count == 1
    message = question.call_args.args[2]
    assert "选择「否」" in message
    assert "全部 3 个通道" in message
    members = widget._group_members(group_id)
    assert len(members) == 3
    assert all(block.params["window_points"] == 9 for _p, _l, block in members)
    widget.close()


def test_unchanged_linked_params_do_not_show_detach_prompt(qtbot):
    widget, data_id = _linked_widget(qtbot)
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

    with patch.object(QMessageBox, "question") as question:
        widget._on_plugin_params_applied(
            source_block.block_id,
            dict(params),
            linked=True,
        )

    question.assert_not_called()
    widget.close()
