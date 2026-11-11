from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from PySide6.QtWidgets import QMessageBox

from base.batch_runner import BatchRunner, BatchTarget
from base.data_manager import DataManager
from core.em_data import EMData
from core.pipeline_model import PipelineBlock
from core.recipe import recipe_from_pipeline
from base.plugin_runtime import run_plugin_process
from plugin_api import ChannelBlock
from plugins.remove_step_baseline.plugin import StepBaselinePlugin
from ui.data_process_pipeline import DataProcessPipelineWidget
from ui.tools import registry
from ui.widgets.plugin_editor import PluginSliceEditor


class _Data(EMData):
    def restore_data(self, channel):
        pass


def _data():
    n = 500
    index = pd.date_range("2025-01-01", periods=n, freq="s", tz="UTC")
    position = np.arange(n, dtype=float)
    values = position * 0.002 + 6 * np.clip((position - 190) / 20, 0, 1)
    values[260:265] = np.nan
    data = _Data("/tmp/step-suffix.mat")
    data.name = "step"
    data.dt = pd.Timedelta(seconds=1)
    data.datetime_index = index
    data.start_time, data.end_time = index[0], index[-1]
    data.kp_data = None
    data.data = xr.Dataset({"Ex": (("time",), values)}, coords={"time": index})
    return data, values


def test_baseline_block_uses_region_once_and_corrects_through_end(qtbot, monkeypatch):
    manager = DataManager()
    data, original = _data()
    data_id = manager.add(data)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(data, data_id)
    pipeline = widget._find_pipeline(data_id, "Ex")
    layer = pipeline.layers[0]
    block = PipelineBlock(tool_id="remove_step_baseline", start=190, end=211, status="empty")
    layer.blocks.append(block)
    plugin = registry.get_plugin("remove_step_baseline")
    process_calls = []
    original_process = plugin.process

    def process(*args):
        process_calls.append(1)
        return original_process(*args)

    monkeypatch.setattr(plugin, "process", process)

    widget._open_interactive_editor(pipeline, layer, block)
    dialog = widget._editor_dialogs[-1]
    wrapper = dialog.findChild(PluginSliceEditor)
    editor = wrapper._editor.editor
    assert editor.pipeline_region == (190, 210)
    assert editor.region_frame.isHidden()
    assert editor.signal_plot_widget.double_click_callback is None
    assert len(wrapper._host.block.series) == len(original)
    editor._confirm_and_close()
    assert "先应用校正" in editor.status_label.text()
    assert block.status == "empty"

    editor.context_spin.setValue(60)
    editor._preview_fit()
    qtbot.waitUntil(lambda: editor.last_result is not None, timeout=10000)
    preview = editor.last_result.corrected.copy()
    editor._apply_correction()
    qtbot.waitUntil(lambda: editor._pipeline_applied, timeout=10000)
    editor.context_spin.setValue(61)
    assert not editor._pipeline_applied
    editor._confirm_and_close()
    assert "先应用校正" in editor.status_label.text()
    editor.context_spin.setValue(60)
    editor._apply_correction()
    qtbot.waitUntil(lambda: editor._pipeline_applied, timeout=10000)
    editor._confirm_and_close()
    qtbot.waitUntil(lambda: block.status == "ready", timeout=10000)
    assert process_calls == []
    output = widget._compose_channel_preview(data_id, "Ex", update_status=False)
    np.testing.assert_allclose(output, preview, equal_nan=True)
    np.testing.assert_allclose(output[:190], original[:190], equal_nan=True)
    delta = output[211:] - original[211:]
    finite = np.isfinite(delta)
    assert abs(np.median(delta[finite])) > 5
    np.testing.assert_allclose(delta[finite], delta[finite][0], atol=1e-10)
    assert block.params["region"] == [190, 210]
    assert block.params["effect_to_end"] is True
    assert block.params["operations"] == []

    recipe = recipe_from_pipeline(pipeline, data.datetime_index, "step")
    result = BatchRunner({"remove_step_baseline": StepBaselinePlugin()}).run_target(
        recipe, BatchTarget(data_id, "Ex", data.get_channel_series("Ex"))
    )
    assert result.status == "ok", result.error
    np.testing.assert_allclose(result.series.to_numpy(), output, equal_nan=True)

    earlier_index = pd.date_range(data.datetime_index[0] - pd.Timedelta(seconds=10),
                                  periods=len(original) + 10, freq="s", tz="UTC")
    earlier_values = np.r_[np.arange(-10, 0) * 0.002, original]
    earlier = pd.Series(earlier_values, index=earlier_index, name="Ex")
    shifted = BatchRunner({"remove_step_baseline": StepBaselinePlugin()}).run_target(
        recipe, BatchTarget(2, "Ex", earlier)
    )
    assert shifted.status == "ok", shifted.error
    np.testing.assert_allclose(shifted.series.to_numpy()[:10], earlier_values[:10])
    np.testing.assert_allclose(shifted.series.to_numpy()[10:], output, equal_nan=True)

    incomplete = pd.Series(original[:206], index=data.datetime_index[:206], name="Ex")
    skipped = BatchRunner({"remove_step_baseline": StepBaselinePlugin()}).run_target(
        recipe, BatchTarget(3, "Ex", incomplete)
    )
    assert skipped.status == "ok" and len(skipped.skipped_steps) == 1
    np.testing.assert_allclose(skipped.series.to_numpy(), incomplete.to_numpy(), equal_nan=True)

    monkeypatch.setattr(QMessageBox, "information", lambda *_args: None)
    widget._commit_preview()
    np.testing.assert_allclose(data.get_channel_series("Ex").to_numpy(), output, equal_nan=True)


def test_baseline_block_recalculates_suffix_when_moved(qtbot):
    manager = DataManager()
    data, original = _data()
    data_id = manager.add(data)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(data, data_id)
    pipeline = widget._find_pipeline(data_id, "Ex")
    block = PipelineBlock(tool_id="remove_step_baseline", start=190, end=211,
                          params={"context_points": 60, "effect_to_end": True}, status="configured")
    pipeline.layers[0].blocks.append(block)
    widget._refresh_all_pipeline_statuses()
    qtbot.waitUntil(lambda: block.status == "ready", timeout=10000)
    block.start, block.end = 185, 216
    widget._on_block_geometry_changed(block.block_id)
    qtbot.waitUntil(lambda: block.status == "ready" and block.params["region"] == [185, 215], timeout=10000)
    output = widget._compose_channel_preview(data_id, "Ex", update_status=False)
    np.testing.assert_allclose(output[:185], original[:185], equal_nan=True)
    delta = output[216:] - original[216:]
    finite = np.isfinite(delta)
    assert abs(np.median(delta[finite])) > 5
    np.testing.assert_allclose(delta[finite], delta[finite][0], atol=1e-10)


def test_unconfirmed_or_legacy_baseline_never_autoruns(qtbot):
    manager = DataManager()
    data, original = _data()
    data_id = manager.add(data)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(data, data_id)
    pipeline = widget._find_pipeline(data_id, "Ex")
    block = PipelineBlock(
        tool_id="remove_step_baseline", start=190, end=211, status="ready",
        params={"operations": [{"delta_runs": [[0, [100.0]]]}]},
        input_slice=original[190:211].copy(), output_slice=np.zeros(21),
    )
    pipeline.layers[0].blocks.append(block)
    widget._on_block_geometry_changed(block.block_id)
    qtbot.wait(50)
    assert block.status == "empty"
    assert block.block_id not in widget._plugin_tasks
    result = widget._compose_channel_preview(data_id, "Ex", update_status=False)
    np.testing.assert_allclose(result, original, equal_nan=True)


def test_linked_baseline_blocks_recalculate_each_channel_to_end(qtbot):
    manager = DataManager()
    data, original = _data()
    data.data["Ey"] = (("time",), original * 2)
    data_id = manager.add(data)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(data, data_id)
    pipeline = widget._find_pipeline(data_id, "Ex")
    params = {"context_points": 60, "effect_to_end": True, "region": [190, 210]}
    block = PipelineBlock(tool_id="remove_step_baseline", start=190, end=211,
                          params=params, status="configured")
    pipeline.layers[0].blocks.append(block)
    widget._set_block_linked(block.block_id, params, linked=True)
    members = widget._group_members(block.settings_group_id)
    assert len(members) == 2
    widget._refresh_all_pipeline_statuses()
    qtbot.waitUntil(lambda: all(member.status == "ready" for _, _, member in members), timeout=10000)
    for target, _, member in members:
        source = data.get_channel_series(target.channel).to_numpy()
        corrected = widget._compose_channel_preview(data_id, target.channel, update_status=False)
        delta = corrected[211:] - source[211:]
        finite = np.isfinite(delta)
        np.testing.assert_allclose(delta[finite], delta[finite][0], atol=1e-10)
        assert abs(np.median(delta[finite])) > (5 if target.channel == "Ex" else 10)
        assert member.params["region"] == [190, 210]


def test_changed_block_range_rejects_old_editor_result(qtbot, monkeypatch):
    manager = DataManager()
    data, _original = _data()
    data_id = manager.add(data)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(data, data_id)
    pipeline = widget._find_pipeline(data_id, "Ex")
    layer = pipeline.layers[0]
    block = PipelineBlock(tool_id="remove_step_baseline", start=190, end=211, status="empty")
    layer.blocks.append(block)
    widget._open_interactive_editor(pipeline, layer, block)
    editor = widget._editor_dialogs[-1].findChild(PluginSliceEditor)._editor.editor
    editor._apply_correction()
    qtbot.waitUntil(lambda: editor._pipeline_applied, timeout=10000)
    block.start = 185
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args))
    editor._confirm_and_close()
    assert messages and "Block 范围已改变" in messages[0][2]
    assert block.params == {}
    assert block.input_slice is None and block.output_slice is None


def test_baseline_requires_full_channel_in_multi_dataset_pipeline(qtbot, monkeypatch):
    manager = DataManager()
    full, _ = _data()
    short, _ = _data()
    short.data = short.data.isel(time=slice(0, 400))
    full_id, short_id = manager.add(full), manager.add(short)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.add_channels([(full_id, "Ex"), (short_id, "Ex")])
    layer = widget._find_pipeline(full_id, "Ex").layers[0]
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args))
    widget._add_block_with_tool(full_id, "Ex", layer.layer_id, 190, "remove_step_baseline")
    assert not layer.blocks
    assert messages and "公共时间范围" in messages[0][2]

    standalone = DataProcessPipelineWidget(manager)
    qtbot.addWidget(standalone)
    standalone.init_data(full, full_id)
    source_layer = standalone._find_pipeline(full_id, "Ex").layers[0]
    source_layer.blocks.append(PipelineBlock(tool_id="remove_step_baseline", start=190, end=211))
    added, duplicated, failed = standalone.add_channels([(short_id, "Ex")])
    assert not added and not duplicated and failed == [(short_id, "Ex")]


def test_downward_step_is_corrected_through_later_missing_values():
    data, source = _data()
    series = pd.Series(-source, index=data.datetime_index, name="Ex")
    result, _ = run_plugin_process(
        StepBaselinePlugin(), ChannelBlock("Ex", series),
        {"region": [190, 210], "context_points": 60, "effect_to_end": True},
    )
    corrected = result.series.to_numpy()
    np.testing.assert_array_equal(np.isnan(corrected), np.isnan(source))
    np.testing.assert_allclose(corrected[:190], -source[:190])
    delta = corrected[211:] + source[211:]
    finite = np.isfinite(delta)
    assert np.median(delta[finite]) > 5
    np.testing.assert_allclose(delta[finite], delta[finite][0], atol=1e-10)
