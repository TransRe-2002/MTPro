from __future__ import annotations

from threading import Event

import numpy as np
import pandas as pd
import xarray as xr
from PySide6.QtWidgets import QMessageBox

from base.batch_runner import BatchRunner, BatchTarget
from base.data_manager import DataManager
from core.algorithms.secular_variation import remove_secular_variation_series
from core.em_data import EMData
from core.pipeline_model import PipelineBlock
from core.recipe import Recipe, RecipeStep, recipe_from_pipeline
from plugins.remove_secular_variation.plugin import SecularVariationPlugin
from ui.data_process_pipeline import DataProcessPipelineWidget, LEGACY_SV_RANGE_ERROR
from ui.widgets.plugin_editor import PluginSliceEditor


class _Data(EMData):
    def restore_data(self, channel):
        pass


def _data():
    n = 1200
    index = pd.date_range("2025-01-01", periods=n, freq="s", tz="UTC")
    x = np.arange(n, dtype=float)
    values = 7.0 + x * 0.001 + 0.1 * np.sin(x / 5)
    values[600:620] = np.nan
    data = _Data("/tmp/sv-full.mat")
    data.name = "sv"
    data.dt = pd.Timedelta(seconds=1)
    data.datetime_index = index
    data.start_time, data.end_time = index[0], index[-1]
    data.kp_data = None
    data.data = xr.Dataset({"Ex": (("time",), values)}, coords={"time": index})
    return data, values


def test_sv_pipeline_preview_commit_and_batch_use_full_channel(qtbot, monkeypatch):
    manager = DataManager()
    data, original = _data()
    data_id = manager.add(data)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(data, data_id)
    pipeline = widget._find_pipeline(data_id, "Ex")
    layer = pipeline.layers[0]
    widget._add_block_with_tool(data_id, "Ex", layer.layer_id, 200, "remove_secular_variation")
    block = layer.blocks[0]
    assert (block.start, block.end) == (0, len(original))
    assert widget._lane_widgets[0]._tracks[0]._hit_test(0)[1] == "locked"

    widget._open_interactive_editor(pipeline, layer, block)
    wrapper = widget._editor_dialogs[-1].findChild(PluginSliceEditor)
    editor = wrapper._editor
    assert len(editor._source) == len(original)
    assert editor._x_values[0] == data.datetime_index[0].timestamp()
    editor.window_spin.setValue(101)
    editor.spacing_spin.setValue(50)
    editor._start_preview()
    qtbot.waitUntil(lambda: editor._preview is not None, timeout=10000)
    assert editor._preview.status == "applied"
    editor._apply()
    qtbot.waitUntil(lambda: block.status == "ready", timeout=10000)
    output = widget._compose_channel_preview(data_id, "Ex", update_status=False)
    expected = remove_secular_variation_series(
        data.get_channel_series("Ex"), smoothing_window_points=101, anchor_spacing_points=50,
    ).cleaned.to_numpy()
    np.testing.assert_allclose(output, expected, equal_nan=True)
    np.testing.assert_array_equal(np.isnan(output), np.isnan(original))
    assert abs(output[200] - output[199]) < 0.1
    assert abs(output[400] - output[399]) < 0.1

    recipe = recipe_from_pipeline(pipeline, data.datetime_index, "sv")
    assert recipe.steps[0].time_range is None
    batch = BatchRunner({"remove_secular_variation": SecularVariationPlugin()}).run_target(
        recipe, BatchTarget(data_id, "Ex", data.get_channel_series("Ex"))
    )
    assert batch.status == "ok", batch.error
    np.testing.assert_allclose(batch.series.to_numpy(), output, equal_nan=True)
    explicit_full = Recipe("full", (RecipeStep(
        "remove_secular_variation", dict(block.params),
        (data.datetime_index[0].isoformat(), data.datetime_index[-1].isoformat()),
    ),))
    full_batch = BatchRunner({"remove_secular_variation": SecularVariationPlugin()}).run_target(
        explicit_full, BatchTarget(data_id, "Ex", data.get_channel_series("Ex"))
    )
    assert full_batch.status == "ok", full_batch.error
    np.testing.assert_allclose(full_batch.series.to_numpy(), output, equal_nan=True)
    monkeypatch.setattr(QMessageBox, "information", lambda *_args: None)
    widget._commit_preview()
    np.testing.assert_allclose(data.get_channel_series("Ex").to_numpy(), output, equal_nan=True)


def test_old_local_sv_block_and_recipe_are_not_silently_replayed(qtbot):
    manager = DataManager()
    data, original = _data()
    data_id = manager.add(data)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(data, data_id)
    pipeline = widget._find_pipeline(data_id, "Ex")
    layer = pipeline.layers[0]
    block = PipelineBlock(
        tool_id="remove_secular_variation", start=200, end=400, status="ready",
        params={"smoothing_window_points": 101},
        input_slice=original[200:400].copy(), output_slice=np.zeros(200),
    )
    layer.blocks.append(block)
    widget._refresh_all_pipeline_statuses()
    assert block.status == "error" and block.last_summary == LEGACY_SV_RANGE_ERROR
    assert (block.start, block.end) == (200, 400)
    np.testing.assert_allclose(
        widget._compose_channel_preview(data_id, "Ex", update_status=False), original, equal_nan=True,
    )

    old = Recipe("old", (RecipeStep(
        "remove_secular_variation", {},
        (data.datetime_index[200].isoformat(), data.datetime_index[399].isoformat()),
    ),))
    result = BatchRunner({"remove_secular_variation": SecularVariationPlugin()}).run_target(
        old, BatchTarget(data_id, "Ex", data.get_channel_series("Ex")),
    )
    assert result.status == "error" and "旧版局部" in result.error
    np.testing.assert_allclose(result.series.to_numpy(), original, equal_nan=True)


def test_sv_requires_full_target_time_range(qtbot, monkeypatch):
    manager = DataManager()
    full, _ = _data()
    short, _ = _data()
    short.data = short.data.isel(time=slice(100, 1000))
    full_id, short_id = manager.add(full), manager.add(short)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.add_channels([(full_id, "Ex"), (short_id, "Ex")])
    layer = widget._find_pipeline(full_id, "Ex").layers[0]
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args))
    widget._add_block_with_tool(full_id, "Ex", layer.layer_id, 200, "remove_secular_variation")
    assert not layer.blocks
    assert messages and "公共时间范围" in messages[0][2]

    single = DataProcessPipelineWidget(manager)
    qtbot.addWidget(single)
    single.init_data(full, full_id)
    layer = single._find_pipeline(full_id, "Ex").layers[0]
    single._add_block_with_tool(full_id, "Ex", layer.layer_id, 0, "remove_secular_variation")
    added, duplicated, failed = single.add_channels([(short_id, "Ex")])
    assert not added and not duplicated and failed == [(short_id, "Ex")]


def test_sv_stale_preview_and_failure_reenable_preview(qtbot, monkeypatch):
    import plugins.remove_secular_variation.editor as editor_module

    manager = DataManager()
    data, _ = _data()
    data_id = manager.add(data)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(data, data_id)
    pipeline = widget._find_pipeline(data_id, "Ex")
    layer = pipeline.layers[0]
    widget._add_block_with_tool(data_id, "Ex", layer.layer_id, 0, "remove_secular_variation")
    widget._open_interactive_editor(pipeline, layer, layer.blocks[0])
    editor = widget._editor_dialogs[-1].findChild(PluginSliceEditor)._editor
    started, release = Event(), Event()
    real_compute = editor_module.remove_secular_variation_series

    def slow(source, **params):
        started.set()
        assert release.wait(5)
        return real_compute(source, **params)

    monkeypatch.setattr(editor_module, "remove_secular_variation_series", slow)
    editor._start_preview()
    try:
        qtbot.waitUntil(started.is_set, timeout=5000)
        editor.window_spin.setValue(101)
    finally:
        release.set()
    qtbot.waitUntil(editor.preview_button.isEnabled, timeout=5000)
    assert editor._preview is None and not editor.apply_button.isEnabled()
    assert "参数已变化" in editor.status_label.text()

    def fail(_source, **_params):
        raise ValueError("synthetic failure")

    monkeypatch.setattr(editor_module, "remove_secular_variation_series", fail)
    editor._start_preview()
    qtbot.waitUntil(editor.preview_button.isEnabled, timeout=5000)
    assert not editor.apply_button.isEnabled()
    assert "synthetic failure" in editor.status_label.text()
