from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from PySide6.QtWidgets import QMessageBox

from base.data_manager import DataManager
from core.em_data import EMData
from core.pipeline_model import PipelineBlock
from ui.data_process_pipeline import DataProcessPipelineWidget
from ui.widgets.plugin_editor import PluginSliceEditor


class _Data(EMData):
    def restore_data(self, channel):
        pass


def _setup(qtbot):
    n = 1200
    index = pd.date_range('2025-01-01', periods=n, freq='s', tz='UTC')
    values = np.sin(np.arange(n) / 40) + np.random.default_rng(42).normal(0, 0.3, n)
    data = _Data('/tmp/wavelet-local.mat')
    data.name = 'wavelet'
    data.dt = pd.Timedelta(seconds=1)
    data.datetime_index = index
    data.start_time, data.end_time = index[0], index[-1]
    data.kp_data = None
    data.data = xr.Dataset({'Ex': (('time',), values)}, coords={'time': index})
    manager = DataManager()
    data_id = manager.add(data)
    widget = DataProcessPipelineWidget(manager)
    qtbot.addWidget(widget)
    widget.init_data(data, data_id)
    pipeline = widget._find_pipeline(data_id, 'Ex')
    return widget, pipeline, data, values


def test_wavelet_editor_processes_only_block_and_preserves_neighbors(qtbot, monkeypatch):
    widget, pipeline, data, original = _setup(qtbot)
    layer = pipeline.layers[0]
    block = PipelineBlock(tool_id='remove_wavelet_denoise', start=400, end=600, status='empty')
    layer.blocks.append(block)
    widget._open_interactive_editor(pipeline, layer, block)
    wrapper = widget._editor_dialogs[-1].findChild(PluginSliceEditor)
    editor = wrapper._editor.editor
    assert len(wrapper._host.block.series) == 200
    assert editor._processing_region() == (0, 199)
    editor.mode_combo.setCurrentText('zero')
    editor.level_spin.setValue(2)
    editor._apply_correction()
    qtbot.waitUntil(lambda: editor.processing_status.text().startswith('已应用'), timeout=10000)
    editor._confirm_and_close()
    qtbot.waitUntil(lambda: block.status == 'ready', timeout=10000)
    output = widget._compose_channel_preview(pipeline.data_id, 'Ex', update_status=False)
    np.testing.assert_array_equal(output[:400], original[:400])
    np.testing.assert_array_equal(output[600:], original[600:])
    assert not np.allclose(output[400:600], original[400:600])
    monkeypatch.setattr(QMessageBox, 'information', lambda *_args: None)
    widget._commit_preview()
    np.testing.assert_allclose(data.get_channel_series('Ex').to_numpy(), output)


def test_moving_wavelet_block_discards_old_edits_but_preserves_settings(qtbot):
    widget, pipeline, _, original = _setup(qtbot)
    block = PipelineBlock(tool_id='remove_wavelet_denoise', start=400, end=600,
                          params={'wavelet': 'db6', 'region': [0, 199],
                                  'operations': [{'delta_runs': [[0, [3.0]]]}]},
                          status='ready', input_slice=original[400:600].copy(),
                          output_slice=original[400:600].copy() + 3)
    pipeline.layers[0].blocks.append(block)
    block.start, block.end = 450, 650
    widget._on_block_geometry_changed(block.block_id)
    qtbot.wait(50)
    assert block.status == 'empty'
    assert block.params['wavelet'] == 'db6'
    assert block.params['operations'] == [] and block.params['region'] == []
    assert block.input_slice is None and block.output_slice is None
    assert block.block_id not in widget._plugin_tasks
    np.testing.assert_array_equal(widget._compose_channel_preview(pipeline.data_id, 'Ex', update_status=False), original)


def test_new_wavelet_block_is_movable_and_resizable(qtbot):
    widget, pipeline, _, _ = _setup(qtbot)
    layer = pipeline.layers[0]
    widget._snap.enabled = False
    widget._add_block_with_tool(pipeline.data_id, 'Ex', layer.layer_id, 400, 'remove_wavelet_denoise')
    block = layer.blocks[0]
    assert (block.start, block.end) == (400, 600)
    track = widget._lane_widgets[0]._tracks[0]
    track.set_zoom(1, 1.0)
    assert track._hit_test(400) == (block, 'resize_left')
    assert track._hit_test(500) == (block, 'move')
    assert track._hit_test(600) == (block, 'resize_right')


def test_old_editor_result_is_rejected_if_block_range_changes(qtbot, monkeypatch):
    widget, pipeline, _, original = _setup(qtbot)
    layer = pipeline.layers[0]
    block = PipelineBlock(tool_id='remove_wavelet_denoise', start=400, end=600, status='empty')
    layer.blocks.append(block)
    widget._open_interactive_editor(pipeline, layer, block)
    wrapper = widget._editor_dialogs[-1].findChild(PluginSliceEditor)
    editor = wrapper._editor.editor
    editor.mode_combo.setCurrentText('zero')
    editor.level_spin.setValue(2)
    editor._apply_correction()
    qtbot.waitUntil(lambda: editor.processing_status.text().startswith('已应用'), timeout=10000)
    block.start, block.end = 450, 650
    widget._on_block_geometry_changed(block.block_id)
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args))
    editor._confirm_and_close()
    qtbot.waitUntil(lambda: bool(warnings), timeout=10000)
    assert block.status == 'empty'
    assert not block.params.get('operations')
    np.testing.assert_array_equal(widget._compose_channel_preview(pipeline.data_id, 'Ex', update_status=False), original)
