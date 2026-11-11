from __future__ import annotations

from threading import Event

import numpy as np
import pandas as pd

from plugin_api import TransformResult
from plugins.vmd_denoise.plugin import VMDDenoisePlugin
from ui.widgets.plugin_editor import PluginSliceEditor


def _editor(qtbot, monkeypatch, process):
    plugin = VMDDenoisePlugin()
    monkeypatch.setattr(plugin, "process", process)
    index = pd.date_range("2025-01-01", periods=128, freq="s", tz="UTC")
    series = pd.Series(np.sin(np.arange(128) / 10), index=index, name="Ex")
    wrapper = PluginSliceEditor(plugin, 0, "Ex")
    qtbot.addWidget(wrapper)
    wrapper.load_series("Ex", series)
    return wrapper, wrapper._editor, series


def test_vmd_preview_shows_input_and_applies_same_result_once(qtbot, monkeypatch):
    calls = []

    def process(block, params, _ctx):
        calls.append(dict(params))
        return TransformResult(block.series * 0.5, {
            "processed_segments": [{"center_frequencies_hz": [0.01, 0.05]}],
        })

    wrapper, editor, series = _editor(qtbot, monkeypatch, process)
    assert editor.input_curve.getData()[1] is not None
    assert editor.output_curve.getData()[1] is None
    assert not editor.apply_button.isEnabled()
    editor.keep_modes_edit.setText("0")
    editor.preview_button.click()
    qtbot.waitUntil(lambda: editor.apply_button.isEnabled(), timeout=5000)
    np.testing.assert_allclose(editor.output_curve.getData()[1], series.to_numpy() * 0.5)
    np.testing.assert_allclose(editor.difference_curve.getData()[1], series.to_numpy() * 0.5)
    assert "0.01" in editor.status_label.text()
    received = []
    wrapper.params_applied.connect(lambda params, _broadcast: received.append(params))
    wrapper.result_signal.connect(lambda _channel, values: received.append(values))
    editor.apply_button.click()
    assert len(calls) == 1
    assert received[0]["keep_modes"] == "0"
    np.testing.assert_allclose(received[1], series.to_numpy() * 0.5)


def test_vmd_discards_outdated_preview(qtbot, monkeypatch):
    started, release = Event(), Event()

    def process(block, _params, _ctx):
        started.set()
        assert release.wait(5)
        return TransformResult(block.series.copy(), {
            "processed_segments": [{"center_frequencies_hz": [0.01]}],
        })

    _wrapper, editor, _series = _editor(qtbot, monkeypatch, process)
    editor.preview_button.click()
    try:
        qtbot.waitUntil(started.is_set, timeout=5000)
        editor.alpha_spin.setValue(1234)
    finally:
        release.set()
    qtbot.waitUntil(lambda: not editor._busy, timeout=5000)
    assert editor.preview_button.isEnabled()
    assert not editor.apply_button.isEnabled()
    assert editor.output_curve.getData()[1] is None
    assert "参数已变化" in editor.status_label.text()


def test_vmd_preview_runs_real_algorithm_and_reports_invalid_modes(qtbot):
    index = pd.date_range("2025-01-01", periods=256, freq="s", tz="UTC")
    series = pd.Series(np.sin(np.arange(256) / 8), index=index, name="Ex")
    wrapper = PluginSliceEditor(VMDDenoisePlugin(), 0, "Ex")
    qtbot.addWidget(wrapper)
    wrapper.load_series("Ex", series)
    editor = wrapper._editor
    editor.mode_count_spin.setValue(2)
    editor.max_iter_spin.setValue(20)
    editor.keep_modes_edit.setText("0")
    editor.preview_button.click()
    qtbot.waitUntil(lambda: not editor._busy, timeout=10000)
    assert editor.apply_button.isEnabled(), editor.status_label.text()
    assert len(editor._preview_result.series) == len(series)
    assert len(editor._preview_result.diagnostics["processed_segments"]) == 1

    editor.keep_modes_edit.setText("9")
    editor.preview_button.click()
    qtbot.waitUntil(lambda: not editor._busy, timeout=10000)
    assert not editor.apply_button.isEnabled()
    assert "预览失败" in editor.status_label.text()


def test_vmd_preview_applies_to_pipeline_block(qtbot, monkeypatch):
    from core.pipeline_model import PipelineBlock
    from test_pipeline_block_interactions import _widget
    from ui.tools import registry

    _app, pipeline_widget, data_id = _widget(qtbot)
    pipeline = pipeline_widget._find_pipeline(data_id, "Ex")
    block = PipelineBlock(tool_id="vmd_denoise", start=0, end=12)
    pipeline.layers[0].blocks.append(block)
    plugin = registry.get_plugin("vmd_denoise")
    calls = []

    def process(source, params, _ctx):
        calls.append(dict(params))
        return TransformResult(source.series * 0.5, {
            "processed_segments": [{"center_frequencies_hz": [0.01]}],
        })

    monkeypatch.setattr(plugin, "process", process)
    pipeline_widget._open_interactive_editor(pipeline, pipeline.layers[0], block)
    dialog = pipeline_widget._editor_dialogs[-1]
    editor = dialog.findChild(PluginSliceEditor)._editor
    editor.mode_count_spin.setValue(2)
    editor.keep_modes_edit.setText("0")
    editor.preview_button.click()
    qtbot.waitUntil(lambda: editor.apply_button.isEnabled(), timeout=5000)
    editor.apply_button.click()
    assert len(calls) == 1
    assert block.params["mode_count"] == 2
    assert block.params["keep_modes"] == "0"
    assert block.status == "ready"
    np.testing.assert_allclose(block.output_slice, np.arange(12) * 0.5)
