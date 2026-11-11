from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.pipeline_model import PreviewChannelSlice
from ui.tools import registry


@pytest.mark.parametrize("tool", [
    "remove_step_window",
    "remove_step_baseline",
    "remove_diurnal_sinusoid",
    "remove_wavelet_denoise",
])
def test_confirmed_editor_parameters_are_restored(tool, qtbot):
    index = pd.date_range("2025-01-01", periods=512, freq="s", tz="UTC")
    series = pd.Series(np.sin(np.arange(512) / 20), index=index, name="Ex")
    preview = PreviewChannelSlice("Ex", series)
    editor = registry.create_pipeline_editor(tool, preview, None, params={})
    qtbot.addWidget(editor)
    controls = editor._editor.editor

    if tool == "remove_step_window":
        controls.de_step_threshold.setText("23.5")
        controls.avg_window_edit.setText("37")
        controls.max_steps_edit.setText("4")
    elif tool == "remove_step_baseline":
        controls.transition_degree_spin.setValue(4)
        controls.context_spin.setValue(345)
        controls.loss_combo.setCurrentText("huber")
        controls.last_result = SimpleNamespace(region=(40, 80))
    elif tool == "remove_diurnal_sinusoid":
        controls.harmonics_combo.setCurrentIndex(2)
        controls.window_days_edit.setText("4.5")
        controls.min_valid_samples_edit.setText("64")
    else:
        controls.wavelet_combo.setCurrentText("sym6")
        controls.threshold_scale_spin.setValue(1.7)
        controls.last_result = SimpleNamespace(region=(40, 80))

    saved = []
    editor.params_applied.connect(lambda params, _broadcast: saved.append(params))
    controls.result_signal.emit("Ex", series.to_numpy())
    qtbot.waitUntil(lambda: bool(saved), timeout=10000)

    reopened = registry.create_pipeline_editor(tool, preview, None, params=saved[0])
    qtbot.addWidget(reopened)
    restored = reopened._editor.editor
    if tool == "remove_step_window":
        assert float(restored.de_step_threshold.text()) == 23.5
        assert int(restored.avg_window_edit.text()) == 37
        assert int(restored.max_steps_edit.text()) == 4
    elif tool == "remove_step_baseline":
        assert restored.transition_degree_spin.value() == 4
        assert restored.context_spin.value() == 345
        assert restored.loss_combo.currentText() == "huber"
        assert (restored.selected_indices[0], restored.selected_indices[-1]) == (40, 80)
    elif tool == "remove_diurnal_sinusoid":
        assert tuple(restored.harmonics_combo.currentData()) == (1, 2, 3)
        assert float(restored.window_days_edit.text()) == 4.5
        assert int(restored.min_valid_samples_edit.text()) == 64
    else:
        assert restored.wavelet_combo.currentText() == "sym6"
        assert restored.threshold_scale_spin.value() == pytest.approx(1.7)
        assert restored.scope_combo.currentText() == "全通道"
        assert restored._read_region() == (0, len(series) - 1)


@pytest.mark.parametrize("tool", ["remove_spike", "remove_region"])
def test_manual_removals_survive_reopening_and_confirming(tool, qtbot):
    index = pd.date_range("2025-01-01", periods=100, freq="s", tz="UTC")
    series = pd.Series(np.arange(100, dtype=float), index=index, name="Ex")
    preview = PreviewChannelSlice("Ex", series)
    params = (
        {"indices": [20, 21, 22]} if tool == "remove_spike" else
        {"time_ranges": [[index[20].isoformat(), index[22].isoformat()]]}
    )
    editor = registry.create_pipeline_editor(tool, preview, None, params=params)
    qtbot.addWidget(editor)
    controls = editor._editor.editor
    assert np.isnan(controls.plot_widget.y_data[20:23]).all()
    saved = []
    editor.params_applied.connect(lambda values, _broadcast: saved.append(values))
    controls.result_signal.emit("Ex", controls.plot_widget.confirm_and_exit())
    qtbot.waitUntil(lambda: bool(saved), timeout=10000)
    assert saved[0] == params
    controls.plot_widget.undo_last_deletion()
    assert np.array_equal(controls.plot_widget.y_data, series.to_numpy())
