from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd

from base.plugin_runtime import run_plugin_process
from core.algorithms.spike import AutoSpikeDetection, detect_auto_spikes
from plugin_api import ChannelBlock
from plugins.auto_spike_filter.editor import AutoSpikeFilter
from plugins.auto_spike_filter.plugin import AutoSpikeFilterPlugin
from ui.models.processing_tools import create_processing_tool_model


def _baseline(size: int = 301) -> np.ndarray:
    points = np.arange(size, dtype=np.float64)
    return 0.2 * np.sin(points / 23.0) + 0.001 * points


def test_detect_auto_spikes_masks_isolated_positive_and_negative_pulses():
    values = _baseline()
    values[80] += 100.0
    values[200] -= 100.0

    detection = detect_auto_spikes(values)

    assert set(detection.removed_indices) == {80, 200}
    assert detection.long_runs == ()
    assert detection.protected_runs == ()


def test_detect_auto_spikes_masks_a_pulse_of_up_to_five_points():
    values = _baseline()
    values[140:145] += 100.0

    detection = detect_auto_spikes(values, max_pulse_width=5)

    np.testing.assert_array_equal(detection.removed_indices, np.arange(140, 145))


def test_detect_auto_spikes_keeps_long_pulse_for_manual_or_other_processing():
    values = _baseline()
    values[140:146] += 100.0

    detection = detect_auto_spikes(values, max_pulse_width=5)

    assert detection.removed_indices.size == 0
    assert detection.long_runs == ((140, 145),)


def test_detect_auto_spikes_preserves_step_edges():
    values = _baseline()
    values[160:] += 100.0

    detection = detect_auto_spikes(values)

    assert detection.removed_indices.size == 0
    assert detection.protected_runs or detection.long_runs


def test_auto_spike_plugin_preserves_index_existing_nans_and_recipe_params():
    values = _baseline()
    values[30] = np.nan
    values[180] += 100.0
    index = pd.date_range("2025-01-01", periods=values.size, freq="s", tz="Asia/Shanghai")
    block = ChannelBlock("Ex", pd.Series(values, index=index, name="Ex"))

    result, params = run_plugin_process(AutoSpikeFilterPlugin(), block, {})

    assert result.series.index.equals(index)
    assert np.isnan(result.series.iloc[30])
    assert np.isnan(result.series.iloc[180])
    assert params == {
        "window_points": 101,
        "overlap_percent": 50.0,
        "threshold_sigma": 10.0,
        "max_pulse_width": 5,
        "remove_orange_points": False,
    }
    assert result.diagnostics["removed"] == 1


def test_auto_spike_plugin_can_remove_orange_long_run_candidates():
    values = _baseline()
    values[140:146] += 100.0
    index = pd.date_range(
        "2025-01-01",
        periods=values.size,
        freq="s",
        tz="Asia/Shanghai",
    )
    block = ChannelBlock("Ex", pd.Series(values, index=index, name="Ex"))

    result, params = run_plugin_process(
        AutoSpikeFilterPlugin(),
        block,
        {"max_pulse_width": 5, "remove_orange_points": True},
    )

    assert params["remove_orange_points"] is True
    assert result.series.iloc[140:146].isna().all()
    assert result.diagnostics["short_removed"] == 0
    assert result.diagnostics["orange_removed"] == 6
    assert result.diagnostics["removed"] == 6


def test_auto_spike_editor_checkbox_includes_orange_points_on_confirm(qtbot):
    editor = AutoSpikeFilter(0, "Ex")
    qtbot.addWidget(editor)
    editor._source = np.arange(6, dtype=np.float64)
    editor._detection = AutoSpikeDetection(
        candidate_indices=np.array([1, 2, 3], dtype=np.intp),
        candidate_scores=np.array([20.0, 18.0, 15.0]),
        removed_indices=np.array([1], dtype=np.intp),
        long_runs=((2, 3),),
        protected_runs=(),
        window_count=1,
    )
    editor._preview_params = editor._detection_params()
    results = []
    editor.result_signal.connect(lambda _channel, values: results.append(values))

    editor.remove_orange_checkbox.setChecked(True)
    editor._apply_detection()

    assert editor._params()["remove_orange_points"] is True
    assert "红点和橙点" in editor.apply_button.text()
    assert len(results) == 1
    assert np.isnan(results[0][1:4]).all()
    assert np.isfinite(results[0][[0, 4, 5]]).all()


def test_auto_spike_filter_is_listed_in_processing_tools():
    model = create_processing_tool_model()
    auto_category = model.item(2)
    tool_ids = {
        auto_category.child(row).data()
        for row in range(auto_category.rowCount())
    }

    assert "auto_spike_filter" in tool_ids
