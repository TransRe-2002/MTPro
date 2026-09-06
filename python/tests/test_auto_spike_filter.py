from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication

from base.plugin_runtime import run_plugin_process
from core.algorithms.spike import detect_auto_spikes
from plugin_api import ChannelBlock
from plugins.auto_spike_filter.editor import AutoSpikeFilter
from plugins.auto_spike_filter.plugin import AutoSpikeFilterPlugin
from ui.data_process_shared import create_classic_tool_model
from ui.plugin_adapter import create_classic_plugin_widget
from ui.tools import registry


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
    }
    assert result.diagnostics["removed"] == 1


def test_auto_spike_classic_widget_previews_then_applies(qtbot):
    QApplication.instance() or QApplication([])
    widget = create_classic_plugin_widget(AutoSpikeFilterPlugin(), 7, "Ex")
    assert isinstance(widget, AutoSpikeFilter)
    assert registry.get("auto_spike_filter").kind == "interactive"
    qtbot.addWidget(widget)

    values = _baseline()
    values[180] += 100.0
    series = pd.Series(
        values,
        index=pd.date_range("2025-01-01", periods=values.size, freq="s", tz="Asia/Shanghai"),
        name="Ex",
    )
    emitted: list[np.ndarray] = []
    widget.result_submitted.connect(lambda _data_id, _channel, result: emitted.append(result))
    widget.load_series("Ex", series)

    qtbot.waitUntil(widget.apply_button.isEnabled, timeout=5000)
    widget._apply_detection()

    assert len(emitted) == 1
    assert np.isnan(emitted[0][180])


def test_auto_spike_filter_is_listed_in_classic_auto_tools():
    model = create_classic_tool_model()
    auto_category = model.item(1)
    tool_ids = {
        auto_category.child(row).data()
        for row in range(auto_category.rowCount())
    }

    assert "auto_spike_filter" in tool_ids
