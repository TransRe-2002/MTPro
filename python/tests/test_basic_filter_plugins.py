from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from base.plugin_runtime import run_plugin_process
from core.algorithms.filters import (
    detrend_series,
    iir_filter_series,
    median_filter_series,
    morphological_filter_series,
)
from plugin_api import ChannelBlock
from plugins.detrend.plugin import DetrendPlugin
from plugins.iir_filter.plugin import IIRFilterPlugin
from plugins.median_filter.plugin import MedianFilterPlugin
from plugins.morphological_filter.plugin import MorphologicalFilterPlugin


def _series(values: np.ndarray) -> pd.Series:
    return pd.Series(
        values,
        index=pd.date_range("2025-01-01", periods=values.size, freq="s", tz="Asia/Shanghai"),
        name="Ex",
    )


def test_detrend_removes_linear_slope_without_crossing_nan_gap():
    samples = np.arange(120, dtype=np.float64)
    values = 2.5 + 0.05 * samples + 0.2 * np.sin(samples / 7.0)
    values[50:55] = np.nan

    result = detrend_series(_series(values), "linear", min_segment_points=4)

    assert np.isnan(result.cleaned.iloc[50:55]).all()
    left_slope = np.polyfit(np.arange(50), result.cleaned.iloc[:50], 1)[0]
    right_slope = np.polyfit(np.arange(65), result.cleaned.iloc[55:], 1)[0]
    assert abs(left_slope) < 0.01
    assert abs(right_slope) < 0.01


def test_median_filter_suppresses_single_pulse_and_preserves_gaps():
    values = np.sin(np.arange(101, dtype=np.float64) / 12.0)
    values[40] += 100.0
    values[70:73] = np.nan

    result = median_filter_series(_series(values), window_points=5)

    assert abs(result.cleaned.iloc[40] - np.sin(40.0 / 12.0)) < 0.1
    assert np.isnan(result.cleaned.iloc[70:73]).all()


def test_morphological_open_close_suppresses_short_positive_and_negative_excursions():
    values = 0.02 * np.arange(161, dtype=np.float64)
    values[50:53] += 20.0
    values[110:113] -= 20.0

    result = morphological_filter_series(_series(values), window_points=9, operation="open_close")

    assert abs(result.cleaned.iloc[51] - 0.02 * 51) < 0.25
    assert abs(result.cleaned.iloc[111] - 0.02 * 111) < 0.25


def test_iir_lowpass_attenuates_high_frequency_component_without_crossing_gaps():
    samples = np.arange(512, dtype=np.float64)
    low = np.sin(2.0 * np.pi * 0.04 * samples)
    high = 0.8 * np.sin(2.0 * np.pi * 0.32 * samples)
    values = low + high
    values[250:254] = np.nan

    result = iir_filter_series(
        _series(values),
        filter_kind="lowpass",
        high_fraction=0.16,
        min_segment_points=32,
    )

    finite = np.isfinite(values)
    assert np.corrcoef(result.cleaned[finite], low[finite])[0, 1] > 0.98
    assert np.isnan(result.cleaned.iloc[250:254]).all()


@pytest.mark.parametrize(
    ("plugin", "params"),
    [
        (DetrendPlugin(), {"kind": "linear"}),
        (MedianFilterPlugin(), {"window_points": 5, "min_segment_points": 5}),
        (MorphologicalFilterPlugin(), {"window_points": 9, "min_segment_points": 9}),
        (IIRFilterPlugin(), {"filter_kind": "lowpass", "high_fraction": 0.2}),
    ],
)
def test_basic_filter_plugins_keep_the_series_contract_and_diagnostics_json(plugin, params):
    values = np.sin(np.arange(100, dtype=np.float64) / 9.0)
    values[30:33] = np.nan
    block = ChannelBlock("Ex", _series(values))

    result, completed = run_plugin_process(plugin, block, params)

    assert result.series.index.equals(block.series.index)
    assert np.isnan(result.series.iloc[30:33]).all()
    assert completed
    json.dumps(result.diagnostics)
