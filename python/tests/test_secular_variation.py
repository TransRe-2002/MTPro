from __future__ import annotations

import json

import numpy as np
import pandas as pd

from base.plugin_runtime import run_plugin_process
from core.algorithms.secular_variation import remove_secular_variation_series
from plugin_api import ChannelBlock
from plugins.remove_secular_variation.plugin import SecularVariationPlugin


def _series(values: np.ndarray) -> pd.Series:
    return pd.Series(
        values,
        index=pd.date_range("2025-01-01", periods=values.size, freq="h", tz="Asia/Shanghai"),
        name="Hx",
    )


def test_remove_secular_variation_recovers_smooth_baseline_and_keeps_missing_samples():
    samples = np.arange(2_000, dtype=np.float64)
    baseline = 4.0 + 0.002 * samples + 0.7 * np.sin(2.0 * np.pi * samples / 3_000.0)
    short_period_signal = 0.25 * np.sin(2.0 * np.pi * samples / 9.0)
    values = baseline + short_period_signal
    values[800:850] = np.nan

    result = remove_secular_variation_series(
        _series(values),
        smoothing_window_points=121,
        anchor_spacing_points=61,
        min_valid_fraction=0.25,
    )

    finite = np.isfinite(values)
    interior = finite & (samples >= 200) & (samples < 1_800)
    estimate = result.estimated_baseline.to_numpy(dtype=np.float64)
    assert result.status == "applied"
    assert result.valid_anchor_count >= 2
    # TS.removeSV uses truncated local windows at both ends, so assess the
    # smooth-background estimate away from the expected edge effect.
    assert np.sqrt(np.mean(np.square(estimate[interior] - baseline[interior]))) < 0.12
    assert np.corrcoef(result.cleaned.to_numpy()[interior], short_period_signal[interior])[0, 1] > 0.95
    assert np.isnan(result.cleaned.iloc[800:850]).all()


def test_remove_secular_variation_does_not_change_data_without_two_valid_anchors():
    values = np.full(10, np.nan, dtype=np.float64)
    values[1] = 3.0

    result = remove_secular_variation_series(
        _series(values),
        smoothing_window_points=3,
        anchor_spacing_points=10,
    )

    assert result.status == "insufficient_anchor_estimates"
    np.testing.assert_allclose(result.cleaned.to_numpy(), values, equal_nan=True)
    assert result.estimated_baseline.isna().all()


def test_secular_variation_plugin_uses_json_diagnostics_and_preserves_index():
    samples = np.arange(512, dtype=np.float64)
    values = 1.0 + 0.01 * samples + 0.1 * np.sin(samples)
    values[240:245] = np.nan
    block = ChannelBlock("Hx", _series(values))

    result, params = run_plugin_process(
        SecularVariationPlugin(),
        block,
        {
            "smoothing_window_points": 61,
            "anchor_spacing_points": 31,
            "min_valid_fraction": 0.25,
        },
    )

    assert result.series.index.equals(block.series.index)
    assert np.isnan(result.series.iloc[240:245]).all()
    assert result.diagnostics["status"] == "applied"
    assert params["anchor_spacing_points"] == 31
    json.dumps(result.diagnostics)
