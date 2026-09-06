from __future__ import annotations

import json

import numpy as np
import pandas as pd

from base.plugin_runtime import run_plugin_process
from core.algorithms import vmd as vmd_module
from core.algorithms.vmd import VMDConfig, vmd_decompose, vmd_denoise_series
from plugin_api import ChannelBlock, ChannelMeta
from plugins.vmd_denoise.plugin import VMDDenoisePlugin


def _two_mode_signal(size: int = 512) -> np.ndarray:
    samples = np.arange(size, dtype=np.float64)
    return (
        np.sin(2.0 * np.pi * 0.04 * samples)
        + 0.4 * np.sin(2.0 * np.pi * 0.20 * samples)
    )


def test_vmd_separates_two_stationary_components_in_frequency_order():
    result = vmd_decompose(
        _two_mode_signal(),
        VMDConfig(mode_count=2, alpha=2000.0, max_iter=800, min_segment_points=4),
    )

    assert result.modes.shape == (2, 512)
    assert result.iterations > 0
    assert result.center_frequencies[0] < result.center_frequencies[1]
    assert abs(result.center_frequencies[0] - 0.04) < 0.03
    assert abs(result.center_frequencies[1] - 0.20) < 0.04
    assert np.isfinite(result.modes).all()
    assert np.corrcoef(_two_mode_signal(), result.modes.sum(axis=0))[0, 1] > 0.99
    if vmd_module._HAS_RUST_VMD:
        assert result.used_rust


def test_vmd_denoise_preserves_index_nans_and_short_finite_runs():
    values = _two_mode_signal()
    values[180:185] = np.nan
    values[400:410] = np.nan
    index = pd.date_range("2025-01-01", periods=values.size, freq="s", tz="Asia/Shanghai")
    source = pd.Series(values, index=index, name="Ex")

    result = vmd_denoise_series(
        source,
        VMDConfig(mode_count=2, alpha=2000.0, max_iter=800, min_segment_points=32),
        keep_modes="0",
        sample_rate_hz=1.0,
    )

    assert result.cleaned.index.equals(index)
    assert np.isnan(result.cleaned.iloc[180:185]).all()
    assert np.isnan(result.cleaned.iloc[400:410]).all()
    assert result.selected_modes == (0,)
    assert len(result.processed_segments) == 3
    assert all(len(item.center_frequencies_hz) == 2 for item in result.processed_segments)
    assert all(item.reconstruction_residual_rms < 0.2 for item in result.processed_segments)


def test_vmd_plugin_returns_json_diagnostics_and_keeps_series_contract():
    values = _two_mode_signal()
    values[200:205] = np.nan
    index = pd.date_range("2025-01-01", periods=values.size, freq="s", tz="Asia/Shanghai")
    block = ChannelBlock(
        "Ex",
        pd.Series(values, index=index, name="Ex"),
        ChannelMeta(sample_rate=1.0),
    )

    result, params = run_plugin_process(
        VMDDenoisePlugin(),
        block,
        {
            "mode_count": 2,
            "alpha": 2000.0,
            "max_iter": 800,
            "keep_modes": "0",
        },
    )

    assert result.series.index.equals(index)
    assert np.isnan(result.series.iloc[200:205]).all()
    assert params["keep_modes"] == "0"
    assert result.diagnostics["selected_modes"] == [0]
    assert len(result.diagnostics["processed_segments"]) == 2
    json.dumps(result.diagnostics)
