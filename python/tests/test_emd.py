from __future__ import annotations

import json

import numpy as np
import pandas as pd

from base.plugin_runtime import run_plugin_process
from core.algorithms import emd as emd_module
from core.algorithms.emd import EMDConfig, emd_decompose, emd_denoise_series
from plugin_api import ChannelBlock
from plugins.emd_denoise.plugin import EMDDenoisePlugin


def _signal(size: int = 512) -> np.ndarray:
    samples = np.arange(size, dtype=np.float64)
    return (
        0.002 * samples
        + np.sin(2.0 * np.pi * 0.04 * samples)
        + 0.35 * np.sin(2.0 * np.pi * 0.20 * samples)
    )


def test_emd_imfs_and_residue_reconstruct_a_finite_signal():
    source = _signal()
    result = emd_decompose(
        source,
        EMDConfig(max_imfs=4, max_siftings=100, min_segment_points=4),
    )

    assert result.imfs.ndim == 2
    assert result.imfs.shape[1] == source.size
    assert result.imfs.shape[0] == len(result.siftings)
    assert result.imfs.shape[0] > 0
    np.testing.assert_allclose(result.imfs.sum(axis=0) + result.residue, source, atol=1.0e-10)
    if emd_module._HAS_RUST_EMD:
        assert result.used_rust


def test_emd_denoise_preserves_index_and_missing_runs():
    values = _signal()
    values[180:185] = np.nan
    index = pd.date_range("2025-01-01", periods=values.size, freq="s", tz="Asia/Shanghai")
    source = pd.Series(values, index=index, name="Ex")

    result = emd_denoise_series(
        source,
        EMDConfig(max_imfs=4, max_siftings=100, min_segment_points=32),
        keep_imfs="all",
        keep_residue=True,
    )

    assert result.cleaned.index.equals(index)
    assert np.isnan(result.cleaned.iloc[180:185]).all()
    assert len(result.processed_segments) == 2
    assert all(item.reconstruction_residual_rms < 1.0e-10 for item in result.processed_segments)


def test_emd_plugin_produces_json_safe_diagnostics():
    values = _signal()
    values[200:205] = np.nan
    index = pd.date_range("2025-01-01", periods=values.size, freq="s", tz="Asia/Shanghai")
    block = ChannelBlock("Ex", pd.Series(values, index=index, name="Ex"))

    result, params = run_plugin_process(
        EMDDenoisePlugin(),
        block,
        {"max_imfs": 4, "max_siftings": 100, "keep_imfs": "all"},
    )

    assert result.series.index.equals(index)
    assert np.isnan(result.series.iloc[200:205]).all()
    assert params["keep_residue"] is True
    assert len(result.diagnostics["processed_segments"]) == 2
    json.dumps(result.diagnostics)
