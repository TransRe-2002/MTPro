from __future__ import annotations

import json

import numpy as np
import pandas as pd

from base.plugin_runtime import run_plugin_process
from core.algorithms.svd import svd_denoise_series
from plugin_api import ChannelBlock
from plugins.svd_denoise.plugin import SVDDenoisePlugin


def _series(values: np.ndarray) -> pd.Series:
    return pd.Series(
        values,
        index=pd.date_range("2025-01-01", periods=values.size, freq="s", tz="Asia/Shanghai"),
        name="Ex",
    )


def test_hankel_svd_low_rank_reconstruction_reduces_white_noise():
    samples = np.arange(512, dtype=np.float64)
    clean = np.sin(2.0 * np.pi * 0.04 * samples) + 0.4 * np.sin(2.0 * np.pi * 0.11 * samples)
    noisy = clean + np.random.default_rng(5).normal(scale=0.35, size=clean.size)

    result = svd_denoise_series(_series(noisy), embedding_points=80, rank=4, min_segment_points=128)

    assert len(result.processed_segments) == 1
    assert np.mean(np.square(result.cleaned - clean)) < np.mean(np.square(noisy - clean))
    assert result.processed_segments[0].rank == 4


def test_hankel_svd_plugin_preserves_nan_runs_and_serializes_diagnostics():
    values = np.sin(np.arange(256, dtype=np.float64) / 13.0)
    values[120:125] = np.nan
    block = ChannelBlock("Ex", _series(values))

    result, params = run_plugin_process(
        SVDDenoisePlugin(),
        block,
        {"embedding_points": 48, "rank": 3, "min_segment_points": 64},
    )

    assert result.series.index.equals(block.series.index)
    assert np.isnan(result.series.iloc[120:125]).all()
    assert params["rank"] == 3
    json.dumps(result.diagnostics)
