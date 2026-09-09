from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

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


def test_chunked_hankel_svd_matches_full_matrix_reference():
    values = (
        np.sin(np.arange(96, dtype=np.float64) / 7.0)
        + 0.15 * np.cos(np.arange(96, dtype=np.float64) / 2.0)
    )
    embedding_points = 16
    rank = 5
    hankel = np.lib.stride_tricks.sliding_window_view(values, embedding_points).T
    left, singular_values, right = np.linalg.svd(hankel, full_matrices=False)
    reconstructed = (left[:, :rank] * singular_values[:rank]) @ right[:rank]
    expected = np.zeros(values.size, dtype=np.float64)
    counts = np.zeros(values.size, dtype=np.float64)
    for row in range(embedding_points):
        expected[row:row + reconstructed.shape[1]] += reconstructed[row]
        counts[row:row + reconstructed.shape[1]] += 1.0
    expected /= counts

    result = svd_denoise_series(
        _series(values),
        embedding_points=embedding_points,
        rank=rank,
        min_segment_points=4,
    )

    np.testing.assert_allclose(result.cleaned, expected, atol=1e-10, rtol=1e-10)


def test_hankel_svd_rejects_an_unbounded_embedding_workspace():
    values = np.sin(np.arange(12_000, dtype=np.float64) / 11.0)

    with pytest.raises(ValueError, match="embedding_points is too large"):
        svd_denoise_series(
            _series(values),
            embedding_points=6_000,
            rank=4,
            min_segment_points=4,
        )
