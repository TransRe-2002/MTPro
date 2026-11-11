from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from base.plugin_runtime import run_plugin_process
from core.algorithms.wavelet import wavelet_full_channel_denoise, wavelet_multiscale_denoise
from plugin_api import ChannelBlock
from plugins.remove_wavelet_denoise.plugin import WaveletDenoisePlugin


def _noisy_signal(size: int = 4096) -> np.ndarray:
    rng = np.random.default_rng(20260906)
    x = np.arange(size, dtype=np.float64)
    return np.sin(2.0 * np.pi * x / 400.0) + 0.35 * rng.normal(size=size)


def test_full_channel_processes_finite_segments_and_preserves_gaps():
    pytest.importorskip("pywt")
    first = _noisy_signal(300)
    first[100:103] = np.nan
    second = _noisy_signal(240)
    short = np.arange(8, dtype=float)
    source = np.r_[np.full(4, np.nan), first, np.full(20, np.nan), second, np.full(20, np.nan), short]
    result = wavelet_full_channel_denoise(source, mode="zero", wavelet_name="db4", level=3)
    np.testing.assert_array_equal(np.isnan(result.corrected), np.isnan(source))
    expected_first = wavelet_multiscale_denoise(first, (0, len(first) - 1), mode="zero", wavelet_name="db4", level=3)
    expected_second = wavelet_multiscale_denoise(second, (0, len(second) - 1), mode="zero", wavelet_name="db4", level=3)
    np.testing.assert_allclose(result.corrected[4:304], expected_first.corrected, equal_nan=True)
    np.testing.assert_allclose(result.corrected[324:564], expected_second.corrected, equal_nan=True)
    np.testing.assert_array_equal(result.corrected[-8:], short)
    assert "已处理 2 个有效数据段" in result.message
    assert "过短片段" in result.message
    assert "最长有效段" in result.message
    assert result.region == (0, len(source) - 1)
    index = pd.date_range("2025-01-01", periods=len(source), freq="s", tz="UTC")
    block = ChannelBlock("Ex", pd.Series(source, index=index))
    plugin_result, _ = run_plugin_process(
        WaveletDenoisePlugin(), block,
        {"mode": "zero", "wavelet": "db4", "level": 3},
    )
    np.testing.assert_allclose(plugin_result.series.to_numpy(), result.corrected, equal_nan=True)
    assert "过短片段" in plugin_result.diagnostics["message"]


def test_full_channel_rejects_data_without_long_enough_segment():
    with pytest.raises(ValueError, match="没有足够长"):
        wavelet_full_channel_denoise(np.r_[np.ones(8), np.full(20, np.nan), np.ones(8)])


def test_full_channel_editor_previews_and_confirms_across_long_gap(qtbot):
    from plugins.remove_wavelet_denoise.editor import RemoveWaveletDenoise

    values = _noisy_signal(800)
    values[300:320] = np.nan
    index = pd.date_range("2025-01-01", periods=len(values), freq="s", tz="UTC")
    widget = RemoveWaveletDenoise(pd.Series(values, index=index, name="Ex"), "Ex")
    qtbot.addWidget(widget)
    widget._preview_fit()
    qtbot.waitUntil(lambda: widget.last_result is not None, timeout=10000)
    assert "已处理 2 个有效数据段" in widget.processing_status.text()
    widget._apply_correction()
    qtbot.waitUntil(lambda: widget.processing_status.text().startswith("已应用"), timeout=10000)
    received = []
    widget.result_signal.connect(lambda _channel, result: received.append(result))
    widget._confirm_and_close()
    assert len(received) == 1
    np.testing.assert_array_equal(np.isnan(received[0]), np.isnan(values))


def test_sureshrink_is_per_level_and_never_exceeds_universal_threshold():
    pytest.importorskip("pywt")
    samples = _noisy_signal()
    kwargs = {
        "region": (1800, 2000),
        "wavelet_name": "sym6",
        "level": 5,
        "sample_rate_hz": 2.0,
        "padding_points": 300,
        "mode": "soft",
    }

    universal = wavelet_multiscale_denoise(
        samples, threshold_method="universal", **kwargs,
    )
    sure = wavelet_multiscale_denoise(
        samples, threshold_method="sure", **kwargs,
    )

    assert sure.threshold_method == "sure"
    assert sure.thresholds.shape == universal.thresholds.shape == (5,)
    assert np.all(sure.thresholds <= universal.thresholds + 1e-12)


def test_protection_period_selects_level_from_sample_rate():
    pytest.importorskip("pywt")
    samples = _noisy_signal()

    result = wavelet_multiscale_denoise(
        samples,
        (1000, 1500),
        wavelet_name="sym6",
        level_policy="auto_by_period",
        protect_period_seconds=64.0,
        sample_rate_hz=2.0,
        padding_points=200,
        threshold_method="sure",
    )

    # fs * P = 128, so the retained approximation is selected at J=6.
    assert result.level == 6
    assert result.level_policy == "auto_by_period"
    assert result.protect_period_seconds == 64.0


def test_seconds_padding_is_converted_using_sample_rate():
    pytest.importorskip("pywt")
    samples = _noisy_signal()

    result = wavelet_multiscale_denoise(
        samples,
        (1000, 1010),
        wavelet_name="sym4",
        level=3,
        sample_rate_hz=10.0,
        padding_points=8,
        padding_seconds=20.0,
        threshold_method="sure",
    )

    assert result.padding_points == 200
    assert result.local_window == (800, 1210)


def test_swt_reconstructs_and_preserves_missing_samples():
    pytest.importorskip("pywt")
    samples = _noisy_signal()
    samples[1900:1910] = np.nan

    result = wavelet_multiscale_denoise(
        samples,
        (1850, 2000),
        transform_kind="swt",
        wavelet_name="sym6",
        level=4,
        sample_rate_hz=1.0,
        padding_points=200,
        threshold_method="sure",
    )

    assert result.transform_kind == "swt"
    assert len(result.band_labels) == 4
    assert result.corrected.shape == samples.shape
    assert np.all(np.isnan(result.corrected[1900:1910]))
    assert np.all(np.isfinite(result.corrected[np.isfinite(samples)]))


def test_plugin_api_accepts_adaptive_strategy_fields():
    pytest.importorskip("pywt")
    samples = _noisy_signal()
    index = pd.date_range("2026-09-06", periods=samples.size, freq="s", tz="Asia/Shanghai")
    block = ChannelBlock("Ex", pd.Series(samples, index=index, name="Ex"))

    result, params = run_plugin_process(
        WaveletDenoisePlugin(),
        block,
        {
            "region": [1500, 1800],
            "transform": "dwt",
            "wavelet": "sym6",
            "threshold_method": "sure",
            "level_policy": "auto_by_period",
            "protect_period_seconds": 128.0,
            "padding_unit": "seconds",
            "padding_seconds": 30.0,
        },
    )

    assert params["threshold_method"] == "sure"
    assert params["level_policy"] == "auto_by_period"
    assert params["padding_unit"] == "seconds"
    assert params["padding_seconds"] == 30.0
    assert result.series.shape == block.series.shape
