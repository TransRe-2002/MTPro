import numpy as np

from emtf.estimate import (
    _decimation_from_window_size,
    decimation_with_fft_window,
    estimate_single_station,
)


def test_decimation_from_window_size_sets_emtf_nwin_and_overlap():
    levels = _decimation_from_window_size(256)

    assert levels is not None
    assert len(levels) == 6
    assert [level.nwin for level in levels] == [256] * 6
    assert [level.window_length for level in levels] == [64.0] * 6
    assert [level.nfft for level in levels] == [64] * 6


def test_decimation_from_window_size_rejects_bundled_binary_limit():
    try:
        _decimation_from_window_size(1024)
    except ValueError as exc:
        assert "supports window_size <= 256" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_pywrap_window_size_does_not_override_dnff_decset(monkeypatch):
    captured = {}

    def fake_estimate(*args, **kwargs):
        captured["decimation_levels"] = kwargs["decimation_levels"]
        return object()

    monkeypatch.setattr("emtf.estimate._estimate", fake_estimate)

    estimate_single_station(
        np.ones(8),
        np.ones(8),
        np.ones(8),
        np.ones(8),
        1.0,
        window_size=2048,
    )

    assert captured["decimation_levels"] is None


def test_decimation_with_fft_window_sets_safe_dnff_nwin():
    levels = decimation_with_fft_window(1.0, 64)

    assert len(levels) == 6
    assert [level.nwin for level in levels] == [64] * 6
    assert [level.window_length for level in levels] == [16.0] * 6
    assert [level.nfft for level in levels] == [16] * 6


def test_decimation_with_fft_window_rejects_bundled_binary_limit():
    try:
        decimation_with_fft_window(1.0, 2048)
    except ValueError as exc:
        assert "FFT短窗点数 <= 256" in str(exc)
    else:
        raise AssertionError("expected ValueError")
