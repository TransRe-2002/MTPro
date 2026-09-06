"""utils/signal_decimate.py 的单元测试（纯函数，无 Qt 依赖）。"""

import numpy as np
import pytest

import utils.signal_decimate as signal_decimate
from utils.signal_decimate import decimated_length, decimate_nan_aware


class TestDecimatedLength:
    def test_divisible(self):
        assert decimated_length(100, 5) == 20

    def test_non_divisible(self):
        assert decimated_length(103, 5) == 21

    def test_empty(self):
        assert decimated_length(0, 5) == 0

    def test_factor_larger_than_npts(self):
        assert decimated_length(3, 5) == 1


class TestDecimateNanAware:
    def test_sine_wave_preserves_frequency_and_amplitude(self):
        # 1000 点上 5 个完整周期，抽取 5 倍 → 200 点仍是 5 个周期，
        # 归一化频率 0.025，远在抗混叠滤波器通带内
        n = 1000
        t = np.arange(n, dtype=np.float64)
        y = np.sin(2 * np.pi * 5 * t / n)

        factor = 5
        new_npts = decimated_length(n, factor)
        out = decimate_nan_aware(y, factor, new_npts)

        assert out.shape == (new_npts,)
        # boxcar FIR + 带内正弦：低频幅度近似保持
        assert np.nanmax(np.abs(out)) == pytest.approx(1.0, abs=0.01)
        # 5 个周期 → 过零 ~10 次
        zero_crossings = int(np.sum(np.diff(np.signbit(out)) != 0))
        assert 9 <= zero_crossings <= 11

    def test_no_nan_matches_boxcar_means(self):
        y = np.arange(20, dtype=np.float64)
        factor = 4
        new_npts = decimated_length(len(y), factor)

        out = decimate_nan_aware(y, factor, new_npts)
        expected = y.reshape(-1, factor).mean(axis=1)
        np.testing.assert_allclose(out, expected)

    def test_sparse_nan_window_uses_finite_mean(self):
        # 10 点窗口缺 1 点，TS 兼容默认策略仍输出有效点。
        y = np.arange(100, dtype=np.float64)
        y[23] = np.nan  # 位于第 2 个窗口 [20, 30)
        factor = 10
        new_npts = decimated_length(len(y), factor)

        out = decimate_nan_aware(y, factor, new_npts)
        assert out[2] == pytest.approx(np.nanmean(y[20:30]))
        assert np.isfinite(out[0]) and np.isfinite(out[1])
        assert np.isfinite(out[3])

    def test_sparse_nan_in_multiple_windows_does_not_erase_channel(self):
        y = np.ones(100, dtype=np.float64)
        y[10] = np.nan  # 窗口 1
        y[20] = np.nan  # 窗口 2
        y[30] = np.nan  # 窗口 3
        factor = 10
        new_npts = decimated_length(len(y), factor)

        out = decimate_nan_aware(y, factor, new_npts)
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out, 1.0)

    def test_window_over_missing_threshold_produces_nan(self):
        y = np.arange(20, dtype=np.float64)
        y[2:5] = np.nan  # 首窗口缺 3/10 = 30% > 指定的 25%

        out = decimate_nan_aware(y, 10, 2, 0.25)

        assert np.isnan(out[0])
        assert np.isfinite(out[1])

    def test_exact_missing_threshold_is_accepted(self):
        y = np.arange(8, dtype=np.float64)
        y[1] = np.nan  # 首窗口缺 1/4 = 25%

        out = decimate_nan_aware(y, 4, 2, 0.25)

        assert out[0] == pytest.approx(np.nanmean(y[:4]))

    def test_all_nan(self):
        y = np.full(50, np.nan)
        factor = 5
        new_npts = decimated_length(len(y), factor)

        out = decimate_nan_aware(y, factor, new_npts)
        assert out.shape == (new_npts,)
        assert np.isnan(out).all()

    def test_non_divisible_length(self):
        y = np.arange(103, dtype=np.float64)
        factor = 5
        new_npts = decimated_length(len(y), factor)
        assert new_npts == 21

        out = decimate_nan_aware(y, factor, new_npts)
        assert out.shape == (new_npts,)
        assert np.isfinite(out).all()

    def test_trailing_nan_in_partial_window_with_strict_threshold(self):
        # 末尾不足一个完整窗口，缺失 1/3 > 指定的 25%。
        y = np.arange(103, dtype=np.float64)
        y[102] = np.nan
        factor = 5
        new_npts = decimated_length(len(y), factor)

        out = decimate_nan_aware(y, factor, new_npts, 0.25)
        assert np.isnan(out[-1])
        assert np.isfinite(out[0])

    def test_ts_compatible_default_keeps_isolated_samples(self):
        # 复现 22-...-dt2_fixed.mat：有效点全部被 NaN 隔开。
        y = np.full(25, np.nan)
        y[::5] = np.arange(5, dtype=np.float64)

        out = decimate_nan_aware(y, 5, 5)

        np.testing.assert_allclose(out, np.arange(5, dtype=np.float64))

    def test_short_series_uses_boxcar(self):
        y = np.arange(10, dtype=np.float64)
        factor = 2
        new_npts = decimated_length(len(y), factor)

        out = decimate_nan_aware(y, factor, new_npts)
        np.testing.assert_allclose(out, y.reshape(-1, factor).mean(axis=1))

    def test_factor_validation(self):
        with pytest.raises(ValueError):
            decimate_nan_aware(np.arange(10.0), 1, 10)

    def test_missing_fraction_validation(self):
        with pytest.raises(ValueError):
            decimate_nan_aware(np.arange(10.0), 2, 5, -0.1)
        with pytest.raises(ValueError):
            decimate_nan_aware(np.arange(10.0), 2, 5, 1.1)

    def test_new_npts_validation(self):
        with pytest.raises(ValueError):
            decimate_nan_aware(np.arange(10.0), 2, 4)

    def test_numpy_fallback_matches_public_result(self, monkeypatch):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(10_003)
        y[::17] = np.nan
        y[::101] = np.inf
        factor = 7
        new_npts = decimated_length(len(y), factor)
        accelerated = decimate_nan_aware(y, factor, new_npts, 0.25)

        monkeypatch.setattr(signal_decimate, "_HAS_RUST_BOXCAR", False)
        fallback = decimate_nan_aware(y, factor, new_npts, 0.25)

        np.testing.assert_allclose(accelerated, fallback, equal_nan=True)


def test_rust_extension_exposes_nan_boxcar_when_installed():
    if not signal_decimate._HAS_RUST_BOXCAR:
        pytest.skip("installed rust_lib does not include nan_boxcar_decimate_rs")
    assert callable(signal_decimate.rust_lib.nan_boxcar_decimate_rs)
