"""utils/signal_decimate.py 的单元测试（纯函数，无 Qt 依赖）。"""

import numpy as np
import pytest
from scipy.signal import decimate as scipy_decimate

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
        # 零相位滤波 + 带内正弦：幅度近似保持
        assert np.nanmax(np.abs(out)) == pytest.approx(1.0, abs=0.01)
        # 5 个周期 → 过零 ~10 次
        zero_crossings = int(np.sum(np.diff(np.signbit(out)) != 0))
        assert 9 <= zero_crossings <= 11

    def test_no_nan_matches_scipy(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(500)
        factor = 4
        new_npts = decimated_length(len(y), factor)

        out = decimate_nan_aware(y, factor, new_npts)
        expected = scipy_decimate(y, factor)[:new_npts]
        np.testing.assert_allclose(out, expected)

    def test_nan_window_produces_nan_output(self):
        # 单个 NaN 所在窗口 → 对应输出点为 NaN，其余窗口不受影响
        y = np.arange(100, dtype=np.float64)
        y[23] = np.nan  # 位于第 2 个窗口 [20, 30)
        factor = 10
        new_npts = decimated_length(len(y), factor)

        out = decimate_nan_aware(y, factor, new_npts)
        assert np.isnan(out[2])
        assert np.isfinite(out[0]) and np.isfinite(out[1])
        assert np.isfinite(out[3])

    def test_nan_positions_multiple_windows(self):
        y = np.ones(100, dtype=np.float64)
        y[10] = np.nan  # 窗口 1
        y[20] = np.nan  # 窗口 2
        y[30] = np.nan  # 窗口 3
        factor = 10
        new_npts = decimated_length(len(y), factor)

        out = decimate_nan_aware(y, factor, new_npts)
        assert np.isnan(out[1]) and np.isnan(out[2]) and np.isnan(out[3])
        assert np.isfinite(out[0]) and np.isfinite(out[4])

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

    def test_trailing_nan_in_partial_window(self):
        # 末尾不足一个完整窗口的部分含 NaN → 末输出点为 NaN
        y = np.arange(103, dtype=np.float64)
        y[102] = np.nan
        factor = 5
        new_npts = decimated_length(len(y), factor)

        out = decimate_nan_aware(y, factor, new_npts)
        assert np.isnan(out[-1])
        assert np.isfinite(out[0])

    def test_short_series_falls_back_to_striding(self):
        # 低于滤波最小长度：退化为直接抽点，不抛错
        y = np.arange(10, dtype=np.float64)
        factor = 2
        new_npts = decimated_length(len(y), factor)

        out = decimate_nan_aware(y, factor, new_npts)
        np.testing.assert_allclose(out, y[::factor][:new_npts])

    def test_factor_validation(self):
        with pytest.raises(ValueError):
            decimate_nan_aware(np.arange(10.0), 1, 10)
