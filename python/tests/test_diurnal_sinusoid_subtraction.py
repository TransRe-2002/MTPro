import numpy as np
import pandas as pd
import xarray as xr

from core.em_data import EMData
from processor.diurnal_sinusoid_subtraction import (
    PeriodicSubtractionConfig,
    subtract_diurnal_signal_channel,
    subtract_diurnal_signal_series,
)


class DiurnalDemoData(EMData):
    def __init__(self, series: pd.Series):
        super().__init__("/tmp/diurnal_demo.mat")
        self.name = "diurnal-demo"
        self.start_time = series.index[0]
        self.end_time = series.index[-1]
        self.dt = series.index[1] - series.index[0]
        self.datetime_index = series.index
        self.kp_data = None
        self.npts = len(series)
        self.data = xr.Dataset(coords={"time": self.datetime_index})
        self.data["Ex1"] = xr.DataArray(
            series.to_numpy(dtype=np.float64),
            dims=("time",),
            coords={"time": self.datetime_index},
        )

    def restore_data(self, ch: str):
        return None


def test_subtract_diurnal_signal_series_removes_24h_and_12h_components():
    index = pd.date_range("2024-01-01", periods=14 * 24 * 6, freq="10min")
    time_seconds = np.arange(len(index), dtype=np.float64) * 600.0

    # 长周期漂移远小于日变振幅（~6 nT），是 MT 实测的典型比例
    drift = 1.5 * time_seconds / (14.0 * 86400.0)  # 0 → 1.5 nT over 14 days
    target_signal = (
        drift
        + 1.4 * np.sin(2.0 * np.pi * time_seconds / (36.0 * 3600.0) + 0.3)
        + 0.6 * np.cos(2.0 * np.pi * time_seconds / (7.0 * 24.0 * 3600.0) - 0.2)
    )
    diurnal_noise = (
        6.0 * np.sin(2.0 * np.pi * time_seconds / (24.0 * 3600.0) + 0.7)
        + 2.5 * np.cos(2.0 * np.pi * time_seconds / (12.0 * 3600.0) - 0.1)
    )
    observed = pd.Series(target_signal + diurnal_noise, index=index, name="Ex1")

    result = subtract_diurnal_signal_series(
        observed,
        harmonics=(1, 2),
        window_days=3.0,
        step_days=1.0,
        trend_degree=1,
        min_valid_fraction=0.8,
        min_valid_samples=200,
    )

    observed_rmse = np.sqrt(np.mean((observed.to_numpy() - target_signal) ** 2))
    cleaned_rmse = np.sqrt(np.mean((result.cleaned.to_numpy() - target_signal) ** 2))

    # 过滤后 RMSE 应显著减小（>90% 日变噪声被去除）
    assert cleaned_rmse < observed_rmse * 0.15
    # 窗口诊断表应包含谐波信息
    assert not result.window_diagnostics.empty
    assert {"harmonic_1_amplitude", "harmonic_2_amplitude"}.issubset(result.window_diagnostics.columns)
    # 各窗口谐波振幅应该在合理范围内（接近注入的 6.0 和 2.5）
    ok_rows = result.window_diagnostics[result.window_diagnostics["status"] == "ok"]
    assert not ok_rows.empty
    avg_h1 = float(ok_rows["harmonic_1_amplitude"].mean())
    avg_h2 = float(ok_rows["harmonic_2_amplitude"].mean())
    assert 4.0 < avg_h1 < 8.0   # 注入 6.0，允许因窗口造成的合理偏差
    assert 1.5 < avg_h2 < 4.0   # 注入 2.5


def test_subtract_diurnal_signal_channel_preserves_nan_locations_and_can_write_back():
    index = pd.date_range("2024-02-01", periods=8 * 24, freq="1h")
    time_seconds = np.arange(len(index), dtype=np.float64) * 3600.0
    signal = 0.05 * time_seconds / 3600.0 + 4.0 * np.sin(2.0 * np.pi * time_seconds / (24.0 * 3600.0))
    series = pd.Series(signal, index=index, name="Ex1")
    series.iloc[20:28] = np.nan

    em_data = DiurnalDemoData(series)
    result = subtract_diurnal_signal_channel(
        em_data,
        "Ex1",
        harmonics=(1,),
        window_days=2.0,
        step_days=1.0,
        trend_degree=1,
        min_valid_fraction=0.7,
        min_valid_samples=24,
        inplace=True,
    )

    updated = em_data.get_channel_series("Ex1")

    assert result.cleaned.index.equals(series.index)
    assert updated.index.equals(series.index)
    assert np.array_equal(result.cleaned.isna().to_numpy(), series.isna().to_numpy())
    assert np.array_equal(updated.isna().to_numpy(), series.isna().to_numpy())
    np.testing.assert_allclose(
        updated[updated.notna()].to_numpy(),
        result.cleaned[result.cleaned.notna()].to_numpy(),
    )
    assert result.coverage_weight[result.coverage_weight.notna()].gt(0.0).all()


def test_periodic_subtraction_config_rejects_invalid_harmonics():
    try:
        PeriodicSubtractionConfig(harmonics=(0, 1))
    except ValueError as exc:
        assert "positive integers" in str(exc)
    else:
        raise AssertionError("PeriodicSubtractionConfig should reject non-positive harmonics")


def _harmonic_amplitude(series: pd.Series, period_hours: float) -> float:
    """测量序列中特定周期成分的振幅 — 纯谐波拟合，不含趋势项。"""
    valid = series.notna().to_numpy()
    samples = series.to_numpy(dtype=np.float64, copy=False)[valid]
    time_seconds = (
        series.index.view("i8").astype(np.float64) / 1e9
        - series.index.view("i8")[0] / 1e9
    )[valid]
    phase = 2.0 * np.pi * time_seconds / (period_hours * 3600.0)
    design = np.column_stack([np.cos(phase), np.sin(phase)])
    coefficients, *_ = np.linalg.lstsq(design, samples, rcond=None)
    return float(np.hypot(coefficients[0], coefficients[1]))
