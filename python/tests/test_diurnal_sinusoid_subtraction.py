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

    target_signal = (
        0.002 * time_seconds
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

    before_24h = _harmonic_amplitude(observed, 24.0)
    after_24h = _harmonic_amplitude(result.cleaned, 24.0)
    before_12h = _harmonic_amplitude(observed, 12.0)
    after_12h = _harmonic_amplitude(result.cleaned, 12.0)

    observed_rmse = np.sqrt(np.mean((observed.to_numpy() - target_signal) ** 2))
    cleaned_rmse = np.sqrt(np.mean((result.cleaned.to_numpy() - target_signal) ** 2))

    target_24h = _harmonic_amplitude(pd.Series(target_signal, index=index), 24.0)
    target_12h = _harmonic_amplitude(pd.Series(target_signal, index=index), 12.0)

    assert after_24h < before_24h * 0.15
    assert after_24h <= target_24h + 0.05
    assert after_12h < before_12h * 0.20
    assert after_12h <= target_12h + 0.05
    assert cleaned_rmse < observed_rmse * 0.25
    assert not result.window_diagnostics.empty
    assert {"harmonic_1_amplitude", "harmonic_2_amplitude"}.issubset(result.window_diagnostics.columns)


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
    valid = series.notna().to_numpy()
    samples = series.to_numpy(dtype=np.float64, copy=False)[valid]
    time_seconds = (
        series.index.view("i8").astype(np.float64) / 1e9
        - series.index.view("i8")[0] / 1e9
    )[valid]
    centered_time = time_seconds - np.mean(time_seconds)
    scale = max(np.max(np.abs(centered_time)), 1.0)
    phase = 2.0 * np.pi * time_seconds / (period_hours * 3600.0)
    design = np.column_stack(
        [
            np.ones_like(phase),
            centered_time / scale,
            np.cos(phase),
            np.sin(phase),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, samples, rcond=None)
    return float(np.hypot(coefficients[2], coefficients[3]))
