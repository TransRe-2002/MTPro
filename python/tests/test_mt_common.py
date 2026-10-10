from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from core.em_data import EMData
from processor._mt_common import (
    BACKEND_EMTF_PYWRAP,
    BACKEND_EMTF_RS,
    build_band_configs,
    build_spectrum_kwargs,
    detect_channels,
    has_electric_channel,
    has_magnetic_channel,
    is_e_channel,
    is_h_channel,
    native_emtf_rs_available,
    preprocess_channel,
)


class _StubEMData(EMData):
    def __init__(self, channel_names: list[str]):
        super().__init__("/tmp/stub.mat")
        idx = pd.date_range("2024-01-01", periods=3, freq="5s")
        self.data = xr.Dataset(coords={"time": idx})
        for ch in channel_names:
            self.data[ch] = xr.DataArray(np.zeros(3), dims=("time",), coords={"time": idx})

    def restore_data(self, ch: str):
        return None


def test_detect_channels_maps_standard_names():
    mapping = detect_channels(["Ex", "Ey", "Hx", "Hy", "Hz"])
    assert mapping == {"Ex": "Ex", "Ey": "Ey", "Hx": "Hx", "Hy": "Hy", "Hz": "Hz"}


def test_detect_channels_handles_alternative_prefixes():
    mapping = detect_channels(["Ex1", "Ey1", "Bx", "By", "Bz"])
    assert mapping["Ex"] == "Ex1"
    assert mapping["Ey"] == "Ey1"
    assert mapping["Hx"] == "Bx"
    assert mapping["Hy"] == "By"
    assert mapping["Hz"] == "Bz"


def test_detect_channels_leaves_missing_as_none():
    mapping = detect_channels(["Ex", "Hy"])
    assert mapping["Ex"] == "Ex"
    assert mapping["Ey"] is None
    assert mapping["Hx"] is None
    assert mapping["Hy"] == "Hy"
    assert mapping["Hz"] is None


def test_detect_channels_does_not_remap_occupied_slot():
    mapping = detect_channels(["Ex", "Ex2", "Ex3"])
    assert mapping["Ex"] == "Ex"
    assert mapping["Ey"] is None


def test_is_e_channel_and_is_h_channel():
    assert is_e_channel("Ex1")
    assert is_e_channel("ey")
    assert not is_e_channel("Hx")
    assert is_h_channel("Hx")
    assert is_h_channel("Bz")
    assert not is_h_channel("Ex")


def test_has_magnetic_and_electric_channels():
    mag_only = _StubEMData(["Hx", "Hy", "Hz"])
    assert has_magnetic_channel(mag_only)
    assert not has_electric_channel(mag_only)

    elec_only = _StubEMData(["Ex", "Ey"])
    assert not has_magnetic_channel(elec_only)
    assert has_electric_channel(elec_only)

    both = _StubEMData(["Ex", "Hx"])
    assert has_magnetic_channel(both)
    assert has_electric_channel(both)


def test_native_emtf_rs_available_returns_bool():
    assert isinstance(native_emtf_rs_available(), bool)


def test_build_band_configs_returns_list():
    periods = np.logspace(np.log10(1.0), np.log10(100.0), 5)
    configs = build_band_configs(periods)
    assert len(configs) == 4
    assert hasattr(configs[0], "min_period")
    assert hasattr(configs[0], "max_period")


def test_preprocess_channel_detrends_clean_data():
    # Linear trend -> detrend should remove it.
    arr = np.arange(100, dtype=float) * 2.0 + 5.0
    result = preprocess_channel(arr, "Ex")
    assert np.allclose(result, np.zeros(100), atol=1e-10)


def test_preprocess_channel_skips_nan_data():
    arr = np.arange(100, dtype=float)
    arr[50] = np.nan
    result = preprocess_channel(arr, "Ex")
    # Should return the array unchanged (with NaN).
    assert np.isnan(result[50])
    assert np.array_equal(result[~np.isnan(result)], arr[~np.isnan(arr)])


def test_build_spectrum_kwargs_rs_backend_uses_emtf_fc_grid():
    kwargs = build_spectrum_kwargs(
        sample_rate=100.0,
        window_size=1024,
        step_pct=50.0,
        huber_threshold=1.5,
        max_iterations=25,
        tolerance=1e-8,
        hz_arr=None,
        backend=BACKEND_EMTF_RS,
    )
    assert kwargs["sample_rate_hz"] == 100.0
    assert kwargs["window_size"] == 1024
    assert kwargs["step_size"] == 512
    assert kwargs["huber_threshold"] == 1.5
    assert kwargs["decimation_levels"][0].nwin == 128
    assert kwargs["decimation_levels"][0].nfft == 32
    assert kwargs["bands"]
    assert kwargs["station_name"] == "STATION"


def test_build_spectrum_kwargs_pywrap_backend_adds_grid_keys():
    kwargs = build_spectrum_kwargs(
        sample_rate=100.0,
        window_size=1024,
        step_pct=50.0,
        huber_threshold=1.5,
        max_iterations=25,
        tolerance=1e-8,
        hz_arr=None,
        backend=BACKEND_EMTF_PYWRAP,
        fft_window_size=128,
        n_bands=10,
        station_name="TEST01",
    )
    assert "decimation_levels" in kwargs
    assert "bands" in kwargs
    assert kwargs["station_name"] == "TEST01"


def test_build_spectrum_kwargs_step_size_clamps_to_one():
    kwargs = build_spectrum_kwargs(
        sample_rate=100.0,
        window_size=1024,
        step_pct=0.0,
        huber_threshold=1.5,
        max_iterations=25,
        tolerance=1e-8,
        hz_arr=None,
        backend=BACKEND_EMTF_RS,
    )
    assert kwargs["step_size"] == 1
