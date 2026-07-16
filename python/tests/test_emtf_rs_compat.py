import numpy as np


def test_emtf_exposes_rs_compatible_api():
    import emtf
    from emtf._binaries import resolve_binary

    assert "/python/src/emtf/" in emtf.__file__
    assert "/python/src/bin/" in str(resolve_binary("dnff"))
    assert hasattr(emtf, "estimate_single_station_spectrum")
    assert hasattr(emtf, "estimate_remote_reference_spectrum")
    assert not hasattr(emtf, "estimate_single_station")
    assert not hasattr(emtf, "estimate_remote_reference")
    assert hasattr(emtf, "SpectrumResult")
    assert hasattr(emtf, "BandConfig")


def test_spectrum_result_to_bands_exports_atm_rpc_fields():
    import emtf

    result = emtf.SpectrumResult(
        frequencies_hz=np.array([1.0, 0.5, 0.25], dtype=float),
        zxx=np.array([1 + 1j, 2 + 2j, 3 + 3j]),
        zxy=np.array([2 + 0j, 4 + 0j, 8 + 0j]),
        zyx=np.array([1j, 2j, 4j]),
        zyy=np.array([1 - 1j, 2 - 2j, 3 - 3j]),
        zxx_se=np.array([0.1, 0.2, 0.3]),
        zxy_se=np.array([0.1, 0.2, 0.3]),
        zyx_se=np.array([0.1, 0.2, 0.3]),
        zyy_se=np.array([0.1, 0.2, 0.3]),
        rho_xy=np.array([0.8, 6.4, 51.2]),
        rho_yx=np.array([0.2, 1.6, 12.8]),
        phi_xy=np.zeros(3),
        phi_yx=np.full(3, 90.0),
        rho_xy_se=np.array([0.1, 0.2, 0.3]),
        rho_yx_se=np.array([0.1, 0.2, 0.3]),
        phi_xy_se=np.array([0.1, 0.2, 0.3]),
        phi_yx_se=np.array([0.1, 0.2, 0.3]),
        n_observations=np.array([10, 20, 30], dtype=np.int32),
    )

    bands = result.to_bands([
        emtf.BandConfig("wide", min_period=0.5, max_period=5.0),
    ])

    assert bands.center_frequencies_hz.shape == (1,)
    assert bands.zxy.shape == (1,)
    assert bands.rho_xy.shape == (1,)
    assert bands.phi_yx_se.shape == (1,)
    assert bands.n_observations.tolist() == [60]
    assert np.isfinite(bands.rho_xy[0])


def test_emtf_rs_shim_passes_native_bands_to_pywrap(monkeypatch):
    import emtf
    from emtf.types import FrequencyBand

    captured = {}

    def fake_estimate(*args, **kwargs):
        captured["bands"] = kwargs["bands"]
        return object()

    monkeypatch.setattr("emtf.estimate.estimate_single_station", fake_estimate)
    monkeypatch.setattr(
        emtf.api,
        "_from_zfile_data",
        lambda _zdata, station_name, mode, sample_rate_hz=1.0: (station_name, mode, sample_rate_hz),
    )
    bands = [FrequencyBand(1, 5, 5)]

    emtf.estimate_single_station_spectrum(
        np.ones(8),
        np.ones(8),
        np.ones(8),
        np.ones(8),
        1.0,
        bands=bands,
    )

    assert captured["bands"] is bands


def test_emtf_rs_shim_passes_window_size_to_pywrap(monkeypatch):
    import emtf

    captured = {}

    def fake_estimate(*args, **kwargs):
        captured["window_size"] = kwargs["window_size"]
        return object()

    monkeypatch.setattr("emtf.estimate.estimate_single_station", fake_estimate)
    monkeypatch.setattr(
        emtf.api,
        "_from_zfile_data",
        lambda _zdata, station_name, mode, sample_rate_hz=1.0: (station_name, mode, sample_rate_hz),
    )

    emtf.estimate_single_station_spectrum(
        np.ones(8),
        np.ones(8),
        np.ones(8),
        np.ones(8),
        1.0,
        window_size=256,
    )

    assert captured["window_size"] == 256


def test_emtf_rs_shim_passes_decimation_levels_to_pywrap(monkeypatch):
    import emtf
    from emtf.types import DecimationLevel

    captured = {}

    def fake_estimate(*args, **kwargs):
        captured["decimation_levels"] = kwargs["decimation_levels"]
        return object()

    monkeypatch.setattr("emtf.estimate.estimate_single_station", fake_estimate)
    monkeypatch.setattr(
        emtf.api,
        "_from_zfile_data",
        lambda _zdata, station_name, mode, sample_rate_hz=1.0: (station_name, mode, sample_rate_hz),
    )
    levels = [DecimationLevel(), DecimationLevel()]

    emtf.estimate_single_station_spectrum(
        np.ones(8),
        np.ones(8),
        np.ones(8),
        np.ones(8),
        1.0,
        decimation_levels=levels,
    )

    assert captured["decimation_levels"] is levels


def test_zfile_periods_from_pywrap_are_scaled_to_seconds(monkeypatch):
    import emtf
    from emtf.io import BandedResult

    banded = BandedResult(
        periods=np.array([22.456, 2048.0], dtype=np.float64),
        zxx=np.zeros(2, dtype=np.complex128),
        zxy=np.ones(2, dtype=np.complex128),
        zyx=np.ones(2, dtype=np.complex128),
        zyy=np.zeros(2, dtype=np.complex128),
        zxx_se=np.zeros(2, dtype=np.float64),
        zxy_se=np.zeros(2, dtype=np.float64),
        zyx_se=np.zeros(2, dtype=np.float64),
        zyy_se=np.zeros(2, dtype=np.float64),
        n_observations=np.array([1, 1], dtype=np.int32),
    )
    monkeypatch.setattr("emtf.io.to_banded_result", lambda *_args, **_kwargs: banded)

    result = emtf.api._from_zfile_data(
        object(),
        station_name="TEST",
        mode="ss",
        sample_rate_hz=0.2,
    )

    np.testing.assert_allclose(result.periods, [112.28, 10240.0])
    np.testing.assert_allclose(result.frequencies_hz, [1.0 / 112.28, 1.0 / 10240.0])
