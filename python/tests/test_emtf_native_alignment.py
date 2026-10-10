"""Native/Fortran integration using deterministic generated data, no MAT fixtures."""
import sys

import numpy as np
import pytest

from base.emtf_native import estimate_native_emtf, native_level_specs
from emtf.estimate import decimation_with_fft_window
from emtf.types import FrequencyBand


def native_backend():
    native = pytest.importorskip("emtf_rs")
    if not hasattr(native, "emtf_spectra"):
        pytest.skip("Rebuild emtf-rs with the EMTF compatibility API")
    return native


@pytest.mark.skipif(sys.platform != "linux", reason="Fortran reference is Linux-only")
@pytest.mark.parametrize("window,gaps", [(64, False), (128, False), (256, False), (128, True)])
def test_native_fc_matches_dnff_all_levels_including_first_window(window, gaps):
    from emtf.pipeline.dnff import dnff_from_arrays
    from emtf.io.fc import FCOutput

    native = native_backend()
    levels = decimation_with_fft_window(.2, window)[:4]
    # Exactly representable at ASC precision, so compare numerical processing
    # independently of the wrapper's integer time-series quantisation.
    data = np.round(np.random.default_rng(4).normal(size=(4, 32768)) * 1000) / 100
    if gaps:
        data[:, 4100:4120] = np.nan
        data[0, 1024] = np.nan
        data[2, 8100:8103] = np.nan
    ws = dnff_from_arrays(list(data), 5., levels, channel_ids=["Hx", "Hy", "Ex", "Ey"],
                          scales=[100.] * 4)
    try:
        fc = FCOutput.read(next((ws.path / "FC").glob("*.f*")))
        output = native.emtf_spectra(data, .2, native_level_specs(levels))
        for index, (rate, starts, values) in enumerate(output):
            stride = 4 ** index
            assert rate == pytest.approx(.2 / stride)
            np.testing.assert_array_equal(
                np.rint(np.asarray(starts) / (stride * window * .75)).astype(int) + 1,
                fc.i_sets[index],
            )
            for frequency in (1, 5, levels[index].nfft):
                reference = fc.load_spectrum(index + 1, frequency)
                actual = values[:, frequency - 1]
                # Original FC packing quantises mantissas to 12 bits.
                error = np.linalg.norm(reference - actual) / np.linalg.norm(reference)
                assert error < .002, (index, frequency, error)
    finally:
        ws.cleanup_now()


@pytest.mark.skipif(sys.platform != "linux", reason="Fortran reference is Linux-only")
@pytest.mark.parametrize("remote", [False, True])
def test_native_band_periods_and_impedance_match_fortran(remote):
    import emtf

    native_backend()
    rng = np.random.default_rng(8)
    hx, hy = rng.normal(size=(2, 32768)) * 100
    ex = .3 * hx + .8 * hy + rng.normal(size=hx.size)
    ey = -.7 * hx + .2 * hy + rng.normal(size=hx.size)
    hz = .1 * hx - .2 * hy + rng.normal(size=hx.size) * .1
    arrays = [ex, ey, hx, hy]
    if remote:
        arrays.extend([hx + rng.normal(size=hx.size), hy + rng.normal(size=hy.size)])
    bands = [FrequencyBand(level, lo, hi) for level in range(1, 5)
             for lo, hi in ((18, 25), (10, 14), (5, 7))]
    kwargs = dict(sample_rate_hz=.2, decimation_levels=decimation_with_fft_window(.2, 128)[:4],
                  bands=bands, station_name="ALIGN", hz=hz)
    actual = estimate_native_emtf(*arrays, remote_reference=remote, **kwargs)
    estimator = emtf.estimate_remote_reference_spectrum if remote else emtf.estimate_single_station_spectrum
    reference = estimator(*arrays, **kwargs)
    np.testing.assert_allclose(actual.periods, reference.periods, rtol=5e-5)
    assert actual.processing_scheme == "emtf_fc_bands_v1"
    for field in ("zxy", "zyx", "tzx", "tzy"):
        np.testing.assert_allclose(getattr(actual, field), getattr(reference, field), rtol=.025, atol=.003)


def test_native_rejects_unsupported_offsets():
    levels = decimation_with_fft_window(.2, 128)
    levels[0].decimation_offset = 1
    with pytest.raises(ValueError, match="zero clock"):
        native_level_specs(levels)


def test_both_backends_receive_identical_fc_plan():
    from output_tools.shared.emtf import build_spectrum_kwargs
    common = dict(sample_rate=.2, window_size=2048, step_pct=50., huber_threshold=1.5,
                  max_iterations=25, tolerance=1e-8, hz_arr=None, fft_window_size=128,
                  n_bands=40, station_name="ALIGN")
    rust = build_spectrum_kwargs(backend="emtf_rs", **common)
    reference = build_spectrum_kwargs(backend="emtf_pywrap", **common)
    assert rust == reference


@pytest.mark.parametrize("remote", [False, True])
def test_widgets_keep_native_fc_bands_without_second_averaging(qtbot, monkeypatch, remote):
    from types import SimpleNamespace
    from output_tools.robust_estimate import RobustEstimateWidget
    from output_tools.remote_reference import RemoteReferenceWidget

    widget = (RemoteReferenceWidget if remote else RobustEstimateWidget)()
    qtbot.addWidget(widget)
    for backend in ("emtf_rs", "emtf_pywrap"):
        widget.backend_combo.setCurrentIndex(widget.backend_combo.findData(backend))
        assert not widget.fft_window_combo.isHidden()
        assert all(field.isHidden() for field in widget._rs_only_fields)
    widget._spectrum = SimpleNamespace(processing_scheme="emtf_fc_bands_v1", station_name="ALIGN")
    sentinel = object()
    monkeypatch.setattr("emtf.io.from_result", lambda *_args, **_kwargs: sentinel)
    widget._build_bands(backend="emtf_rs")
    assert widget._bands is sentinel
