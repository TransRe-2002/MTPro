import numpy as np

from ui.widgets.energy_spectrum import compute_welch_psd


def test_welch_psd_finds_sine_peak():
    sample_rate = 100.0
    time = np.arange(8192) / sample_rate
    values = np.sin(2.0 * np.pi * 7.0 * time)
    frequencies, density, info = compute_welch_psd(values, sample_rate, 2048)

    assert abs(frequencies[np.argmax(density)] - 7.0) < 0.1
    assert info["finite_samples"] == len(values)


def test_welch_psd_uses_finite_runs_around_gaps():
    values = np.sin(np.linspace(0, 100, 4096))
    values[1200:1400] = np.nan
    frequencies, density, info = compute_welch_psd(values, 10.0, 1024)

    assert len(frequencies) == len(density) > 0
    assert info["runs_used"] == 2
    assert info["finite_samples"] == 3896
