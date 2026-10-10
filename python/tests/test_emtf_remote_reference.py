import sys

import numpy as np
import pandas as pd
import pytest

from emtf.estimate import decimation_with_fft_window, estimate_remote_reference
from emtf.time_alignment import matching_remote_slice
from emtf.types import FrequencyBand


def test_remote_reference_rejects_mismatched_channel_lengths():
    with pytest.raises(ValueError, match="remote_hx has 7"):
        estimate_remote_reference(
            np.ones(8),
            np.ones(8),
            np.ones(8),
            np.ones(8),
            np.ones(7),
            np.ones(8),
            1.0,
        )


def test_matching_remote_slice_requires_exact_timestamp_alignment():
    local = pd.date_range("2024-01-01", periods=8, freq="s")
    remote = pd.date_range("2023-12-31 23:59:58", periods=12, freq="s")

    assert matching_remote_slice(local, remote, 2, 6) == (4, 8)

    delayed_remote = pd.date_range("2024-01-01 00:00:00.5", periods=8, freq="s")
    with pytest.raises(ValueError, match="逐样点对齐"):
        matching_remote_slice(local, delayed_remote, 0, 4)


def test_matching_remote_slice_accepts_same_instants_in_different_timezones():
    local = pd.date_range("2024-01-01", periods=8, freq="s", tz="Asia/Shanghai")
    remote = pd.date_range("2023-12-31 15:59:58", periods=12, freq="s", tz="UTC")

    assert matching_remote_slice(local, remote, 2, 6) == (4, 8)


@pytest.mark.skipif(sys.platform != "linux", reason="bundled EMTF integration binary is Linux-only")
def test_remote_reference_builds_and_processes_a_two_channel_remote_fc_group():
    n_samples = 4096
    samples = np.arange(n_samples, dtype=np.float64)
    local_hx = np.sin(2 * np.pi * samples / 80.0) + 0.2 * np.sin(2 * np.pi * samples / 31.0)
    local_hy = np.cos(2 * np.pi * samples / 71.0) + 0.1 * np.sin(2 * np.pi * samples / 23.0)
    ex = 0.8 * local_hx + 0.1 * local_hy
    ey = -0.2 * local_hx + 0.7 * local_hy
    remote_hx = local_hx + 0.03 * np.sin(2 * np.pi * samples / 17.0)
    remote_hy = local_hy + 0.02 * np.cos(2 * np.pi * samples / 19.0)

    zdata = estimate_remote_reference(
        ex,
        ey,
        local_hx,
        local_hy,
        remote_hx,
        remote_hy,
        1.0,
        station_name="RRTEST",
        decimation_levels=decimation_with_fft_window(1.0, 64)[:2],
        bands=[FrequencyBand(1, 5, 5)],
    )

    assert zdata.nch == 4
    assert zdata.nche == 2
    assert zdata.nbt == 1
    assert zdata.chid == ["Hx", "Hy", "Ex", "Ey"]
