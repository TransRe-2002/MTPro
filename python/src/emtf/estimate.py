"""High-level estimation API — drop-in replacement for ``emtf_rs``.

These functions accept numpy arrays (same signature as ``emtf_rs``) and
run the full Fortran pipeline under the hood:

    numpy arrays → binary files → dnff → FC files → tranmt → Z file → parse

Unlike ``emtf_rs`` (which does everything in Rust), this module calls
the original Fortran binaries via :func:`emtf.pipeline.dnff` and
:func:`emtf.pipeline.tranmt.tranmt`.

Band aggregation
----------------
``emtf_rs`` returns per-frequency spectra.  The Fortran ``tranmt``
returns per-band results (each band aggregates a range of FC indices).
By default, this module uses a standard 1 Hz band table.  Pass a custom
``bands`` list for different sampling rates or period ranges.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from emtf.io.zfile import ZFileData
from emtf.band_presets import standard_bands
from emtf.pipeline import dnff as _dnff
from emtf.pipeline.tranmt import tranmt as _tranmt
from emtf.types import DecimationLevel, FrequencyBand, ProcessingOptions

_MAX_DNFF_WINDOW_SIZE = 256

# ---------------------------------------------------------------------------
# Default decimation scheme for 1 Hz sampling (standard long-period MT)
# ---------------------------------------------------------------------------
_DECIMATION_1HZ = [
    DecimationLevel(nwin=128, window_length=32.0, decimation_factor=1, nfilt=1, fil_1=1.0),
    DecimationLevel(nwin=128, window_length=32.0, decimation_factor=4, nfilt=4,
                    fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705),
    DecimationLevel(nwin=128, window_length=32.0, decimation_factor=4, nfilt=4,
                    fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705),
    DecimationLevel(nwin=128, window_length=32.0, decimation_factor=4, nfilt=4,
                    fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705),
    DecimationLevel(nwin=128, window_length=32.0, decimation_factor=4, nfilt=4,
                    fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705),
    DecimationLevel(nwin=128, window_length=32.0, decimation_factor=4, nfilt=4,
                    fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705),
]

# Default frequency bands: ~30 s to ~10,000 s for 1 Hz data
_BANDS_1HZ = [
    FrequencyBand(1, 8, 9),
    FrequencyBand(1, 6, 7),
    FrequencyBand(1, 5, 5),
    FrequencyBand(2, 14, 17),
    FrequencyBand(2, 11, 13),
    FrequencyBand(2, 9, 10),
    FrequencyBand(2, 7, 8),
    FrequencyBand(2, 6, 6),
    FrequencyBand(2, 5, 5),
    FrequencyBand(3, 14, 17),
    FrequencyBand(3, 11, 13),
    FrequencyBand(3, 9, 10),
    FrequencyBand(3, 7, 8),
    FrequencyBand(3, 6, 6),
    FrequencyBand(3, 5, 5),
    FrequencyBand(4, 14, 17),
    FrequencyBand(4, 11, 13),
    FrequencyBand(4, 9, 10),
    FrequencyBand(4, 7, 8),
    FrequencyBand(4, 6, 6),
    FrequencyBand(4, 5, 5),
    FrequencyBand(5, 14, 17),
    FrequencyBand(5, 11, 13),
    FrequencyBand(5, 9, 10),
]


def estimate_single_station(
    ex: NDArray[np.float64],
    ey: NDArray[np.float64],
    hx: NDArray[np.float64],
    hy: NDArray[np.float64],
    sample_rate_hz: float,
    *,
    hz: NDArray[np.float64] | None = None,
    station_name: str = "STATION",
    window_size: int | None = None,
    step_size: int | None = None,           # ignored
    max_missing_per_window: int = 0,           # ignored
    max_gap_fill: int = 0,                    # ignored
    huber_threshold: float = 1.5,              # passed through
    max_iterations: int = 25,                  # passed through
    tolerance: float = 1e-8,                   # ignored (Fortran uses its own)
    decimation_levels: Sequence[DecimationLevel] | None = None,
    bands: Sequence[FrequencyBand] | None = None,
    cleanup: bool = True,
) -> ZFileData:
    """Estimate MT impedance — drop-in replacement for ``emtf_rs.estimate_single_station``.

    Parameters
    ----------
    ex, ey, hx, hy:
        1-D numpy float64 arrays, same length.  Time series for each channel.
        Units: E in mV/km, H in nT (EMTF convention).
    sample_rate_hz:
        Sampling rate in Hz (e.g. 1.0 for 1-second data).
    hz:
        Optional vertical magnetic field channel (for tipper estimation).
    station_name:
        Identifier used in output file names.
    window_size:
        Accepted for API compatibility with ``emtf_rs``.  The Fortran EMTF
        backend does not map this to ``decset.cfg`` ``nwin`` because long
        periods are selected through ``bands``/``bs.cfg`` and cascade
        decimation, not by enlarging the FFT window in ``dnff``.
    step_size, max_missing_per_window, max_gap_fill:
        Accepted for API compatibility with ``emtf_rs``, but ignored by
        the Fortran EMTF pipeline.
    huber_threshold:
        Huber parameter for robust estimation (default 1.5).
    max_iterations:
        Max IRLS iterations (default 25).
    tolerance:
        Accepted for API compatibility but **ignored** — Fortran uses its
        own convergence criteria.
    decimation_levels:
        Cascade decimation scheme.  Default: 6 levels at factor 4 (1 Hz MT).
    bands:
        Frequency bands for estimation.  Default: 24 bands from ~30 s to ~10,000 s.
    cleanup:
        If ``True``, remove temporary files after the run.

    Returns
    -------
    :class:`ZFileData` with impedance tensors, apparent resistivity, phase,
    and error estimates for each frequency band.
    """
    return _estimate(
        ex=ex, ey=ey, hx=hx, hy=hy, hz=hz,
        station_name=station_name,
        sample_rate_hz=sample_rate_hz,
        remote_hx=None, remote_hy=None,
        decimation_levels=decimation_levels,
        bands=bands,
        cleanup=cleanup,
    )


def estimate_remote_reference(
    ex: NDArray[np.float64],
    ey: NDArray[np.float64],
    local_hx: NDArray[np.float64],
    local_hy: NDArray[np.float64],
    remote_hx: NDArray[np.float64],
    remote_hy: NDArray[np.float64],
    sample_rate_hz: float,
    *,
    hz: NDArray[np.float64] | None = None,
    station_name: str = "STATION",
    window_size: int | None = None,
    step_size: int | None = None,            # ignored
    max_missing_per_window: int = 0,            # ignored
    max_gap_fill: int = 0,                     # ignored
    huber_threshold: float = 1.5,               # passed through
    max_iterations: int = 25,                   # passed through
    tolerance: float = 1e-8,                    # ignored
    decimation_levels: Sequence[DecimationLevel] | None = None,
    bands: Sequence[FrequencyBand] | None = None,
    cleanup: bool = True,
) -> ZFileData:
    """Estimate MT impedance with remote reference — drop-in replacement
    for ``emtf_rs.estimate_remote_reference``.

    Uses a distant station's magnetic field as the instrument in IV
    regression, removing local noise bias.

    Parameters
    ----------
    ex, ey:
        Local station E-field channels.
    local_hx, local_hy:
        Local station H-field channels.
    remote_hx, remote_hy:
        Remote reference station H-field channels.  Must be sampled
        synchronously at the same rate.
    (other parameters — see :func:`estimate_single_station`)

    Returns
    -------
    :class:`ZFileData`
    """
    return _estimate(
        ex=ex, ey=ey, hx=local_hx, hy=local_hy, hz=hz,
        station_name=station_name,
        sample_rate_hz=sample_rate_hz,
        remote_hx=remote_hx, remote_hy=remote_hy,
        decimation_levels=decimation_levels,
        bands=bands,
        cleanup=cleanup,
    )


# ---------------------------------------------------------------------------
# internal
# ---------------------------------------------------------------------------


def _estimate(
    ex: NDArray[np.float64],
    ey: NDArray[np.float64],
    hx: NDArray[np.float64],
    hy: NDArray[np.float64],
    station_name: str,
    sample_rate_hz: float,
    *,
    hz: NDArray[np.float64] | None = None,
    remote_hx: NDArray[np.float64] | None = None,
    remote_hy: NDArray[np.float64] | None = None,
    decimation_levels: Sequence[DecimationLevel] | None = None,
    bands: Sequence[FrequencyBand] | None = None,
    cleanup: bool = True,
) -> ZFileData:
    # Validate
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be a positive finite value")
    n = len(ex)
    for name, arr in [("ey", ey), ("hx", hx), ("hy", hy)]:
        if len(arr) != n:
            raise ValueError(f"ex has {n} samples but {name} has {len(arr)}")
    if hz is not None and len(hz) != n:
        raise ValueError(f"ex has {n} samples but hz has {len(hz)}")

    if (remote_hx is None) != (remote_hy is None):
        raise ValueError("remote_hx and remote_hy must be provided together")
    is_rr = remote_hx is not None and remote_hy is not None
    if is_rr:
        for name, arr in [("remote_hx", remote_hx), ("remote_hy", remote_hy)]:
            if len(arr) != n:
                raise ValueError(f"ex has {n} samples but {name} has {len(arr)}")

    levels = list(decimation_levels) if decimation_levels else _default_decimation(sample_rate_hz)
    band_list = list(bands) if bands else _default_bands(len(levels))

    from emtf._runner import Workspace as _Workspace
    ws = _Workspace(cleanup=cleanup, prefix="emtf_estimate_")
    try:
        # -- build channel list ------------------------------------------
        chan_names: list[str] = []
        chan_arrays = []
        for name, arr in [("Hx", hx), ("Hy", hy), ("Ex", ex), ("Ey", ey)]:
            chan_names.append(name)
            chan_arrays.append(np.asarray(arr, dtype=np.float64))
        if hz is not None:
            chan_names.append("Hz")
            chan_arrays.append(np.asarray(hz, dtype=np.float64))

        num_channels = len(chan_arrays)

        # -- compute default scales ---------------------------------------
        scales = []
        for cid in chan_names:
            if cid.upper().startswith("H"):
                scales.append(100.0)
            elif cid.upper().startswith("E"):
                scales.append(1000.0)
            else:
                scales.append(1.0)

        # -- write ASC + SP for local station -----------------------------
        from emtf.io.binary import write_asc, write_sp

        asc_name = f"{station_name}.asc"
        write_asc(ws.path / asc_name, chan_arrays, 1.0 / sample_rate_hz, scales=scales)
        write_sp(
            ws.path / f"{station_name}.sp",
            chan_names, 1.0 / sample_rate_hz,
            station_name=station_name,
            scales=scales,
        )

        # -- write remote ASC + SP if RR ----------------------------------
        rr_name: str | None = None
        if is_rr:
            rr_arrays = [np.asarray(remote_hx, dtype=np.float64),
                         np.asarray(remote_hy, dtype=np.float64)]
            rr_name = f"{station_name}_RR"
            write_asc(ws.path / f"{rr_name}.asc", rr_arrays, 1.0 / sample_rate_hz,
                      scales=[100.0, 100.0])
            write_sp(ws.path / f"{rr_name}.sp",
                     ["Hx", "Hy"], 1.0 / sample_rate_hz,
                     station_name=rr_name,
                     scales=[100.0, 100.0])

        # -- run dnff -a (ASCII mode) -------------------------------------
        from emtf._binaries import resolve_binary
        from emtf._config import render_paths, render_decset, render_pwset, render_dnff_stdin
        from emtf._runner import run as _run

        ws.write("paths.cfg", render_paths(
            data_dir=".", sp_dir=".", fc_dir="FC",
            decset="CF/decset.cfg", pwset="CF/pwset.cfg",
        ))
        ws.write("CF/decset.cfg", render_decset(levels))
        ws.write("CF/pwset.cfg", render_pwset(len(levels), num_channels, ar_order=3))
        (ws.path / "FC").mkdir(parents=True, exist_ok=True)

        binary = resolve_binary("dnff")
        dnff_inputs = [asc_name]
        if rr_name is not None:
            dnff_inputs.append(f"{rr_name}.asc")
        stdin = render_dnff_stdin(dnff_inputs)
        result = _run(binary, stdin=stdin, workspace=ws, args=["-A"])
        if result.returncode != 0:
            raise RuntimeError(f"dnff failed: {result.stderr}")

        # -- find FC files -----------------------------------------------
        fc_dir = ws.path / "FC"
        fc_files = sorted(
            path for path in fc_dir.iterdir()
            if path.is_file() and path.name.startswith(f"{station_name}.f")
        )
        if not fc_files:
            raise RuntimeError(f"No local FC files produced by dnff in {fc_dir}")
        remote_files: list[Path] = []
        if rr_name is not None:
            remote_files = sorted(
                path for path in fc_dir.iterdir()
                if path.is_file() and path.name.startswith(f"{rr_name}.f")
            )
            if not remote_files:
                raise RuntimeError(f"No remote-reference FC files produced by dnff in {fc_dir}")

        # -- run tranmt --------------------------------------------------
        options = ProcessingOptions(
            header="Robust Single station" if not is_rr else "Robust Remote Reference EMAP",
            robust=True,
            remote_reference=is_rr,
        )

        ws_tranmt = _tranmt(
            fc_files=fc_files,
            bands=band_list,
            station_name=station_name,
            options=options,
            num_channels=num_channels,
            remote_fc_files=remote_files if is_rr else None,
            remote_num_channels=2 if is_rr else None,
            workspace=ws,
            cleanup=False,  # ws handles cleanup
        )

        # -- parse Z file ------------------------------------------------
        output_pattern = "*.zrr" if is_rr else "*.zss"
        z_files = sorted((ws.path / "MT").glob(output_pattern))
        if not z_files:
            z_files = sorted((ws.path / "MT").glob("Z.*"))
        if not z_files:
            mode_name = "remote-reference" if is_rr else "single-station"
            raise RuntimeError(f"No {mode_name} Z files produced by tranmt in {ws.path / 'MT'}")

        zdata = ZFileData.read(z_files[0])
        if zdata.nbt == 0:
            raise RuntimeError(
                f"tranmt produced an empty Z file ({z_files[0].name}); "
                "select a longer overlapping time range or shorter FFT window"
            )
        return zdata

    finally:
        if cleanup:
            ws.cleanup_now()


def _default_decimation(sample_rate_hz: float) -> list[DecimationLevel]:
    """Choose a default decimation scheme based on sampling rate."""
    if abs(sample_rate_hz - 1.0) < 0.1:
        return list(_DECIMATION_1HZ)
    # For other rates, scale window_length proportionally
    base_len = 32.0 * sample_rate_hz
    return [
        DecimationLevel(nwin=128, window_length=base_len, decimation_factor=1, nfilt=1, fil_1=1.0),
        DecimationLevel(nwin=128, window_length=base_len, decimation_factor=4, nfilt=4,
                        fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705),
        DecimationLevel(nwin=128, window_length=base_len, decimation_factor=4, nfilt=4,
                        fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705),
        DecimationLevel(nwin=128, window_length=base_len, decimation_factor=4, nfilt=4,
                        fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705),
        DecimationLevel(nwin=128, window_length=base_len, decimation_factor=4, nfilt=4,
                        fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705),
        DecimationLevel(nwin=128, window_length=base_len, decimation_factor=4, nfilt=4,
                        fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705),
    ]


def _default_bands(num_levels: int) -> list[FrequencyBand]:
    """Build a default band table for the given number of decimation levels."""
    return standard_bands(num_levels)


def decimation_with_fft_window(
    sample_rate_hz: float,
    fft_window_size: int | None,
) -> list[DecimationLevel]:
    levels = _default_decimation(sample_rate_hz)
    if fft_window_size is None:
        return levels
    nwin = int(fft_window_size)
    if nwin <= 0:
        raise ValueError("fft_window_size must be positive")
    if nwin > _MAX_DNFF_WINDOW_SIZE:
        raise ValueError(
            f"EMTF dnff supports FFT短窗点数 <= {_MAX_DNFF_WINDOW_SIZE} "
            f"with the bundled binaries; got {nwin}"
        )
    overlap = float(nwin // 4)
    nfft = max(1, min(nwin // 4, nwin // 2))
    return [
        DecimationLevel(
            nwin=nwin,
            window_length=overlap,
            decimation_factor=level.decimation_factor,
            nfilt=level.nfilt,
            fil_1=level.fil_1,
            fil_2=level.fil_2,
            fil_3=level.fil_3,
            fil_4=level.fil_4,
            nfft=nfft,
            decimation_offset=level.decimation_offset,
        )
        for level in levels
    ]


def _decimation_from_window_size(window_size: int | None) -> list[DecimationLevel] | None:
    if window_size is None:
        return None
    nwin = int(window_size)
    if nwin <= 0:
        raise ValueError("window_size must be positive")
    if nwin > _MAX_DNFF_WINDOW_SIZE:
        raise ValueError(
            f"EMTF dnff supports window_size <= {_MAX_DNFF_WINDOW_SIZE} "
            f"with the bundled binaries; got {nwin}"
        )
    overlap = float(nwin // 4)
    nfuse = max(1, min(nwin // 4, nwin // 2))
    return [
        DecimationLevel(nwin=nwin, window_length=overlap, decimation_factor=1, nfilt=1, fil_1=1.0, nfft=nfuse),
        DecimationLevel(nwin=nwin, window_length=overlap, decimation_factor=4, nfilt=4,
                        fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705, nfft=nfuse),
        DecimationLevel(nwin=nwin, window_length=overlap, decimation_factor=4, nfilt=4,
                        fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705, nfft=nfuse),
        DecimationLevel(nwin=nwin, window_length=overlap, decimation_factor=4, nfilt=4,
                        fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705, nfft=nfuse),
        DecimationLevel(nwin=nwin, window_length=overlap, decimation_factor=4, nfilt=4,
                        fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705, nfft=nfuse),
        DecimationLevel(nwin=nwin, window_length=overlap, decimation_factor=4, nfilt=4,
                        fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705, nfft=nfuse),
    ]
