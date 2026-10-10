"""Pipeline: ``dnff`` — cascade decimation and Fourier transform.

Wraps the Fortran ``dnff`` binary: window time series, apply cascade
decimation with anti-alias filtering, compute FFT, and output Fourier
coefficient (``.fN``) files.

This is the first stage of the EMTF processing workflow.  Its output
feeds into ``tranmt`` for transfer function estimation.
"""

from pathlib import Path
from collections.abc import Sequence

from emtf.binaries import resolve_binary
from emtf.config import render_paths, render_decset, render_pwset, render_dnff_stdin
from emtf.runner import Workspace, run
from emtf.types import DecimationLevel


def dnff(
    input_files: Sequence[str | Path],
    decimation_levels: Sequence[DecimationLevel],
    num_channels: int = 5,
    *,
    data_dir: str = "data",
    sp_dir: str = "SP",
    fc_dir: str = "FC",
    ar_order: int = 3,
    ascii_mode: bool = False,
    workspace: Workspace | None = None,
    cleanup: bool = True,
) -> Workspace:
    """Run ``dnff`` — decimate + FFT time series → Fourier coefficient files.

    Parameters
    ----------
    input_files:
        Paths to binary (or ASCII, if ``ascii_mode=True``) time series files.
        These are copied into the workspace ``data/`` subdirectory.
    decimation_levels:
        Cascade decimation scheme, one :class:`DecimationLevel` per stage.
        A standard 1 Hz MT setup uses 6 levels with decimation factor 4.
    num_channels:
        Number of channels per file (default 5 for standard MT: Ex Ey Hx Hy Hz).
    data_dir:
        Workspace-relative directory for input time series.
    sp_dir:
        Workspace-relative directory for system parameter (``.sp``) files.
    fc_dir:
        Workspace-relative directory for Fourier coefficient output.
    ar_order:
        AR prewhitening filter order (default 3, appropriate for most MT data).
    ascii_mode:
        If ``True``, pass ``-a`` so dnff reads ASCII input instead of binary.
    workspace:
        Pre-configured :class:`Workspace`.  Created fresh if ``None``.
    cleanup:
        If ``True`` (default), remove the temp directory after the run.

    Returns
    -------
    :class:`Workspace` whose ``path`` contains the output FC files,
    typically in ``<workspace>/FC/``.
    """
    ws = workspace or Workspace(cleanup=cleanup, prefix="emtf_dnff_")

    # -- config files in CF/ ---------------------------------------------
    ws.write("paths.cfg", render_paths(
        data_dir=data_dir,
        sp_dir=sp_dir,
        decset="CF/decset.cfg",
        pwset="CF/pwset.cfg",
        fc_dir=fc_dir,
    ))
    ws.write("CF/decset.cfg", render_decset(decimation_levels))
    ws.write("CF/pwset.cfg", render_pwset(
        num_levels=len(decimation_levels),
        num_channels=num_channels,
        ar_order=ar_order,
    ))

    # -- copy data files into workspace -----------------------------------
    (ws.path / data_dir).mkdir(parents=True, exist_ok=True)
    (ws.path / sp_dir).mkdir(parents=True, exist_ok=True)
    (ws.path / fc_dir).mkdir(parents=True, exist_ok=True)
    for src in input_files:
        src = Path(src)
        ws.write_binary(f"{data_dir}/{src.name}", src)

    # -- run dnff ---------------------------------------------------------
    binary = resolve_binary("dnff")
    stdin = render_dnff_stdin(
        [f"{data_dir}/{Path(f).name}" for f in input_files],
        continue_after_each=True,
    )
    args = ["-A"] if ascii_mode else []
    result = run(binary, stdin=stdin, workspace=ws, args=args)

    if result.returncode != 0:
        ws.cleanup_now()
        raise RuntimeError(f"dnff failed (exit {result.returncode}):\n{result.stderr}")

    return ws


def dnff_from_arrays(
    channels: list,
    dt: float,
    decimation_levels: Sequence[DecimationLevel],
    *,
    channel_ids: list[str] | None = None,
    station_name: str = "STATION",
    scales: Sequence[float] | None = None,
    latitude: float = 0.0,
    longitude: float = 0.0,
    elevation: float = 0.0,
    declination: float = 0.0,
    line_lengths_km: Sequence[float] | None = None,
    ar_order: int = 3,
    workspace: Workspace | None = None,
    cleanup: bool = True,
) -> Workspace:
    """Run ``dnff`` directly from numpy arrays — write ASC + SP, then run ``dnff -a``.

    This is the full MATLAB-equivalent path::

        numpy arrays → write_asc() + write_sp() → dnff -a → FC files

    Parameters
    ----------
    channels:
        One 1-D float64 array per channel (e.g. [hx, hy, hz, ex, ey]).
    dt:
        Sampling interval in seconds.
    decimation_levels, ar_order, workspace, cleanup:
        Same as :func:`dnff`.
    channel_ids:
        Per-channel identifiers, e.g. ``["Hx", "Hy", "Hz", "Ex", "Ey"]``.
    station_name:
        Station name for the ``.sp`` header and output file names.
    scales:
        Per-channel integer scale factors for the ASC file.  Default:
        100 for H channels, 1000 for E channels.
    latitude, longitude, elevation, declination:
        Station metadata for the ``.sp`` file.
    line_lengths_km:
        Per-channel electric dipole lengths (km).  Ignored for H channels.

    Returns
    -------
    :class:`Workspace` with FC output in ``<workspace>/FC/``.
    """
    from emtf.io.binary import write_asc, write_sp

    nch = len(channels)
    if channel_ids is None:
        channel_ids = [f"CH{ich + 1}" for ich in range(nch)]

    if scales is None:
        scales = []
        for cid in channel_ids:
            if cid.upper().startswith("H"):
                scales.append(100.0)
            elif cid.upper().startswith("E"):
                scales.append(1000.0)
            else:
                scales.append(1.0)

    ws = workspace or Workspace(cleanup=cleanup, prefix="emtf_dnff_")

    # Write ASC data file
    data_name = f"{station_name}.asc"
    write_asc(
        ws.path / data_name,
        list(channels),
        dt,
        scales=scales,
    )

    # Write SP file
    sp_name = f"{station_name}.sp"
    write_sp(
        ws.path / sp_name,
        channel_ids,
        dt,
        station_name=station_name,
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        declination=declination,
        scales=scales,
        line_lengths_km=line_lengths_km,
    )

    # Write config files
    ws.write("paths.cfg", render_paths(
        data_dir=".",
        sp_dir=".",
        decset="CF/decset.cfg",
        pwset="CF/pwset.cfg",
        fc_dir="FC",
    ))
    ws.write("CF/decset.cfg", render_decset(decimation_levels))
    ws.write("CF/pwset.cfg", render_pwset(
        num_levels=len(decimation_levels),
        num_channels=nch,
        ar_order=ar_order,
    ))
    (ws.path / "FC").mkdir(parents=True, exist_ok=True)

    # Run dnff -a (ASCII mode)
    binary = resolve_binary("dnff")
    stdin = render_dnff_stdin([data_name], continue_after_each=False)
    result = run(binary, stdin=stdin, workspace=ws, args=["-A"])

    if result.returncode != 0:
        ws.cleanup_now()
        raise RuntimeError(f"dnff failed (exit {result.returncode}):\n{result.stderr}")

    return ws
