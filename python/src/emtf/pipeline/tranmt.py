"""Pipeline: ``tranmt`` — robust single-station / remote-reference TF estimation.

Wraps the Fortran ``tranmt`` (compiled as ``tranmtlr``) binary to
estimate magnetotelluric impedance tensors from Fourier coefficient
(``.fN``) files produced by ``dnff``.
"""

from pathlib import Path
from collections.abc import Sequence

from emtf.binaries import resolve_binary
from emtf.config import (
    render_bands,
    render_options,
    render_paths,
    render_tranmt_control,
)
from emtf.runner import Workspace, run
from emtf.types import FrequencyBand, ProcessingOptions


def tranmt(
    fc_files: Sequence[str | Path],
    bands: Sequence[FrequencyBand],
    *,
    station_name: str = "station",
    options: ProcessingOptions | None = None,
    num_channels: int = 5,
    remote_fc_files: Sequence[str | Path] | None = None,
    remote_num_channels: int | None = None,
    fc_dir: str = "FC",
    workspace: Workspace | None = None,
    cleanup: bool = True,
) -> Workspace:
    """Run ``tranmt`` to estimate transfer functions from FC files.

    Supports both single-station and remote-reference modes.

    Parameters
    ----------
    fc_files:
        Paths to FC (``.fN``) files for the local station, from ``dnff``.
    bands:
        Frequency bands over which to estimate transfer functions.
    station_name:
        Station identifier used in output file names and Z-file headers.
    options:
        :class:`ProcessingOptions` controlling robust vs LS, SS vs RR, etc.
        Default: robust single-station with one rotation at 0°.
    num_channels:
        Number of channels per FC file (default 5).
    remote_fc_files:
        If given, enables remote-reference mode.  These FC files serve
        as the remote station.
    remote_num_channels:
        Number of channels in each remote FC file.  Defaults to
        ``num_channels``.  The high-level remote-reference estimator writes
        a two-channel remote Hx/Hy FC file, so it passes ``2`` here.
    fc_dir:
        Workspace-relative directory where FC files are copied.
    workspace:
        Pre-configured :class:`Workspace`.  Created fresh if ``None``.
    cleanup:
        If ``True`` (default), remove workspace after the run.

    Returns
    -------
    :class:`Workspace` whose ``path`` contains the output Z-files,
    typically in ``<workspace>/MT/``.
    """
    opts = options or ProcessingOptions()
    local_files = [Path(src) for src in fc_files]
    remote_files = [Path(src) for src in remote_fc_files or []]
    is_rr = bool(remote_files)
    remote_channel_count = num_channels if remote_num_channels is None else remote_num_channels

    if not local_files:
        raise ValueError("tranmt requires at least one local FC file")
    if opts.remote_reference and not is_rr:
        raise ValueError("remote_reference=True requires at least one remote FC file")
    if num_channels < 1:
        raise ValueError("num_channels must be positive")
    if is_rr and remote_channel_count < 1:
        raise ValueError("remote_num_channels must be positive")

    all_files = [*local_files, *remote_files]
    missing_files = [str(src) for src in all_files if not src.is_file()]
    if missing_files:
        raise FileNotFoundError(f"FC input file(s) not found: {', '.join(missing_files)}")
    duplicate_names = sorted({src.name for src in all_files if sum(f.name == src.name for f in all_files) > 1})
    if duplicate_names:
        raise ValueError(
            "local and remote FC files must have distinct basenames: "
            + ", ".join(duplicate_names)
        )

    if is_rr:
        opts.remote_reference = True

    ws = workspace or Workspace(cleanup=cleanup, prefix="emtf_tranmt_")

    # -- configuration files ----------------------------------------------
    options_name = "options_RR.cfg" if is_rr else "options_SS.cfg"
    ws.write("paths.cfg", render_paths(
        data_dir=fc_dir,  # For tranmt context, "data" = FC directory
        sp_dir="SP",
        br_dir="data/BR",
        decset="CF/decset.cfg",
        pwset="CF/pwset.cfg",
        fc_dir=fc_dir,
        options_ss="CF/options_SS.cfg",
        options_rr="CF/options_RR.cfg",
        bs_file="CF/bs.cfg",
    ))
    ws.write("CF/bs.cfg", render_bands(bands))
    opts.fc_dir = fc_dir
    opts.output_dir = "MT"
    ws.write(f"CF/{options_name}", render_options(opts))

    # -- copy FC files ----------------------------------------------------
    (ws.path / fc_dir).mkdir(parents=True, exist_ok=True)
    for src in all_files:
        dest = ws.path / fc_dir / src.name
        if src.resolve() != dest.resolve():
            ws.write_binary(f"{fc_dir}/{src.name}", src)

    groups: list[tuple[Sequence[Path], int]] = [(local_files, num_channels)]
    if is_rr:
        groups.append((remote_files, remote_channel_count))
    runs = [
        (station_name, f"CF/{options_name}", groups, False)
    ]

    ws.write("tranmt.cfg", render_tranmt_control(runs))
    ws.write("MT/.gitkeep", "")

    # -- run --------------------------------------------------------------
    binary = resolve_binary("tranmt")
    stdin = "tranmt.cfg\n"
    result = run(binary, stdin=stdin, workspace=ws)

    if result.returncode != 0:
        ws.cleanup_now()
        raise RuntimeError(f"tranmt failed (exit {result.returncode}):\n{result.stderr}")

    return ws
