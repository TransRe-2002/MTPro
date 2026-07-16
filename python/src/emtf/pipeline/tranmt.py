"""Pipeline: ``tranmt`` — robust single-station / remote-reference TF estimation.

Wraps the Fortran ``tranmt`` (compiled as ``tranmtlr``) binary to
estimate magnetotelluric impedance tensors from Fourier coefficient
(``.fN``) files produced by ``dnff``.
"""

from pathlib import Path
from typing import Optional, Sequence

from emtf._binaries import resolve_binary
from emtf._config import (
    render_bands,
    render_options,
    render_paths,
    render_tranmt_control,
)
from emtf._runner import Workspace, run
from emtf.types import FrequencyBand, ProcessingOptions


def tranmt(
    fc_files: Sequence[str | Path],
    bands: Sequence[FrequencyBand],
    *,
    station_name: str = "station",
    options: Optional[ProcessingOptions] = None,
    num_channels: int = 5,
    remote_fc_files: Optional[Sequence[str | Path]] = None,
    fc_dir: str = "FC",
    workspace: Optional[Workspace] = None,
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
    is_rr = remote_fc_files is not None and len(remote_fc_files) > 0
    is_ss = not is_rr

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
    for src in fc_files:
        src = Path(src)
        dest = ws.path / fc_dir / src.name
        if src.resolve() != dest.resolve():
            ws.write_binary(f"{fc_dir}/{src.name}", src)

    n_fc = len(fc_files)
    runs = [
        (station_name, f"CF/{options_name}", 1, n_fc, num_channels,
         Path(fc_files[0]).name, False)
    ]
    if is_rr:
        rr_files = list(remote_fc_files)  # type: ignore[arg-type]
        n_rr = len(rr_files)
        runs.append(
            (f"{station_name}_RR", f"CF/{options_name}", 2, n_rr, num_channels,
             Path(rr_files[0]).name, False)
        )
    # For multi-file FC input, we list all files for the local station
    # in the tranmt.cfg.  This is a simplification — real surveys often
    # concatenate multiple FC files.  See EMTF CF/README for details.

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
