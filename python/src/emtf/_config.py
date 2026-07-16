"""Generate EMTF configuration files from Python objects.

Each Fortran program reads ``.cfg`` files that control its behaviour.
This module renders Python data as config file contents, to be written
into a :class:`Workspace` before invoking the binary.
"""

from textwrap import dedent
from typing import Sequence

from emtf.types import DecimationLevel, FrequencyBand, ProcessingOptions


def render_paths(
    data_dir: str = "data",
    sp_dir: str = "SP",
    br_dir: str = "data/BR",
    decset: str = "CF/decset.cfg",
    pwset: str = "CF/pwset.cfg",
    fc_dir: str = "FC",
    options_ss: str = "CF/options_SS.cfg",
    options_rr: str = "CF/options_RR.cfg",
    bs_file: str = "CF/bs.cfg",
) -> str:
    """Render a ``paths.cfg`` file for ``dnff``."""
    return "\n".join([
        data_dir, sp_dir, br_dir,
        decset, pwset, fc_dir,
        options_ss, options_rr, bs_file,
    ]) + "\n"


def render_decset(levels: Sequence[DecimationLevel]) -> str:
    """Render a ``decset.cfg`` — cascade decimation scheme."""
    n = len(levels)
    if n < 2:
        raise ValueError("EMTF dnff requires at least two decimation levels")
    offset = 0
    out = [f"{n}  {offset}      # of decimation level, & decimation offset"]
    for lvl in levels:
        out.append(lvl.to_cfg_line())
        coeffs = [lvl.fil_1, lvl.fil_2, lvl.fil_3, lvl.fil_4][: max(0, int(lvl.nfilt))]
        out.append("  ".join(f"{coeff:g}" for coeff in coeffs))
    return "\n".join(out) + "\n"


def render_pwset(
    num_levels: int,
    num_channels: int,
    ar_order: int = 3,
) -> str:
    """Render a ``pwset.cfg`` — prewhitening AR filter order per level."""
    out = [str(num_levels)]
    for _ in range(num_levels):
        out.append(str(ar_order))
    return "\n".join(out) + "\n"


def render_bands(bands: Sequence[FrequencyBand]) -> str:
    """Render a ``bs.cfg`` — frequency band definitions."""
    out = [str(len(bands))]
    for b in bands:
        out.append(f"{b.decimation_level}  {b.fc_start}  {b.fc_end}")
    return "\n".join(out) + "\n"


def render_options(opts: ProcessingOptions) -> str:
    """Render an ``options.cfg`` for ``tranmt``."""
    return dedent(f"""\
    {opts.header}
    {opts.fc_dir}
    {opts.output_dir}
    CF/bs.cfg
    {'y' if opts.robust else 'n'}                robust
    {'y' if opts.remote_reference else 'n'}                remote reference
    {'y' if opts.e_field_ref else 'n'}                e field ref
    {'y' if opts.coherence_output else 'n'}                output coherence vs set no.
    {opts.coherence_params}    coherence sorting parameters
    {len(opts.rotations)}                # of rotations
    """ + "\n".join(f"{angle:.3f}" for angle in opts.rotations) + "\n")


def render_tranmt_control(
    runs: Sequence[tuple[str, str, int, int, str, bool]],
) -> str:
    """Render a ``tranmt.cfg`` — control file listing processing runs.

    Each element: ``(station_name, options_cfg, num_stations, num_fc_files,
    num_channels, fc_file, continue_flag)``.
    """
    out = []
    for i, (name, opt_cfg, n_sta, n_fc, n_ch, fc_file, cont) in enumerate(runs):
        out.append(name)
        out.append(opt_cfg)
        out.append(str(n_sta))
        out.append(f"{n_fc}  {n_ch}")
        out.append(fc_file)
        out.append("y" if cont else "n")
    return "\n".join(out) + "\n"


def render_dnff_stdin(
    input_files: list[str],
    continue_after_each: bool = True,
) -> str:
    """Build stdin for dnff's interactive prompts.

    dnff prompts for each input file name and asks whether to process
    another.  This function builds the answers.
    """
    lines = []
    for i, fname in enumerate(input_files):
        lines.append(fname)
        if i < len(input_files) - 1:
            lines.append("y" if continue_after_each else "n")
        else:
            lines.append("n")
    return "\n".join(lines) + "\n"
