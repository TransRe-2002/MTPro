"""Domain types for EMTF processing parameters."""

from dataclasses import dataclass, field
import math
from typing import Optional, Sequence


# ── Configuration types ────────────────────────────────────────────────────


@dataclass
class DecimationLevel:
    """One cascade-decimation stage in ``decset.cfg``.

    Attributes
    ----------
    nwin:
        FFT window length in samples.  Also controls output frequency count
        (nwin // 2 + 1).
    window_length:
        Window length in seconds.
    decimation_factor:
        Factor relative to the previous level (1 = no decimation for level 1).
    nfilt:
        Number of anti-alias filter coefficients (0-4).
    fil_1 - fil_4:
        Filter coefficients (ignored if nfilt=0).
    nfft:
        FFT length in samples.
    decimation_offset:
        Offset parameter (typically 0).
    """

    nwin: int = 128
    window_length: float = 32.0
    decimation_factor: int = 4
    nfilt: int = 4
    fil_1: float = 0.0
    fil_2: float = 0.0
    fil_3: float = 0.0
    fil_4: float = 0.0
    nfft: int = 32
    decimation_offset: int = 0

    def to_cfg_line(self) -> str:
        """Format as one line for ``decset.cfg``."""
        return (
            f"{self.nwin}  {self.window_length}   "
            f"{self.decimation_factor}   "
            f"{self.decimation_offset}   0   7   4   "
            f"{self.nfft}   {self.nfilt}"
        )


@dataclass
class FrequencyBand:
    """One frequency band in ``bs.cfg``.

    Defines which Fourier coefficients (by decimation level and index)
    contribute to estimates at this band.
    """

    decimation_level: int
    """1-indexed decimation level whose FCs are used."""

    fc_start: int
    """First FC index (inclusive)."""

    fc_end: int
    """Last FC index (inclusive)."""


@dataclass
class ProcessingOptions:
    """Options controlling a ``tranmt`` run.

    Renders to ``options.cfg`` (options_SS.cfg or options_RR.cfg).
    """

    header: str = "Robust Single station"
    """Label written into output Z-file headers."""

    fc_dir: str = "FC"
    """Directory containing input FC files."""

    output_dir: str = "MT"
    """Directory where output Z-files are written."""

    robust: bool = True
    """Use robust (M-estimator) estimation.  ``False`` = ordinary LS."""

    remote_reference: bool = False
    """Enable remote-reference mode (needs remote FC files)."""

    e_field_ref: bool = False
    """Use E-field reference channels (rarely used)."""

    coherence_output: bool = False
    """Produce coherence-vs-set-number output."""

    coherence_params: str = "0. 0. 0. 0. 0"
    """Five floats: coh_target, cohp1, cohp2, coh_min, nu_min."""

    rotations: Sequence[float] = (0.0,)
    """Rotation angles to output (degrees E of geomagnetic north)."""


# ── Domain types ───────────────────────────────────────────────────────────


@dataclass
class Station:
    """Metadata for one MT station."""

    name: str = ""
    lat: float = 0.0
    lon: float = 0.0
    elevation: float = 0.0
    declination: float = 0.0


@dataclass
class TransferFunction:
    """Estimated impedance tensor for a single frequency band.

    Each attribute is a scalar for one band.  For multi-band results
    use :class:`ZFileData`, which contains full arrays and error/covariance data.
    """

    period: float = 0.0
    n_data: int = 0
    Zxx: complex = 0j
    Zxy: complex = 0j
    Zyx: complex = 0j
    Zyy: complex = 0j
    Zxx_err: float = 0.0
    Zxy_err: float = 0.0
    Zyx_err: float = 0.0
    Zyy_err: float = 0.0

    @property
    def apparent_resistivity_xy(self) -> float:
        """XY-mode apparent resistivity (ohm·m)."""
        if self.period <= 0:
            return float("nan")
        return 0.2 * self.period * abs(self.Zxy) ** 2

    @property
    def apparent_resistivity_yx(self) -> float:
        """YX-mode apparent resistivity (ohm·m)."""
        if self.period <= 0:
            return float("nan")
        return 0.2 * self.period * abs(self.Zyx) ** 2

    @property
    def phase_xy(self) -> float:
        """XY-mode phase in degrees."""
        return _emtf_phase_deg_scalar(self.Zxy)

    @property
    def phase_yx(self) -> float:
        """YX-mode phase in degrees."""
        return _emtf_phase_deg_scalar(self.Zyx)


def _emtf_phase_deg_scalar(z: complex) -> float:
    if not math.isfinite(z.real) or not math.isfinite(z.imag):
        return float("nan")
    if z.real != 0.0:
        return math.degrees(math.atan(z.imag / z.real))
    return math.degrees(math.atan2(z.imag, z.real))
