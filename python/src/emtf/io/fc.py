"""IO: Fourier coefficient (``.fN``) binary file reader.

Ported from ``EMTF/matlab/FC/{fcOpen.m,loadArrayRecord.m,unpack.m}``.
FC files are binary Fortran outputs produced by ``dnff``.  Each file
contains complex FCs for one station, organized by decimation level,
frequency index, and time window (segment).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


_M = 65536
_Mo2 = _M // 2


def _unpack(ifc: NDArray[np.int32], scales: NDArray[np.float32]) -> NDArray[np.complex128]:
    """Unpack packed integer FCs into complex values with scaling.

    Port of ``EMTF/matlab/FC/unpack.m``.

    Parameters
    ----------
    ifc:
        Packed integers, shape ``(nch, nsets)`` — the raw ``fread(id, [nch+1, nsets], 'integer*4')``
        output with the set-number row already stripped off.
    scales:
        Channel scale factors, shape ``(nch,)``, from the FC file header.

    Returns
    -------
    Complex FCs, shape ``(nch, nsets)``.
    """
    it2 = np.floor(ifc.astype(np.float64) / _M)
    it1 = ifc.astype(np.float64) - _M * it2 - _Mo2
    k1 = it1 - 8 * np.floor(it1 / 8)
    k2 = it2 - 8 * np.floor(it2 / 8)
    it1 = (it1 - k1) / _Mo2 + 0.000122
    it2 = (it2 - k2) / _Mo2 + 0.000122
    fc = it1 * (10.0 ** k1) + 1j * it2 * (10.0 ** k2)
    # Apply per-channel scale
    return fc * scales[:, np.newaxis]


@dataclass
class FCOutput:
    """Header and data from a Fourier coefficient (``.fN``) file.

    This is a concrete reader that stores the file path and provides
    methods to load FC spectra for individual bands.
    """

    file: str = ""
    """Path to the FC file."""

    endian: str = "little"
    """Byte order: ``"little"`` or ``"big"``."""

    lpack: bool = True
    """Whether FCs are packed (True, default) or raw float pairs (False)."""

    nch: int = 0
    """Number of channels."""

    nd: int = 0
    """Number of decimation levels."""

    nf: int = 0
    """Number of frequency bands per decimation level."""

    decs: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros((2, 1), dtype=np.int32)
    )
    """Decimation parameters per level, shape ``(2, nd)``."""

    drs: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64)
    )
    """Decimation rates, shape ``(nd,)``."""

    chid: list[str] = field(default_factory=list)
    """Channel identifiers."""

    orient: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((2, 1), dtype=np.float64)
    )
    """Channel orientations, shape ``(2, nch)``."""

    stcor: tuple[float, float] = (0.0, 0.0)
    """Station coordinates (latitude, longitude)."""

    decl: float = 0.0
    """Magnetic declination in degrees."""

    start_dec: list[int] = field(default_factory=list)
    """File byte offset for the start of each decimation level."""

    start_freqs: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros((1, 1), dtype=np.int32)
    )
    """File byte offsets for (level, freq_idx), shape ``(nd, nf)``. 1-based indexing."""

    i_sets: list[NDArray[np.int32]] = field(default_factory=list)
    """Set numbers per decimation level, ``list[ndarray]``, one array per level."""

    @classmethod
    def read(cls, path: str | Path, lpack: bool = True, endian: str = "little") -> "FCOutput":
        """Read FC file header metadata.

        Ported from ``EMTF/matlab/FC/fcOpen.m``.
        """
        from emtf.io.fc_reader import _read_fc_open
        return _read_fc_open(path, lpack=lpack, endian=endian)

    def load_spectrum(
        self,
        decimation_level: int,
        freq_idx: int,
    ) -> NDArray[np.complex128]:
        """Load FCs for one ``(decimation_level, freq_idx)`` combination.

        Parameters
        ----------
        decimation_level:
            1-indexed decimation level to read.
        freq_idx:
            1-indexed frequency index within the decimation level.

        Returns
        -------
        Complex FC array of shape ``(nch, nsegments)``.
        """
        id_zero = decimation_level - 1
        if_zero = freq_idx - 1
        if not (0 <= id_zero < self.nd and 0 <= if_zero < self.nf):
            raise ValueError("FC level/frequency index out of range")
        if self.start_freqs[id_zero, if_zero] < 0:
            raise ValueError("FC level/frequency not present in file")
        nch = self.nch

        dt = np.dtype(">i4") if self.endian == "big" else np.dtype("<i4")
        dt_f4 = np.dtype(">f4") if self.endian == "big" else np.dtype("<f4")

        with open(self.file, "rb") as fh:
            if self.lpack:
                irecl = nch + 1
            else:
                irecl = 2 * nch + 1

            fh.seek(int(self.start_freqs[id_zero, if_zero]))
            head = np.frombuffer(fh.read(4 * irecl), dtype=dt)
            nsets = int(head[2])
            # The scale record contains nch floats followed by record padding;
            # unlike the data records it does NOT start with a set number.
            scale_record = fh.read(4 * irecl)
            scales = np.frombuffer(scale_record[:4 * nch], dtype=dt_f4) / 1000.0

            if self.lpack:
                # Read packed integers
                ifc = np.frombuffer(fh.read(4 * (nch + 1) * nsets), dtype=dt)
                ifc = ifc.reshape((nch + 1, nsets), order="F")
                # Row 0 = set numbers, rows 1:nch+1 = packed FC values
                return _unpack(ifc[1:, :], scales)
            else:
                # Each record starts with an integer set number, followed by
                # interleaved real/imaginary values normalised by the scales.
                raw = np.frombuffer(fh.read(4 * irecl * nsets), dtype=dt_f4)
                fc = raw.reshape((irecl, nsets), order="F")[1:]
                return (fc[0::2, :] + 1j * fc[1::2, :]) * scales[:, None]


def read_fc_spectrum(
    path: str | Path,
    decimation_level: int,
    freq_idx: int,
    lpack: bool = True,
    endian: str = "little",
) -> NDArray[np.complex128]:
    """Quick-read FCs for one frequency from a ``.fN`` file.

    Returns a complex array of shape ``(nch, nsegments)``.
    """
    header = FCOutput.read(path, lpack=lpack, endian=endian)
    return header.load_spectrum(decimation_level, freq_idx)
