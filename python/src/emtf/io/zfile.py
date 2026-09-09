"""IO: Z-file (transfer function) reader and writer.

Ported from ``EMTF/matlab/ZPLT/{readZfile.m,writeZfile.m}``.
Z-files are ASCII output files produced by ``tranmt`` and ``multmtrn``.
They contain impedance tensors, vertical-field transfer functions, and
error covariance matrices for each frequency band.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_RNAN: float = -999.0  # EMTF sentinel for missing data


def emtf_phase_deg(z: NDArray[np.complex128] | complex) -> NDArray[np.float64]:
    """EMTF MATLAB phase convention: degrees(atan(imag(z) / real(z)))."""
    arr = np.asarray(z, dtype=np.complex128)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(arr.real) & np.isfinite(arr.imag)
    nonzero_real = valid & (arr.real != 0.0)
    out[nonzero_real] = np.degrees(
        np.arctan(arr.imag[nonzero_real] / arr.real[nonzero_real])
    )
    zero_real = valid & ~nonzero_real
    out[zero_real] = np.degrees(np.arctan2(arr.imag[zero_real], arr.real[zero_real]))
    return out


@dataclass
class ZFileData:
    """Contents of an EMTF Z-file (impedance transfer function output).

    Mirrors the MATLAB structure from ``readZfile.m``.
    """

    tf: NDArray[np.complex128] = field(
        default_factory=lambda: np.zeros((2, 0, 0), dtype=np.complex128)
    )
    """Transfer functions, shape ``(2, nche, nbt)`` — [Hx, Hy] × predicted channels × bands."""

    sig_s: NDArray[np.complex128] = field(
        default_factory=lambda: np.zeros((2, 2, 0), dtype=np.complex128)
    )
    """Inverse signal covariance, shape ``(2, 2, nbt)``."""

    sig_e: NDArray[np.complex128] = field(
        default_factory=lambda: np.zeros((0, 0, 0), dtype=np.complex128)
    )
    """Residual covariance, shape ``(nche, nche, nbt)``."""

    periods: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    """Periods in seconds, shape ``(nbt,)``."""

    ndf: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    """Number of data points per band, shape ``(nbt,)``."""

    stcor: tuple[float, float] = (0.0, 0.0)
    """Station coordinates (latitude, longitude)."""

    declination: float = 0.0
    """Magnetic declination in degrees."""

    orient: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((2, 4), dtype=np.float64)
    )
    """Channel orientations, shape ``(2, nch)``."""

    chead: str = ""
    """Comment/header line (processing mode, e.g. "Robust Remote Reference")."""

    stname: str = ""
    """Station name."""

    ibandlim: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros((2, 0), dtype=np.int32)
    )
    """Frequency band limits per band, shape ``(2, nbt)``. [start, end] FC numbers."""

    level: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    """Decimation level per band, shape ``(nbt,)``."""

    samp_rate: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    """Sampling rate per band (Hz), shape ``(nbt,)``."""

    nch: int = 4
    """Total number of channels."""

    nche: int = 2
    """Number of predicted channels (= nch - 2)."""

    nbt: int = 0
    """Number of frequency bands."""

    chid: list[str] = field(default_factory=list)
    """Channel IDs (e.g. "Hx", "Ey")."""

    sta: list[str] = field(default_factory=list)
    """Station abbreviation per channel."""

    # ── derived properties ─────────────────────────────────────────────

    @property
    def Zxx(self) -> NDArray[np.complex128]:
        """Zxx impedance component, shape ``(nbt,)``."""
        ex = self._predicted_index("EX", default=0)
        return self.tf[0, ex, :] if ex < self.tf.shape[1] else np.zeros(self.nbt, dtype=np.complex128)

    @property
    def Zxy(self) -> NDArray[np.complex128]:
        """Zxy impedance component, shape ``(nbt,)``."""
        ex = self._predicted_index("EX", default=0)
        return self.tf[1, ex, :] if ex < self.tf.shape[1] else np.zeros(self.nbt, dtype=np.complex128)

    @property
    def Zyx(self) -> NDArray[np.complex128]:
        """Zyx impedance component, shape ``(nbt,)``."""
        ey = self._predicted_index("EY", default=1)
        return self.tf[0, ey, :] if ey < self.tf.shape[1] else np.zeros(self.nbt, dtype=np.complex128)

    @property
    def Zyy(self) -> NDArray[np.complex128]:
        """Zyy impedance component, shape ``(nbt,)``."""
        ey = self._predicted_index("EY", default=1)
        return self.tf[1, ey, :] if ey < self.tf.shape[1] else np.zeros(self.nbt, dtype=np.complex128)

    @property
    def Tzx(self) -> NDArray[np.complex128] | None:
        """Tzx tipper component, or None if no predicted Hz channel exists."""
        hz = self._predicted_index("HZ", default=-1)
        return self.tf[0, hz, :] if hz >= 0 and hz < self.tf.shape[1] else None

    @property
    def Tzy(self) -> NDArray[np.complex128] | None:
        """Tzy tipper component, or None if no predicted Hz channel exists."""
        hz = self._predicted_index("HZ", default=-1)
        return self.tf[1, hz, :] if hz >= 0 and hz < self.tf.shape[1] else None

    def _predicted_index(self, prefix: str, default: int) -> int:
        """Return predicted-channel index for a channel prefix such as EX/EY/HZ."""
        prefix = prefix.upper()
        for i, chid in enumerate(self.chid[2:]):
            if chid.upper().startswith(prefix):
                return i
        return default

    def apparent_resistivity_xy(self) -> NDArray[np.float64]:
        """XY-mode apparent resistivity (ohm·m): ρ = 0.2 * T * |Zxy|²."""
        valid = self.periods > 0
        rho = np.full(self.nbt, np.nan)
        rho[valid] = 0.2 * self.periods[valid] * np.abs(self.Zxy[valid]) ** 2
        return rho

    def apparent_resistivity_yx(self) -> NDArray[np.float64]:
        """YX-mode apparent resistivity (ohm·m)."""
        valid = self.periods > 0
        rho = np.full(self.nbt, np.nan)
        rho[valid] = 0.2 * self.periods[valid] * np.abs(self.Zyx[valid]) ** 2
        return rho

    def phase_xy_deg(self) -> NDArray[np.float64]:
        """XY-mode impedance phase in degrees."""
        return emtf_phase_deg(self.Zxy)

    def phase_yx_deg(self) -> NDArray[np.float64]:
        """YX-mode impedance phase in degrees."""
        return emtf_phase_deg(self.Zyx)

    def _impedance_error_variance(self) -> NDArray[np.float64]:
        """Compute EMTF impedance error variance terms, shape (2, nche, nbt).

        This is the ``zerr`` variable from ``EMTF/matlab/ZPLT/ap_res.m``:
        ``s * e'``, where ``s`` is the inverse signal covariance diagonal and
        ``e`` is the residual covariance diagonal.
        """
        nbt = self.nbt
        nche = self.nche
        zvar = np.zeros((2, nche, nbt))
        for ib in range(nbt):
            s = np.real(np.diag(self.sig_s[:, :, ib]))  # (2,)
            e = np.real(np.diag(self.sig_e[:, :, ib]))  # (nche,)
            zvar[:, :, ib] = np.outer(s, e)
        return zvar

    def _impedance_errors(self) -> NDArray[np.float64]:
        """Compute one-sigma impedance errors for all elements.

        EMTF's MATLAB plotting code uses ``sqrt(zerr / 2)`` as the impedance
        uncertainty for apparent-resistivity and phase propagation.
        """
        zvar = np.maximum(self._impedance_error_variance(), 0.0)
        return np.sqrt(zvar / 2.0)

    def Zxy_err(self) -> NDArray[np.float64]:
        """Standard error of Zxy, shape (nbt,)."""
        zerr = self._impedance_errors()
        ex = self._predicted_index("EX", default=0)
        return zerr[1, ex, :]

    def Zyx_err(self) -> NDArray[np.float64]:
        """Standard error of Zyx, shape (nbt,)."""
        zerr = self._impedance_errors()
        ey = self._predicted_index("EY", default=1)
        return zerr[0, ey, :]

    def rho_xy_err(self) -> NDArray[np.float64]:
        """Standard error of XY apparent resistivity."""
        zse = self._impedance_errors()
        ex = self._predicted_index("EX", default=0)
        z = self.Zxy
        valid = (self.periods > 0) & (np.abs(z) > 0)
        out = np.full(self.nbt, np.nan)
        out[valid] = 2.0 * self.apparent_resistivity_xy()[valid] * zse[1, ex, valid] / np.abs(z[valid])
        return out

    def rho_yx_err(self) -> NDArray[np.float64]:
        """Standard error of YX apparent resistivity."""
        zse = self._impedance_errors()
        ey = self._predicted_index("EY", default=1)
        z = self.Zyx
        valid = (self.periods > 0) & (np.abs(z) > 0)
        out = np.full(self.nbt, np.nan)
        out[valid] = 2.0 * self.apparent_resistivity_yx()[valid] * zse[0, ey, valid] / np.abs(z[valid])
        return out

    def phi_xy_err_deg(self) -> NDArray[np.float64]:
        """Standard error of XY phase, degrees."""
        zse = self._impedance_errors()
        ex = self._predicted_index("EX", default=0)
        return np.degrees(zse[1, ex, :] / np.maximum(np.abs(self.Zxy), 1e-20))

    def phi_yx_err_deg(self) -> NDArray[np.float64]:
        """Standard error of YX phase, degrees."""
        zse = self._impedance_errors()
        ey = self._predicted_index("EY", default=1)
        return np.degrees(zse[0, ey, :] / np.maximum(np.abs(self.Zyx), 1e-20))

    @classmethod
    def read(cls, path: str | Path) -> "ZFileData":
        """Read an EMTF Z-file from disk.

        Handles both standard (station:) and legacy (ZMM) header formats.
        Ported from ``EMTF/matlab/ZPLT/readZfile.m``.
        """
        path = Path(path)
        with open(path, "r") as fh:
            lines = fh.readlines()

        if len(lines) < 6:
            raise ValueError(
                f"Malformed EMTF Z-file {path}: expected at least six header lines, got {len(lines)}"
            )

        data = cls()
        data.chead = lines[2].rstrip("\n").rstrip("\r").strip()

        # Station name: line 4 (0-indexed: lines[3])
        st_match = re.search(r"station\s*:?\s*(\S.*)", lines[3], re.IGNORECASE)
        if st_match:
            data.stname = st_match.group(1).strip()

        # Coordinate line
        coord = re.search(
            r"coordinate\s+([-\d.]+)\s+([-\d.]+)\s+declination\s+([-\d.]+)",
            lines[4],
        )
        if coord:
            data.stcor = (float(coord.group(1)), float(coord.group(2)))
            data.declination = float(coord.group(3))

        # Number of channels / frequencies
        counts = re.search(
            r"number of channels\s+(-?\d+)\s+number of frequencies\s+(-?\d+)",
            lines[5],
        )
        if counts is None:
            raise ValueError(
                f"Malformed EMTF Z-file {path}: missing channel/frequency count header"
            )
        data.nch = int(counts.group(1))
        data.nbt = int(counts.group(2))
        if data.nch < 4:
            raise ValueError(
                f"Malformed EMTF Z-file {path}: expected at least four channels, got {data.nch}"
            )
        if data.nbt < 0:
            raise ValueError(
                f"Malformed EMTF Z-file {path}: frequency count cannot be negative ({data.nbt})"
            )
        data.nche = data.nch - 2

        min_header_lines = 7 + data.nch
        if len(lines) < min_header_lines:
            raise ValueError(
                f"Malformed EMTF Z-file {path}: incomplete channel-orientation header"
            )

        # Channel orientations
        data.chid = []
        data.sta = []
        orient_list: list[tuple[float, float]] = []
        line_idx = 7  # skip "orientations and tilts" line
        for _ in range(data.nch):
            parts = lines[line_idx].split()
            if len(parts) >= 5:
                orient_list.append((float(parts[1]), float(parts[2])))
                data.sta.append(parts[3])
                data.chid.append(parts[4])
            line_idx += 1
        data.orient = np.array(orient_list, dtype=np.float64).T

        # Skip blank line
        if line_idx < len(lines) and lines[line_idx].strip() == "":
            line_idx += 1

        nche = data.nche
        nbt = data.nbt
        if nbt == 0:
            # Empty Z-file (template / header-only)
            return data
        periods = np.zeros(nbt, dtype=np.float64)
        ndf_arr = np.zeros(nbt, dtype=np.int32)
        ibandlim_arr = np.zeros((2, nbt), dtype=np.int32)
        level_arr = np.zeros(nbt, dtype=np.int32)
        samp_rate_arr = np.zeros(nbt, dtype=np.float64)
        tf_arr = np.zeros((2, nche, nbt), dtype=np.complex128)
        sig_s_arr = np.zeros((2, 2, nbt), dtype=np.complex128)
        sig_e_arr = np.zeros((nche, nche, nbt), dtype=np.complex128)

        for ib in range(nbt):
            # Period line
            period_match = re.search(r"period\s*:\s*([\d.Ee+\-]+)", lines[line_idx])
            if period_match:
                periods[ib] = float(period_match.group(1))
            band_match = re.search(
                r"decimation level\s+(\d+)\s+.*band from\s+(\d+)\s+to\s+(\d+)",
                lines[line_idx],
            )
            if band_match:
                level_arr[ib] = int(band_match.group(1))
                ibandlim_arr[0, ib] = int(band_match.group(2))
                ibandlim_arr[1, ib] = int(band_match.group(3))
            line_idx += 1

            # NDF + sampling rate line
            ndf_match = re.search(
                r"number of data point\s+(\d+)\s+sampling freq\.\s+([\d.]+)\s+Hz",
                lines[line_idx],
            )
            if ndf_match:
                ndf_arr[ib] = int(ndf_match.group(1))
                samp_rate_arr[ib] = float(ndf_match.group(2))
            line_idx += 1

            # "Transfer Functions" header
            line_idx += 1

            # Read TF values: Fortran writes 4 E12.4 values per line.
            # Total: 2 rows (Hx, Hy) × nche channels × 2 (re, im) values.
            tf_flat: list[tuple[int, int, float]] = []  # (irow, icol, value)
            for irow in range(2):
                for j in range(nche):
                    # Need 2 values (re, im) per (row, col) = 2*nche values total per row
                    pass  # we read below

            # Actually read lines until we have 4*nche real values total
            # (2 rows × nche channels × 2 re/im)
            needed_reals = 4 * nche  # 2 rows × 2*nche
            tf_buffer: list[float] = []
            while len(tf_buffer) < needed_reals:
                if line_idx >= len(lines):
                    break
                ln = lines[line_idx].strip()
                line_idx += 1
                # Check for sentinel (next section header)
                if ln.startswith("Inverse Coherent") or ln.startswith("Residual Covariance"):
                    break
                tf_buffer.extend(_parse_fortran_e12_line(ln))
            # MATLAB readZfile.m reads a 4 x nche matrix and maps each
            # predicted channel as [re(Hx), im(Hx), re(Hy), im(Hy)].
            idx_r = 0
            for j in range(nche):
                if idx_r + 3 < len(tf_buffer):
                    tf_arr[0, j, ib] = complex(tf_buffer[idx_r], tf_buffer[idx_r + 1])
                    tf_arr[1, j, ib] = complex(tf_buffer[idx_r + 2], tf_buffer[idx_r + 3])
                idx_r += 4

            # Now line_idx should be at "Inverse Coherent Signal Power Matrix"
            # Skip one extra header line if we're on it
            if line_idx < len(lines) and lines[line_idx].strip().startswith("Inverse Coherent"):
                pass  # already positioned correctly, need to skip 2 total

            # "Inverse Coherent Signal Power Matrix" header (2 lines total)
            # The TF reader stopped at the first "Inverse Coherent" line.
            # Skip the second header line to reach the S matrix data.
            if line_idx < len(lines) and "Inverse Coherent" in lines[line_idx]:
                line_idx += 1  # skip second "Inverse Coherent Signal Power Matrix" line

            # Read signal covariance: lower triangle of 2×2 packed Hermitian
            # = 3 complex = 6 real values
            s_needed = 6
            s_buffer: list[float] = []
            while len(s_buffer) < s_needed and line_idx < len(lines):
                ln = lines[line_idx].strip()
                line_idx += 1
                if ln == "":
                    continue
                s_buffer.extend(_parse_fortran_e12_line(ln))
            idx_s = 0
            for k in range(2):
                for l in range(k + 1):
                    if idx_s + 1 < len(s_buffer):
                        v = complex(s_buffer[idx_s], s_buffer[idx_s + 1])
                        sig_s_arr[k, l, ib] = v
                        sig_s_arr[l, k, ib] = v.conjugate()
                    idx_s += 2

            # "Residual Covariance" header (2 lines)
            while line_idx < len(lines) and "Residual Covariance" not in lines[line_idx]:
                line_idx += 1
            line_idx += 1  # skip second header line

            # Read residual: lower triangle of nche×nche packed Hermitian
            # = nche*(nche+1)/2 complex = nche*(nche+1) real values
            e_needed = nche * (nche + 1)
            e_buffer: list[float] = []
            while len(e_buffer) < e_needed and line_idx < len(lines):
                ln = lines[line_idx].strip()
                line_idx += 1
                if ln == "":
                    continue
                e_buffer.extend(_parse_fortran_e12_line(ln))
            idx_e = 0
            for k in range(nche):
                for l in range(k + 1):
                    if idx_e + 1 < len(e_buffer):
                        v = complex(e_buffer[idx_e], e_buffer[idx_e + 1])
                        sig_e_arr[k, l, ib] = v
                        sig_e_arr[l, k, ib] = v.conjugate()
                    idx_e += 2

            # Skip blank line between bands
            while line_idx < len(lines) and lines[line_idx].strip() == "":
                line_idx += 1

        data.periods = periods
        data.ndf = ndf_arr
        data.ibandlim = ibandlim_arr
        data.level = level_arr
        data.samp_rate = samp_rate_arr
        data.tf = tf_arr
        data.sig_s = sig_s_arr
        data.sig_e = sig_e_arr

        return data

    def write(self, path: str | Path) -> None:
        """Write data to an EMTF-format Z-file.

        Ported from ``EMTF/matlab/ZPLT/writeZfile.m``.
        """
        path = Path(path)
        with open(path, "w") as fh:
            fh.write(" **** IMPEDANCE IN MEASUREMENT COORDINATES ****\n")
            fh.write(" ********** WITH FULL ERROR COVARINCE**********\n")
            fh.write(f"{self.chead}\n")
            fh.write(f"station    :{self.stname:>20}\n")
            fh.write(
                f"coordinate {self.stcor[0]:9.3f} {self.stcor[1]:9.3f}  "
                f"declination {self.declination:8.2f}\n"
            )
            fh.write(f"number of channels {self.nch:3d}   number of frequencies {self.nbt:4d}\n")
            fh.write(" orientations and tilts of each channel \n")

            for i in range(self.nch):
                sta_str = self.sta[i] if i < len(self.sta) else self.stname[:3]
                chid_str = self.chid[i] if i < len(self.chid) else "   "
                fh.write(
                    f"{i + 1:5d} {self.orient[0, i]:8.2f} {self.orient[1, i]:8.2f} "
                    f"{sta_str:3s} {chid_str:6s}\n"
                )
            fh.write("\n")

            nche = self.nche
            for ib in range(self.nbt):
                fh.write(
                    f"period : {self.periods[ib]:12.5f}    decimation level "
                    f"{self.level[ib]:3d}    freq. band from "
                    f"{self.ibandlim[0, ib]:4d} to {self.ibandlim[1, ib]:4d}\n"
                )
                fh.write(
                    f"number of data point {self.ndf[ib]:6d} "
                    f"sampling freq. {self.samp_rate[ib]:7.3f} Hz\n"
                )
                fh.write(" Transfer Functions\n")

                # Transfer Functions: one row per predicted channel:
                # [re(Hx), im(Hx), re(Hy), im(Hy)].
                for j in range(nche):
                    tf_flat: list[str] = []
                    for irow in range(2):
                        z = self.tf[irow, j, ib]
                        tf_flat.append(_fmt_e12(z.real))
                        tf_flat.append(_fmt_e12(z.imag))
                    # Write 4 values per line (Fortran cform140 is 16 values,
                    # but standard EMTF output uses 4 per line)
                    for vi in range(0, len(tf_flat), 4):
                        chunk = "".join(tf_flat[vi:vi + 4])
                        fh.write(" " + chunk + "\n")

                # S matrix: 1 header line, then 1 row per k (lower triangle)
                fh.write(" Inverse Coherent Signal Power Matrix\n")
                for k in range(2):
                    s_flat: list[str] = []
                    for l in range(k + 1):
                        s = self.sig_s[k, l, ib]
                        s_flat.append(_fmt_e12(s.real))
                        s_flat.append(_fmt_e12(s.imag))
                    for vi in range(0, len(s_flat), 4):
                        chunk = "".join(s_flat[vi:vi + 4])
                        fh.write(" " + chunk + "\n")

                # E matrix: 1 header line, then 1 row per k (lower triangle)
                fh.write(" Residual Covariance\n")
                for k in range(nche):
                    e_flat: list[str] = []
                    for l in range(k + 1):
                        e = self.sig_e[k, l, ib]
                        e_flat.append(_fmt_e12(e.real))
                        e_flat.append(_fmt_e12(e.imag))
                    for vi in range(0, len(e_flat), 4):
                        chunk = "".join(e_flat[vi:vi + 4])
                        fh.write(" " + chunk + "\n")

                fh.write("\n")

    def to_edi(self, path: str | Path, *, mode: str = "ss", remote_station_name: str = "") -> None:
        """Export to SEG EDI format (standard MT data interchange).

        Produces an EDI file compatible with WingLink, MTpy, and other
        MT processing software.  See the ``emtf.io.edi`` module for the
        full implementation.
        """
        from emtf.io.edi import write_edi
        write_edi(self, path, mode=mode, remote_station_name=remote_station_name)


# ── convenience wrappers ─────────────────────────────────────────────────


def read_z_file(path: str | Path) -> ZFileData:
    """Read an EMTF Z-file.  Convenience wrapper."""
    return ZFileData.read(path)


def write_z_file(path: str | Path, data: ZFileData) -> None:
    """Write an EMTF Z-file.  Convenience wrapper."""
    data.write(path)


# ── Fortran E12.4 format helpers ────────────────────────────────────────


def _fmt_e12(value: float) -> str:
    """Format a float in Fortran E12.4 style (12-char signed exponential)."""
    if value == 0.0:
        return "  0.0000E+00"
    if np.isnan(value):
        return " -9.9900E+02"
    s = f"{value: 11.4E}"
    if len(s) > 12:
        # Compress exponent to 2 digits
        s = s.replace("E+0", "E+").replace("E-0", "E-")
    return s[:12].rjust(12)


def _parse_fortran_e12_line(line: str, expected: int | None = None) -> list[float]:
    """Parse space-delimited Fortran E12.4 values from one line."""
    values: list[float] = []
    for token in line.split():
        try:
            v = float(token.replace("D", "E").replace("d", "e"))
            if v == _RNAN:
                v = float("nan")
            values.append(v)
        except ValueError:
            continue
    if expected is not None:
        while len(values) < expected:
            values.append(float("nan"))
        values = values[:expected]
    return values


# Legacy alias
ZFile = ZFileData
