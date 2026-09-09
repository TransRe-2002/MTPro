"""Read / write EMTF time-series formats.

Two formats are supported:

*   **EMTF Standard Binary** — block-interleaved int32.  Used by ``clean``,
    ``rfasc``, and ``dnff`` (without ``-a``).  No scaling loss; float data
    is rounded to nearest 32-bit integer before writing.

*   **EMTF ASCII (``.asc``)** — space-separated decimal integers in rows
    of *nch* columns, with a three-line time header and a two-line footer.
    Read by ``dnff -a``.  Requires integer scaling factors (see below).
    This is the format produced by the MATLAB ``TS.writeAscTS`` method.

The System Parameter (``.sp``) file provides per-channel calibration
metadata and is always required alongside an ASCII data file.
"""

from __future__ import annotations

import struct
from pathlib import Path
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

NBLK = 256  # standard block size from datsz.inc
_MISVAL = 44444444  # sentinel for missing data in ASC format


# ============================================================================
#  EMTF Standard Binary
# ============================================================================


def read_binary(
    path: str | Path,
    n_channels: int,
    endian: str = "big",
) -> dict:
    """Read an EMTF standard binary time-series file.

    Returns
    -------
    dict
        ``station_name`` (str), ``n_channels`` (int), ``block_size`` (int),
        ``n_blocks`` (int), ``channels`` (list of ndarray, float64).
    """
    path = Path(path)
    with open(path, "rb") as fh:
        header_raw = fh.read(80)
        station_name = header_raw.decode("ascii", errors="replace").strip()

        fmt = f">{NBLK * n_channels}i"
        block_bytes = 4 * NBLK * n_channels

        buffers: list[list[float]] = [[] for _ in range(n_channels)]
        n_blocks = 0
        while True:
            data = fh.read(block_bytes)
            if len(data) < block_bytes:
                break
            samples = struct.unpack(fmt, data)
            for ich in range(n_channels):
                buffers[ich].extend(samples[ich::n_channels])
            n_blocks += 1

    return {
        "station_name": station_name,
        "n_channels": n_channels,
        "block_size": NBLK,
        "n_blocks": n_blocks,
        "channels": [np.array(buf, dtype=np.float64) for buf in buffers],
    }


def write_binary(
    path: str | Path,
    channels: list[NDArray[np.float64]],
    station_name: str = "STATION",
    endian: str = "big",
    block_size: int = NBLK,
) -> None:
    """Write EMTF standard binary file readable by ``dnff`` (without ``-a``)."""
    nch = len(channels)
    nsamp = len(channels[0])
    for ch in channels[1:]:
        if len(ch) != nsamp:
            raise ValueError(f"Channel length mismatch: {nsamp} vs {len(ch)}")

    path = Path(path)
    fmt = ">" if endian == "big" else "<"

    with open(path, "wb") as fh:
        fh.write(station_name[:80].ljust(80).encode("ascii", errors="replace"))
        pos = 0
        while pos < nsamp:
            n = min(block_size, nsamp - pos)
            for i in range(n):
                for ich in range(nch):
                    val = int(np.round(float(channels[ich][pos + i])))
                    fh.write(struct.pack(f"{fmt}i", val))
            if n < block_size:
                pad = block_size - n
                for _ in range(pad):
                    for _ in range(nch):
                        fh.write(struct.pack(f"{fmt}i", 0))
            pos += n


# ============================================================================
#  EMTF ASCII (.asc) — port of TS.writeAscTS
# ============================================================================


def write_asc(
    path: str | Path,
    channels: list[NDArray[np.float64]],
    dt: float,
    *,
    scales: Sequence[float] | None = None,
    start_time: Sequence[int] | None = None,
    zero_time: Sequence[int] | None = None,
) -> None:
    """Write EMTF ASCII (``.asc``) file — port of ``TS.writeAscTS``.

    ASCII format is the standard input to ``dnff -a``.  Each channel is
    multiplied by *scales*, rounded to the nearest integer, and written
    as a 9-digit decimal field.  NaN values are replaced by the
    sentinel 44444444.

    Parameters
    ----------
    path:
        Output ``.asc`` file path.
    channels:
        One float64 array per channel, same length.  Units should be
        physical (nT for H, mV/km for E).
    dt:
        Sampling interval in seconds.
    scales:
        Per-channel integer scaling factors.  Default: 100 for H,
        1000 for E (these record ~0.01 nT / 0.001 mV/km precision).
        Must have one entry per channel.
    start_time:
        ``[year, month, day, hour, minute, second]`` of the first sample.
        Default: ``[2020, 1, 1, 0, 0, 0]``.
    zero_time:
        Reference zero time for EMTF sample numbering, same format as
        *start_time*.  Defaults to *start_time*.
    """
    nch = len(channels)
    nsamp = len(channels[0])
    for ch in channels[1:]:
        if len(ch) != nsamp:
            raise ValueError(f"Channel length mismatch: {nsamp} vs {len(ch)}")

    if scales is None:
        scales = [100.0] * nch
    if len(scales) != nch:
        raise ValueError(f"scales length {len(scales)} != nch {nch}")

    if start_time is None:
        start_time = (2020, 1, 1, 0, 0, 0)
    if zero_time is None:
        zero_time = start_time

    # Prepare data: x = round(scale * data), NaN → _MISVAL
    data_int = np.full((nch, nsamp), _MISVAL, dtype=np.int64)
    for ich in range(nch):
        arr = np.asarray(channels[ich], dtype=np.float64)
        finite = np.isfinite(arr)
        data_int[ich, finite] = np.round(arr[finite] * scales[ich]).astype(np.int64)

    with open(path, "w") as fh:
        # Header: dt, startTime, zeroTime
        fh.write(f"{dt}\n")
        fh.write(" ".join(str(v) for v in start_time) + "\n")
        fh.write(" ".join(str(v) for v in zero_time) + "\n")

        # Data: one line per sample, 9-digit columns
        for i in range(nsamp):
            row = " ".join(f"{data_int[ich, i]:9d}" for ich in range(nch)) + "\n"
            fh.write(row)

        # Footer: misVals row, scales row
        misvals = " ".join(f"{_MISVAL:9d}" for _ in range(nch)) + "\n"
        fh.write(misvals)
        scale_row = " ".join(f"{int(s):9d}" for s in scales) + "\n"
        fh.write(scale_row)


# ============================================================================
#  System Parameter (.sp) — port of TS.defaultSP
# ============================================================================


def write_sp(
    path: str | Path,
    channel_ids: list[str],
    dt: float,
    *,
    station_name: str = "STATION",
    latitude: float = 0.0,
    longitude: float = 0.0,
    elevation: float = 0.0,
    declination: float = 0.0,
    scales: Sequence[float] | None = None,
    line_lengths_km: Sequence[float] | None = None,
) -> None:
    """Write EMTF System Parameter (``.sp``) file — port of ``TS.defaultSP``.

    The SP file tells ``dnff`` how many channels are present, their
    orientations, and how raw integer counts convert to physical units.

    Parameters
    ----------
    path:
        Output ``.sp`` file path.
    channel_ids:
        Channel identifiers, e.g. ``["Hx", "Hy", "Hz", "Ex", "Ey"]``.
        Leading character ``H`` → magnetic; ``E`` → electric.
    dt:
        Sampling interval in seconds.
    station_name:
        Station name written in the header.
    latitude, longitude:
        Station coordinates (decimal degrees).
    elevation:
        Station elevation in metres.
    declination:
        Magnetic declination in degrees.
    scales:
        Per-channel integer scaling factors, same as for :func:`write_asc`.
        The SP file stores ``count_conversion = 1 / scales[ich]``.
        Default: 100 for H, 1000 for E.
    line_lengths_km:
        Per-channel dipole line length in km (for E channels only).
        Must have one entry per channel; ignored for H channels.
    """
    nch = len(channel_ids)

    if scales is None:
        scales = []
        for cid in channel_ids:
            if cid.upper().startswith("H"):
                scales.append(100.0)
            elif cid.upper().startswith("E"):
                scales.append(1000.0)
            else:
                scales.append(1.0)
    if len(scales) != nch:
        raise ValueError(f"scales length {len(scales)} != nch {nch}")

    if line_lengths_km is None:
        line_lengths_km = [1.0] * nch
    if len(line_lengths_km) != nch:
        raise ValueError(f"line_lengths_km length {len(line_lengths_km)} != nch {nch}")

    with open(path, "w") as fh:
        fh.write(f"{station_name}\n")
        fh.write(f"{latitude:10.4f} {longitude:10.4f} {elevation:7.0f}\n")
        fh.write(f"{declination:6.2f}\n")
        fh.write(f"{nch}\n")
        fh.write(f"{dt:10.4f}\n")
        fh.write("0 0\n")  # clock offset & linear drift (legacy)

        for ich in range(nch):
            cid = channel_ids[ich]
            fh.write(f"{cid}\n")

            if cid.upper().startswith("H"):
                az = 0.0 if cid.upper() == "HX" else (90.0 if cid.upper() == "HY" else 0.0)
                fh.write(f"{az:8.2f} {0.0:8.2f}\n")
                fh.write(f"{1.0 / scales[ich]:g}  0  conversion nT/count, # of filters\n")

            elif cid.upper().startswith("E"):
                az = 0.0 if "X" in cid.upper() else (90.0 if "Y" in cid.upper() else 0.0)
                ll = line_lengths_km[ich]
                fh.write(f"{ll:8.2f} {az:8.2f} {0.0:8.2f} {1.0}\n")
                fh.write(f"{1.0 / scales[ich]:g}  0  conversion mV/count, # of filters\n")

            else:
                fh.write(f"{0.0:8.2f} {0.0:8.2f}\n")
                fh.write(f"{1.0 / scales[ich]:g}  0\n")
