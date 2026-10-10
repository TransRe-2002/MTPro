"""Low-level FC file header parser — implementation detail.

Ported from ``EMTF/matlab/FC/fcOpen.m``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from emtf.io.fc import FCOutput


def _read_fc_open(
    path: str | Path,
    lpack: bool = True,
    endian: str = "little",
) -> FCOutput:
    path = Path(path)
    header_records = 20

    with open(path, "rb") as fh:
        raw = fh.read(512)
        line = raw.decode("ascii", errors="replace")

    # Parse the ASCII header using regex to extract numeric fields robustly
    # (independent of exact Fortran fixed-format column positions).
    nch = _extract_int(line, r'nch\s+(\d+)')
    nd  = _extract_int(line, r'nd\s+(\d+)')
    nf  = _extract_int(line, r'nfmx\s+(\d+)')

    # decimation factors: after "nwin" — 2×nd integer fields
    nwin_match = re.search(r'nwin\s+(.+)', line)
    nwin_tail = nwin_match.group(1) if nwin_match else ""
    nwin_nums = _parse_floats(nwin_tail)
    nwin_ints = [int(x) for x in nwin_nums if x == int(x)]
    decs = np.zeros((2, nd), dtype=np.int32)
    idx = 0
    for j in range(nd):
        for k in range(2):
            if idx < len(nwin_ints):
                decs[k, j] = nwin_ints[idx]
            idx += 1

    # decimation rates: after "dr" — nd float fields
    dr_match = re.search(r'dr\s+(.+)', line)
    dr_tail = dr_match.group(1) if dr_match else ""
    dr_nums = _parse_floats(dr_tail)
    drs = np.zeros(nd, dtype=np.float64)
    for j in range(min(nd, len(dr_nums))):
        drs[j] = dr_nums[j]

    # channels: after "ch,orien" — nch×(chid(6chars)+orientation(2×7chars))
    ch_match = re.search(r'ch,orien(.+)', line)
    ch_tail = ch_match.group(1) if ch_match else ""
    # ch_tail: "HxTS1    0.0    0.00HyTS1   90.0    0.00..."
    # Each channel: 6 char ID + 14 char orientation = 20 chars total
    chid: list[str] = []
    orient = np.zeros((2, nch), dtype=np.float64)
    pos = 0
    for ich in range(nch):
        if pos + 20 > len(ch_tail):
            break
        block = ch_tail[pos:pos + 20]
        chid.append(block[:6].strip())
        ori_nums = _parse_floats(block[6:])
        if len(ori_nums) >= 2:
            orient[0, ich] = ori_nums[0]
            orient[1, ich] = ori_nums[1]
        pos += 20

    # station coordinates + declination: after "scor"
    scor_match = re.search(r'scor\s+([\d.]+)\s+([\d.]+)', line)
    if scor_match:
        stcor = (float(scor_match.group(1)), float(scor_match.group(2)))
    else:
        stcor = (0.0, 0.0)

    decl_match = re.search(r'decl\s+([\d.]+)', line)
    decl = float(decl_match.group(1)) if decl_match else 0.0

    # Determine record length
    if lpack:
        irecl = nch + 1
    else:
        irecl = 2 * nch + 1

    # Build a header record dtype
    dt_rec = np.dtype(">i4") if endian == "big" else np.dtype("<i4")

    start_dec: list[int] = [-1] * nd
    start_freqs = np.full((nd, nf), -1, dtype=np.int64)

    with open(path, "rb") as fh:
        fh.seek(header_records * 4 * irecl)

        file_size = path.stat().st_size
        while fh.tell() < file_size:
            pos_before = fh.tell()
            head = np.frombuffer(fh.read(4 * irecl), dtype=dt_rec)
            if len(head) != irecl:
                raise ValueError("Truncated FC record header")
            idec, ifreq, nsegs = map(int, head[:3])
            if not (1 <= idec <= nd and 1 <= ifreq <= nf and nsegs >= 0):
                raise ValueError(f"Invalid FC record at byte {pos_before}")
            if start_dec[idec - 1] < 0:
                start_dec[idec - 1] = pos_before
            start_freqs[idec - 1, ifreq - 1] = pos_before
            # pack.f writes a whole scale record before the nsegs data records.
            next_record = fh.tell() + 4 * irecl * (nsegs + 1)
            if next_record > file_size:
                raise ValueError("Truncated FC scale/data records")
            fh.seek(next_record)

        # Read set numbers per decimation level
        i_sets: list[NDArray[np.int32]] = []
        for id_ in range(nd):
            if start_dec[id_] < 0:
                i_sets.append(np.zeros(0, dtype=np.int32))
                continue
            fh.seek(start_dec[id_])
            head = np.frombuffer(fh.read(12), dtype=dt_rec)
            nsets = int(head[2])
            if lpack:
                fh.seek(4 * (2 * irecl - 3), 1)
                iskip = 4 * (irecl - 1)
            else:
                fh.seek(4 * (2 * irecl - 3), 1)
                iskip = 4 * (irecl - 1)
            sets_arr = np.zeros(nsets, dtype=np.int32)
            for l in range(nsets):
                sets_arr[l] = np.frombuffer(fh.read(4), dtype=dt_rec)[0]
                fh.seek(iskip, 1)
            i_sets.append(sets_arr)

    return FCOutput(
        file=str(path),
        endian=endian,
        lpack=lpack,
        nch=nch, nd=nd, nf=nf,
        decs=decs, drs=drs,
        chid=chid, orient=orient,
        stcor=stcor, decl=decl,
        start_dec=start_dec,
        start_freqs=start_freqs,
        i_sets=i_sets,
    )


def _extract_int(line: str, pattern: str) -> int:
    m = re.search(pattern, line)
    return int(m.group(1)) if m else 0


def _parse_floats(text: str) -> list[float]:
    """Parse space-separated floats from text, handling Fortran D exponent."""
    values: list[float] = []
    for token in text.split():
        try:
            values.append(float(token.replace("D", "E").replace("d", "e")))
        except ValueError:
            continue
    return values
