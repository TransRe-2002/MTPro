"""IO: System parameter (``.sp``) file reader.

Ported from ``EMTF/matlab/UTIL/readSp.m``.
SP files contain per-station calibration metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SystemParams:
    """Calibration metadata for one magnetotelluric station."""

    run_id: str = ""
    lat: float = 0.0
    lon: float = 0.0
    elev: float = 0.0
    decl: float = 0.0
    nch: int = 0
    dt: float = 0.0  # sampling interval (seconds)

    @dataclass
    class Channel:
        channel_id: str = ""
        azimuth: float = 0.0
        tilt: float = 0.0
        line_length_km: float = 0.0
        gain: float = 1.0
        count_conversion: float = 0.0  # nT/count or mV/km/count
        filters: list["SystemParams.Filter"] = field(default_factory=list)

    @dataclass
    class Filter:
        filter_type: str = ""
        name: str = ""
        parameters: str = ""

    channels: list[Channel] = field(default_factory=list)

    @property
    def hx(self) -> Channel | None:
        for ch in self.channels:
            if ch.channel_id.upper().startswith("H") and "X" in ch.channel_id.upper():
                return ch
        return None

    @property
    def hy(self) -> Channel | None:
        for ch in self.channels:
            if ch.channel_id.upper().startswith("H") and "Y" in ch.channel_id.upper():
                return ch
        return None

    @property
    def hz(self) -> Channel | None:
        for ch in self.channels:
            if ch.channel_id.upper().startswith("H") and "Z" in ch.channel_id.upper():
                return ch
        return None

    @property
    def ex(self) -> Channel | None:
        for ch in self.channels:
            if ch.channel_id.upper().startswith("E") and "X" in ch.channel_id.upper():
                return ch
        return None

    @property
    def ey(self) -> Channel | None:
        for ch in self.channels:
            if ch.channel_id.upper().startswith("E") and "Y" in ch.channel_id.upper():
                return ch
        return None


def read_sp(path: str | Path) -> SystemParams:
    """Read an EMTF system parameter file.

    SP file format (from readSp.m)::

        <run_id>
        <lat> <lon> <elev?>        lat./long.in deg
        <declination>
        <nch>                       # of channels
        <dt>                        sampling rate (seconds)
        <clock_offset> <drift>      clock offset & linear drift
        (per channel: 2-3 lines)
          <chid>                    e.g. HxTS1
          <azimuth> <tilt>          sensor orientation & vertical tilt (H)
          <line_len> <azimuth> <tilt> <gain>   (E only)
          <count_conversion> <n_filters>   count conversion, # of filter
          (<filter_type_and_name>)*n_filters
          (<filter_params>)*n_filters
    """
    path = Path(path)
    with open(path, "r") as fh:
        lines = [l.strip() for l in fh if l.strip()]

    sp = SystemParams()
    sp.run_id = lines[0]

    # Coordinate line (may have trailing commentary like "lat./long.in deg")
    parts = lines[1].split()
    sp.lat = float(parts[0])
    sp.lon = float(parts[1])
    # Third token may be elevation or commentary; try float, fall back to 0
    if len(parts) >= 3:
        try:
            sp.elev = float(parts[2])
        except ValueError:
            sp.elev = 0.0

    # Declination
    sp.decl = float(lines[2].split()[0])

    # Number of channels
    sp.nch = int(lines[3].split()[0])

    # Sampling rate
    sp.dt = float(lines[4].split()[0])

    # Clock offset + drift (skip)
    line_idx = 6  # skip line 5 (clock offset)

    for _ in range(sp.nch):
        if line_idx >= len(lines):
            break
        chid = lines[line_idx].split()[0]
        line_idx += 1

        if line_idx >= len(lines):
            break
        parts = lines[line_idx].split()
        line_idx += 1

        if chid.upper().startswith("H"):
            azimuth = float(parts[0])
            tilt = float(parts[1]) if len(parts) >= 2 else 0.0
            line_len = 0.0
            gain = 1.0
        elif chid.upper().startswith("E"):
            line_len = float(parts[0])
            azimuth = float(parts[1]) if len(parts) >= 2 else 0.0
            tilt = float(parts[2]) if len(parts) >= 3 else 0.0
            gain = float(parts[3]) if len(parts) >= 4 else 1.0
        else:
            azimuth, tilt, line_len, gain = 0.0, 0.0, 0.0, 1.0

        # Count conversion + number of filters
        if line_idx >= len(lines):
            break
        parts = lines[line_idx].split()
        count_conversion = float(parts[0])
        n_filters = int(float(parts[1]))
        line_idx += 1

        filters: list[SystemParams.Filter] = []
        for _ in range(n_filters):
            if line_idx >= len(lines):
                break
            ftype = lines[line_idx]
            line_idx += 1
            if line_idx >= len(lines):
                break
            fparams = lines[line_idx]
            line_idx += 1
            ft = ftype[:2] if len(ftype) >= 2 else ftype
            fn = ftype[2:].strip() if len(ftype) > 2 else ""
            filters.append(SystemParams.Filter(filter_type=ft, name=fn, parameters=fparams))

        sp.channels.append(SystemParams.Channel(
            channel_id=chid,
            azimuth=azimuth, tilt=tilt,
            line_length_km=line_len, gain=gain,
            count_conversion=count_conversion,
            filters=filters,
        ))

    return sp
