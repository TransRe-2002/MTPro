"""EDI export helpers.

The writer emits the common SEG EDI array-section layout used by MT tools:
one ``>FREQ`` block followed by real/imaginary impedance, variance, and
optional tipper arrays.  It also provides a small banded-result adapter for
GUI callers that already hold impedance arrays instead of an EMTF Z-file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from emtf.io.zfile import ZFileData, emtf_phase_deg

EMPTY = 1.0e32


@dataclass
class EdiMetadata:
    """Station/export metadata used by EDI and Z-file formatting."""

    station_name: str = "unknown"
    mode: str = "ss"
    lat: float = 0.0
    lon: float = 0.0
    elev: float = 0.0
    declination: float = 0.0
    acquired_by: str = "EMTF"
    file_by: str = "atm_rpc-emtf"
    remote_station_name: str = ""

    @property
    def z_suffix(self) -> str:
        return "zrr" if self.mode.lower().startswith("rr") else "zss"


@dataclass
class BandedResult:
    """Array-backed impedance result with ``atm_rpc``-friendly export methods."""

    periods: NDArray[np.float64]
    zxx: NDArray[np.complex128]
    zxy: NDArray[np.complex128]
    zyx: NDArray[np.complex128]
    zyy: NDArray[np.complex128]
    zxx_se: NDArray[np.float64]
    zxy_se: NDArray[np.float64]
    zyx_se: NDArray[np.float64]
    zyy_se: NDArray[np.float64]
    n_observations: NDArray[np.int32] | None = None
    tzx: NDArray[np.complex128] | None = None
    tzy: NDArray[np.complex128] | None = None
    tzx_se: NDArray[np.float64] | None = None
    tzy_se: NDArray[np.float64] | None = None
    station_name: str = "unknown"
    mode: str = "ss"
    lat: float = 0.0
    lon: float = 0.0
    elev: float = 0.0
    declination: float = 0.0

    @property
    def center_frequencies_hz(self) -> NDArray[np.float64]:
        out = np.full(len(self.periods), np.nan, dtype=np.float64)
        valid = self.periods > 0
        out[valid] = 1.0 / self.periods[valid]
        return out

    @property
    def rho_xy(self) -> NDArray[np.float64]:
        return _rho(self.periods, self.zxy)

    @property
    def rho_yx(self) -> NDArray[np.float64]:
        return _rho(self.periods, self.zyx)

    @property
    def phi_xy(self) -> NDArray[np.float64]:
        return emtf_phase_deg(self.zxy)

    @property
    def phi_yx(self) -> NDArray[np.float64]:
        return emtf_phase_deg(self.zyx)

    @property
    def rho_xy_se(self) -> NDArray[np.float64]:
        return _rho_se(self.periods, self.zxy, self.zxy_se)

    @property
    def rho_yx_se(self) -> NDArray[np.float64]:
        return _rho_se(self.periods, self.zyx, self.zyx_se)

    @property
    def phi_xy_se(self) -> NDArray[np.float64]:
        return _phase_se(self.zxy, self.zxy_se)

    @property
    def phi_yx_se(self) -> NDArray[np.float64]:
        return _phase_se(self.zyx, self.zyx_se)

    def format_edi_str(self, meta: Any = None, **kwargs: Any) -> str:
        """Return EDI text. Accepts ``meta`` or ``station_name=...`` kwargs."""
        metadata = _coerce_metadata(
            meta,
            default_station=self.station_name,
            default_mode=self.mode,
            default_lat=self.lat,
            default_lon=self.lon,
            default_elev=self.elev,
            default_declination=self.declination,
            **kwargs,
        )
        return format_edi(self, metadata)

    def format_z_str(self, meta: Any = None, **kwargs: Any) -> str:
        """Return a minimal EMTF-style Z-file string."""
        metadata = _coerce_metadata(
            meta,
            default_station=self.station_name,
            default_mode=self.mode,
            default_lat=self.lat,
            default_lon=self.lon,
            default_elev=self.elev,
            default_declination=self.declination,
            **kwargs,
        )
        return format_z(self, metadata)

    def export_to_folder(self, folder: str | Path, meta: Any = None, **kwargs: Any) -> tuple[Path, Path]:
        """Write ``<station>.edi`` and ``Z.<station>.<zss|zrr>`` to *folder*.

        This intentionally supports the call shape used by ``atm_rpc``:
        ``export_to_folder(folder, station_name="STA", mode="ss")``.
        """
        metadata = _coerce_metadata(
            meta,
            default_station=self.station_name,
            default_mode=self.mode,
            default_lat=self.lat,
            default_lon=self.lon,
            default_elev=self.elev,
            default_declination=self.declination,
            **kwargs,
        )
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        edi_path = folder / f"{metadata.station_name}.edi"
        z_path = folder / f"Z.{metadata.station_name}.{metadata.z_suffix}"
        edi_path.write_text(format_edi(self, metadata), encoding="ascii")
        z_path.write_text(format_z(self, metadata), encoding="ascii")
        return edi_path, z_path


def write_edi(
    data: ZFileData,
    path: str | Path,
    *,
    mode: str = "ss",
    remote_station_name: str = "",
    station_name: str | None = None,
    elevation: float = 0.0,
) -> None:
    """Export :class:`ZFileData` to EDI."""
    metadata = EdiMetadata(
        station_name=(station_name or data.stname or "unknown").strip(),
        mode=mode,
        lat=float(data.stcor[0]),
        lon=float(data.stcor[1]),
        elev=elevation,
        declination=float(data.declination),
        remote_station_name=remote_station_name,
    )
    Path(path).write_text(format_edi(data, metadata), encoding="ascii")


def to_banded_result(data: ZFileData, *, station_name: str | None = None, mode: str = "ss") -> BandedResult:
    """Convert a parsed EMTF Z-file to the array-backed export adapter."""
    nbt = data.nbt
    s_diag = np.real(np.diagonal(data.sig_s, axis1=0, axis2=1)) if data.sig_s.size else np.zeros((nbt, 2))
    e_diag = np.real(np.diagonal(data.sig_e, axis1=0, axis2=1)) if data.sig_e.size else np.zeros((nbt, data.nche))

    hz_i = data._predicted_index("HZ", default=-1)
    ex_i = data._predicted_index("EX", default=0)
    ey_i = data._predicted_index("EY", default=1)

    def se(input_i: int, output_i: int) -> NDArray[np.float64]:
        out = np.full(nbt, np.nan, dtype=np.float64)
        if output_i < 0 or output_i >= data.nche or input_i >= 2:
            return out
        valid = (s_diag[:, input_i] >= 0) & (e_diag[:, output_i] >= 0)
        out[valid] = np.sqrt(s_diag[valid, input_i] * e_diag[valid, output_i] / 2.0)
        return out

    tzx = tzy = tzx_se = tzy_se = None
    if hz_i >= 0 and hz_i < data.tf.shape[1]:
        tzx = np.asarray(data.tf[0, hz_i, :], dtype=np.complex128)
        tzy = np.asarray(data.tf[1, hz_i, :], dtype=np.complex128)
        tzx_se = se(0, hz_i)
        tzy_se = se(1, hz_i)

    return BandedResult(
        periods=np.asarray(data.periods, dtype=np.float64),
        zxx=np.asarray(data.Zxx, dtype=np.complex128),
        zxy=np.asarray(data.Zxy, dtype=np.complex128),
        zyx=np.asarray(data.Zyx, dtype=np.complex128),
        zyy=np.asarray(data.Zyy, dtype=np.complex128),
        zxx_se=se(0, ex_i),
        zxy_se=se(1, ex_i),
        zyx_se=se(0, ey_i),
        zyy_se=se(1, ey_i),
        n_observations=np.asarray(data.ndf if len(data.ndf) == nbt else np.zeros(nbt), dtype=np.int32),
        tzx=tzx,
        tzy=tzy,
        tzx_se=tzx_se,
        tzy_se=tzy_se,
        station_name=(station_name or data.stname or "unknown").strip(),
        mode=mode,
        lat=float(data.stcor[0]),
        lon=float(data.stcor[1]),
        declination=float(data.declination),
    )


def from_result(result: Any, *, station_name: str = "unknown", mode: str = "ss", **metadata: Any) -> BandedResult:
    """Build :class:`BandedResult` from an ``emtf_rs``/``atm_rpc``-style result object."""
    periods = _array_attr(result, "periods")
    if periods is None:
        freqs = _array_attr(result, "center_frequencies_hz")
        if freqs is None:
            freqs = _array_attr(result, "frequencies_hz")
        if freqs is None:
            raise ValueError("result must provide periods or frequencies_hz")
        periods = np.full(len(freqs), np.nan, dtype=np.float64)
        valid = freqs > 0
        periods[valid] = 1.0 / freqs[valid]

    return BandedResult(
        periods=np.asarray(periods, dtype=np.float64),
        zxx=_complex_attr(result, "zxx"),
        zxy=_complex_attr(result, "zxy"),
        zyx=_complex_attr(result, "zyx"),
        zyy=_complex_attr(result, "zyy"),
        zxx_se=_float_attr(result, "zxx_se", "zxx_err"),
        zxy_se=_float_attr(result, "zxy_se", "zxy_err"),
        zyx_se=_float_attr(result, "zyx_se", "zyx_err"),
        zyy_se=_float_attr(result, "zyy_se", "zyy_err"),
        n_observations=_array_attr(result, "n_observations"),
        tzx=_optional_complex_attr(result, "tzx"),
        tzy=_optional_complex_attr(result, "tzy"),
        tzx_se=_optional_float_attr(result, "tzx_se", "tzx_err"),
        tzy_se=_optional_float_attr(result, "tzy_se", "tzy_err"),
        station_name=station_name,
        mode=mode,
        lat=float(metadata.get("lat", 0.0)),
        lon=float(metadata.get("lon", 0.0)),
        elev=float(metadata.get("elev", 0.0)),
        declination=float(metadata.get("declination", 0.0)),
    )


def export_result_to_folder(result: Any, folder: str | Path, **kwargs: Any) -> tuple[Path, Path]:
    """Convert an array result object and export EDI + Z-file."""
    station_name = str(kwargs.pop("station_name", "unknown"))
    mode = str(kwargs.pop("mode", "ss"))
    banded = from_result(result, station_name=station_name, mode=mode, **kwargs)
    return banded.export_to_folder(folder, station_name=station_name, mode=mode, **kwargs)


def format_edi(data: ZFileData | BandedResult, meta: EdiMetadata | Mapping[str, Any] | None = None) -> str:
    """Return SEG-style EDI text for Z-file or banded array data."""
    metadata = _coerce_metadata(
        meta,
        default_station=_default_station(data),
        default_mode=str(getattr(data, "mode", "ss")),
        default_lat=float(getattr(data, "lat", getattr(data, "stcor", (0.0, 0.0))[0])),
        default_lon=float(getattr(data, "lon", getattr(data, "stcor", (0.0, 0.0))[1])),
        default_elev=float(getattr(data, "elev", 0.0)),
        default_declination=float(getattr(data, "declination", 0.0)),
    )
    banded = to_banded_result(data, station_name=metadata.station_name, mode=metadata.mode) if isinstance(data, ZFileData) else data
    idx = _valid_indices(banded)
    freqs = banded.center_frequencies_hz[idx]
    n = len(idx)
    now = datetime.now(timezone.utc).strftime("%m/%d/%Y")

    lines: list[str] = [
        ">HEAD\n",
        f'    DATAID="{metadata.station_name}"\n',
        f'    ACQBY="{metadata.acquired_by}"\n',
        f'    FILEBY="{metadata.file_by}"\n',
        f'    ACQDATE="{now}"\n',
        f'    FILEDATE="{now}"\n',
        f"    LAT={metadata.lat:.6f}\n",
        f"    LONG={metadata.lon:.6f}\n",
        f"    ELEV={metadata.elev:.2f}\n",
        "    UNITS=M\n",
        '    STDVERS="SEG 1.0"\n',
        f"    EMPTY={EMPTY:.6E}\n",
        "\n>INFO\n",
        f"    Processed by atm_rpc EMTF in {metadata.mode.upper()} mode.\n",
    ]
    if metadata.remote_station_name:
        lines.append(f"    Remote reference station: {metadata.remote_station_name}\n")

    has_tipper = banded.tzx is not None and banded.tzy is not None
    maxchan = 7 if has_tipper else 4
    lines.extend([
        "\n>=DEFINEMEAS\n",
        f"    MAXCHAN={maxchan}\n",
        "    MAXRUN=999\n",
        "    MAXMEAS=9999\n",
        "    UNITS=M\n",
        "    REFTYPE=CART\n",
        f"    REFLAT={metadata.lat:.6f}\n",
        f"    REFLONG={metadata.lon:.6f}\n",
        f"    REFELEV={metadata.elev:.2f}\n",
        f'    REFLOC="{metadata.station_name}"\n',
        "    >HMEAS ID=101.001 CHTYPE=HX X=0.0 Y=0.0 Z=0.0 AZM=0.0\n",
        "    >HMEAS ID=102.001 CHTYPE=HY X=0.0 Y=0.0 Z=0.0 AZM=90.0\n",
    ])
    if has_tipper:
        lines.append("    >HMEAS ID=105.001 CHTYPE=HZ X=0.0 Y=0.0 Z=0.0 AZM=0.0\n")
        lines.append("    >EMEAS ID=106.001 CHTYPE=EX X=0.0 Y=0.0 Z=0.0 X2=0.0 Y2=0.0 Z2=0.0\n")
        lines.append("    >EMEAS ID=107.001 CHTYPE=EY X=0.0 Y=0.0 Z=0.0 X2=0.0 Y2=0.0 Z2=0.0\n")
    else:
        lines.append("    >EMEAS ID=103.001 CHTYPE=EX X=0.0 Y=0.0 Z=0.0 X2=0.0 Y2=0.0 Z2=0.0\n")
        lines.append("    >EMEAS ID=104.001 CHTYPE=EY X=0.0 Y=0.0 Z=0.0 X2=0.0 Y2=0.0 Z2=0.0\n")

    lines.extend([
        "\n>=MTSECT\n",
        f'    SECTID="{metadata.station_name}"\n',
        "    NFREQ=%d\n" % n,
        "    HX=101.001\n",
        "    HY=102.001\n",
    ])
    if has_tipper:
        lines.append("    HZ=105.001\n")
        lines.append("    EX=106.001\n")
        lines.append("    EY=107.001\n")
    else:
        lines.append("    EX=103.001\n")
        lines.append("    EY=104.001\n")

    lines.append(_array_block("FREQ", freqs, order_dec=True))
    lines.append(_array_block("ZROT", np.zeros(n, dtype=np.float64), decimals=4))
    lines.append(_array_block("ZXXR", banded.zxx.real[idx]))
    lines.append(_array_block("ZXXI", banded.zxx.imag[idx]))
    lines.append(_array_block("ZXX.VAR", _square(banded.zxx_se[idx])))
    lines.append(_array_block("ZXYR", banded.zxy.real[idx]))
    lines.append(_array_block("ZXYI", banded.zxy.imag[idx]))
    lines.append(_array_block("ZXY.VAR", _square(banded.zxy_se[idx])))
    lines.append(_array_block("ZYXR", banded.zyx.real[idx]))
    lines.append(_array_block("ZYXI", banded.zyx.imag[idx]))
    lines.append(_array_block("ZYX.VAR", _square(banded.zyx_se[idx])))
    lines.append(_array_block("ZYYR", banded.zyy.real[idx]))
    lines.append(_array_block("ZYYI", banded.zyy.imag[idx]))
    lines.append(_array_block("ZYY.VAR", _square(banded.zyy_se[idx])))

    if banded.tzx is not None and banded.tzy is not None:
        tx_se = banded.tzx_se if banded.tzx_se is not None else np.full(len(banded.periods), np.nan)
        ty_se = banded.tzy_se if banded.tzy_se is not None else np.full(len(banded.periods), np.nan)
        lines.append(_array_block("TXR.EXP", banded.tzx.real[idx]))
        lines.append(_array_block("TXI.EXP", banded.tzx.imag[idx]))
        lines.append(_array_block("TXVAR.EXP", _square(tx_se[idx])))
        lines.append(_array_block("TYR.EXP", banded.tzy.real[idx]))
        lines.append(_array_block("TYI.EXP", banded.tzy.imag[idx]))
        lines.append(_array_block("TYVAR.EXP", _square(ty_se[idx])))

    lines.append("\n>END\n")
    return "".join(lines)


def format_z(data: BandedResult, meta: EdiMetadata | Mapping[str, Any] | None = None) -> str:
    """Return a compact EMTF-style Z-file string from array data."""
    metadata = _coerce_metadata(meta, default_station=data.station_name, default_mode=data.mode)
    idx = _valid_indices(data)
    has_tipper = data.tzx is not None and data.tzy is not None
    nch = 5 if has_tipper else 4
    nche = nch - 2
    lines = [
        " **** IMPEDANCE IN MEASUREMENT COORDINATES ****\n",
        " ********** WITH FULL ERROR COVARINCE**********\n",
        f"Robust {'Remote Reference' if metadata.mode.lower().startswith('rr') else 'Single station'}\n",
        f"station    :{metadata.station_name:>20}\n",
        f"coordinate {metadata.lat:9.3f} {metadata.lon:9.3f}  declination {metadata.declination:8.2f}\n",
        f"number of channels {nch:3d}   number of frequencies {len(idx):4d}\n",
        " orientations and tilts of each channel \n",
        f"{1:5d} {0.0:8.2f} {0.0:8.2f} {metadata.station_name[:3]:3s} {'Hx':6s}\n",
        f"{2:5d} {90.0:8.2f} {0.0:8.2f} {metadata.station_name[:3]:3s} {'Hy':6s}\n",
    ]
    if has_tipper:
        lines.append(f"{3:5d} {0.0:8.2f} {0.0:8.2f} {metadata.station_name[:3]:3s} {'Hz':6s}\n")
        lines.append(f"{4:5d} {0.0:8.2f} {0.0:8.2f} {metadata.station_name[:3]:3s} {'Ex':6s}\n")
        lines.append(f"{5:5d} {90.0:8.2f} {0.0:8.2f} {metadata.station_name[:3]:3s} {'Ey':6s}\n")
    else:
        lines.append(f"{3:5d} {0.0:8.2f} {0.0:8.2f} {metadata.station_name[:3]:3s} {'Ex':6s}\n")
        lines.append(f"{4:5d} {90.0:8.2f} {0.0:8.2f} {metadata.station_name[:3]:3s} {'Ey':6s}\n")
    lines.append("\n")

    nobs = data.n_observations if data.n_observations is not None else np.zeros(len(data.periods), dtype=np.int32)
    for out_i, ib in enumerate(idx, start=1):
        lines.append(
            f"period : {data.periods[ib]:12.5f}    decimation level {1:3d}    "
            f"freq. band from {out_i:4d} to {out_i:4d}\n"
        )
        lines.append(f"number of data point {int(nobs[ib]) if ib < len(nobs) else 0:6d} sampling freq. {1.0:7.3f} Hz\n")
        lines.append(" Transfer Functions\n")
        if has_tipper:
            tfs = [
                [data.tzx[ib], data.tzy[ib]],  # type: ignore[index]
                [data.zxx[ib], data.zxy[ib]],
                [data.zyx[ib], data.zyy[ib]],
            ]
        else:
            tfs = [
                [data.zxx[ib], data.zxy[ib]],
                [data.zyx[ib], data.zyy[ib]],
            ]
        for row in tfs:
            vals: list[float] = []
            for val in row:
                vals.extend([float(np.real(val)), float(np.imag(val))])
            lines.extend(_z_number_lines(vals))
        lines.append(" Inverse Coherent Signal Power Matrix\n")
        lines.extend(_z_number_lines([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        lines.append(" Residual Covariance\n")
        se = [data.zxx_se[ib], data.zyx_se[ib]]
        if has_tipper and data.tzx_se is not None:
            se = [data.tzx_se[ib], data.zxx_se[ib], data.zyx_se[ib]]
        e_vals: list[float] = []
        for k in range(nche):
            for l in range(k + 1):
                e_vals.extend([float(se[k] ** 2) if k == l else 0.0, 0.0])
        lines.extend(_z_number_lines(e_vals))
        lines.append("\n")
    return "".join(lines)


def _array_block(
    name: str,
    values: Sequence[float],
    *,
    order_dec: bool = False,
    decimals: int | None = None,
) -> str:
    arr = np.asarray(values, dtype=np.float64)
    if order_dec:
        out = [f"\n>{name} ORDER=DEC  //          {len(arr)}\n"]
    else:
        out = [f"\n>{name}  //          {len(arr)}\n"]
    for start in range(0, len(arr), 6):
        out.append(" ".join(_fmt_edi(v, decimals=decimals) for v in arr[start:start + 6]) + "\n")
    return "".join(out)


def _fmt_edi(value: float, *, decimals: int | None = None) -> str:
    if not np.isfinite(value):
        value = EMPTY
    if decimals is not None:
        return f"{value:.{decimals}f}"
    return f"{value: .6E}"


def _z_number_lines(values: Sequence[float]) -> list[str]:
    out = []
    formatted = [_fmt_z(v) for v in values]
    for start in range(0, len(formatted), 4):
        out.append(" " + "".join(formatted[start:start + 4]) + "\n")
    return out


def _fmt_z(value: float) -> str:
    if not np.isfinite(value):
        value = -999.0
    return f"{value:12.4E}"


def _valid_indices(data: BandedResult) -> NDArray[np.int64]:
    valid = np.isfinite(data.periods) & (data.periods > 0)
    valid &= np.isfinite(data.zxy.real) & np.isfinite(data.zyx.real)
    return np.flatnonzero(valid)


def _coerce_metadata(
    meta: Any = None,
    *,
    default_station: str = "unknown",
    default_mode: str = "ss",
    default_lat: float = 0.0,
    default_lon: float = 0.0,
    default_elev: float = 0.0,
    default_declination: float = 0.0,
    **kwargs: Any,
) -> EdiMetadata:
    if isinstance(meta, EdiMetadata):
        base = meta
    else:
        values: dict[str, Any] = {}
        if isinstance(meta, Mapping):
            values.update(meta)
        elif meta is not None:
            for attr in (
                "station_name", "name", "mode", "lat", "lon", "elev",
                "elevation", "declination", "remote_station_name",
            ):
                if hasattr(meta, attr):
                    values[attr] = getattr(meta, attr)
        values.update(kwargs)
        station = values.get("station_name", values.get("name", default_station))
        base = EdiMetadata(
            station_name=str(station or default_station),
            mode=str(values.get("mode", default_mode)),
            lat=float(values.get("lat", default_lat)),
            lon=float(values.get("lon", default_lon)),
            elev=float(values.get("elev", values.get("elevation", default_elev))),
            declination=float(values.get("declination", default_declination)),
            remote_station_name=str(values.get("remote_station_name", "")),
        )
    if not base.station_name:
        base.station_name = default_station
    return base


def _default_station(data: ZFileData | BandedResult) -> str:
    return (getattr(data, "stname", "") or getattr(data, "station_name", "") or "unknown").strip()


def _rho(periods: NDArray[np.float64], z: NDArray[np.complex128]) -> NDArray[np.float64]:
    out = np.full(len(periods), np.nan, dtype=np.float64)
    valid = periods > 0
    out[valid] = 0.2 * periods[valid] * np.abs(z[valid]) ** 2
    return out


def _rho_se(periods: NDArray[np.float64], z: NDArray[np.complex128], z_se: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.full(len(periods), np.nan, dtype=np.float64)
    valid = (periods > 0) & (np.abs(z) > 0)
    out[valid] = 2.0 * _rho(periods, z)[valid] * z_se[valid] / np.abs(z[valid])
    return out


def _phase_se(z: NDArray[np.complex128], z_se: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.full(len(z), np.nan, dtype=np.float64)
    valid = np.abs(z) > 0
    out[valid] = np.degrees(z_se[valid] / np.abs(z[valid]))
    return out


def _square(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64) ** 2


def _array_attr(obj: Any, name: str) -> NDArray[np.float64] | None:
    if not hasattr(obj, name):
        return None
    value = getattr(obj, name)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def _complex_attr(obj: Any, name: str) -> NDArray[np.complex128]:
    if not hasattr(obj, name):
        raise ValueError(f"result is missing {name}")
    return np.asarray(getattr(obj, name), dtype=np.complex128)


def _optional_complex_attr(obj: Any, name: str) -> NDArray[np.complex128] | None:
    if not hasattr(obj, name) or getattr(obj, name) is None:
        return None
    return np.asarray(getattr(obj, name), dtype=np.complex128)


def _float_attr(obj: Any, *names: str) -> NDArray[np.float64]:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return np.asarray(value, dtype=np.float64)
    raise ValueError(f"result is missing one of: {', '.join(names)}")


def _optional_float_attr(obj: Any, *names: str) -> NDArray[np.float64] | None:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return np.asarray(value, dtype=np.float64)
    return None
