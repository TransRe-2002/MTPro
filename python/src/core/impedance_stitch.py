"""Period-boundary assembly of two impedance results."""

from __future__ import annotations

import numpy as np

from emtf.io.edi import BandedResult


_COMPLEX_FIELDS = ("zxx", "zxy", "zyx", "zyy", "tzx", "tzy")
_ERROR_FIELDS = ("zxx_se", "zxy_se", "zyx_se", "zyy_se", "tzx_se", "tzy_se")


def stitch_banded_results(
    short_bands: BandedResult,
    long_bands: BandedResult,
    boundary_period: float,
) -> BandedResult:
    """Copy short-source periods below the boundary and long-source periods above it.

    No interpolation, averaging, or amplitude adjustment is performed. A period
    exactly equal to the boundary belongs to the long-period source.
    """
    boundary = float(boundary_period)
    if not np.isfinite(boundary) or boundary <= 0:
        raise ValueError("分界周期必须是正数。")

    short_periods = np.asarray(short_bands.periods, dtype=np.float64)
    long_periods = np.asarray(long_bands.periods, dtype=np.float64)
    short_indices = np.flatnonzero(
        np.isfinite(short_periods) & (short_periods > 0) & (short_periods < boundary)
    )
    long_indices = np.flatnonzero(
        np.isfinite(long_periods) & (long_periods > 0) & (long_periods >= boundary)
    )
    if not len(short_indices) or not len(long_indices):
        raise ValueError("分界线两侧都必须有数据点；请调整周期分界或来源。")

    selected_periods = np.concatenate((short_periods[short_indices], long_periods[long_indices]))
    order = np.argsort(selected_periods, kind="stable")
    periods = selected_periods[order]

    def copy_field(name: str, *, complex_values: bool = False, optional: bool = False):
        dtype = np.complex128 if complex_values else np.float64
        outputs = []
        for bands, indices in ((short_bands, short_indices), (long_bands, long_indices)):
            value = getattr(bands, name, None)
            if value is None:
                if not optional:
                    raise ValueError(f"阻抗结果缺少必要字段 {name}。")
                outputs.append(np.full(len(indices), np.nan, dtype=dtype))
                continue
            values = np.asarray(value, dtype=dtype)
            if values.ndim != 1 or len(values) != len(bands.periods):
                raise ValueError(f"阻抗结果字段 {name} 与周期数组长度不一致。")
            outputs.append(values[indices])
        merged = np.concatenate(outputs)
        return merged[order]

    values = {
        name: copy_field(name, complex_values=True, optional=name in ("tzx", "tzy"))
        for name in _COMPLEX_FIELDS
    }
    errors = {
        name: copy_field(name, optional=name in ("tzx_se", "tzy_se"))
        for name in _ERROR_FIELDS
    }

    def copy_observations() -> np.ndarray:
        chunks = []
        for bands, indices in ((short_bands, short_indices), (long_bands, long_indices)):
            value = getattr(bands, "n_observations", None)
            if value is None:
                chunks.append(np.zeros(len(indices), dtype=np.int32))
                continue
            values = np.asarray(value, dtype=np.int32)
            if values.ndim != 1 or len(values) != len(bands.periods):
                raise ValueError("阻抗结果字段 n_observations 与周期数组长度不一致。")
            chunks.append(values[indices])
        return np.concatenate(chunks)[order]

    # If neither source has a tipper, retain None instead of an all-NaN array.
    has_tipper = any(
        getattr(bands, field, None) is not None
        for bands in (short_bands, long_bands)
        for field in ("tzx", "tzy")
    )
    if not has_tipper:
        values["tzx"] = values["tzy"] = None
        errors["tzx_se"] = errors["tzy_se"] = None

    return BandedResult(
        periods=periods,
        **values,
        **errors,
        n_observations=copy_observations(),
        station_name=short_bands.station_name or long_bands.station_name,
        mode=short_bands.mode if short_bands.mode == long_bands.mode else "ss",
        lat=short_bands.lat,
        lon=short_bands.lon,
        elev=short_bands.elev,
        declination=short_bands.declination,
    )
