"""emtf-rs-compatible API backed by bundled EMTF binaries.

This module exposes the same high-level result and estimation surface used by
the Rust ``emtf_rs`` package while routing estimates through the original
Fortran EMTF ``dnff``/``tranmt`` pipeline bundled in this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BandConfig:
    """Period range used by the original ``emtf_rs`` banding API."""

    name: str
    min_period: float
    max_period: float


@dataclass
class SpectrumResult:
    """``emtf_rs``-style result backed by an EMTF Z-file result."""

    frequencies_hz: NDArray[np.float64]
    zxx: NDArray[np.complex128]
    zxy: NDArray[np.complex128]
    zyx: NDArray[np.complex128]
    zyy: NDArray[np.complex128]
    zxx_se: NDArray[np.float64]
    zxy_se: NDArray[np.float64]
    zyx_se: NDArray[np.float64]
    zyy_se: NDArray[np.float64]
    rho_xy: NDArray[np.float64]
    rho_yx: NDArray[np.float64]
    phi_xy: NDArray[np.float64]
    phi_yx: NDArray[np.float64]
    rho_xy_se: NDArray[np.float64]
    rho_yx_se: NDArray[np.float64]
    phi_xy_se: NDArray[np.float64]
    phi_yx_se: NDArray[np.float64]
    segment_count: int = 0
    n_observations: NDArray[np.int32] | None = None
    residual_scale_ex: NDArray[np.float64] | None = None
    residual_scale_ey: NDArray[np.float64] | None = None
    magnetic_condition_number: NDArray[np.float64] | None = None
    tzx: NDArray[np.complex128] | None = None
    tzy: NDArray[np.complex128] | None = None
    tzx_se: NDArray[np.float64] | None = None
    tzy_se: NDArray[np.float64] | None = None
    station_name: str = "unknown"
    mode: str = "ss"

    def __post_init__(self) -> None:
        n = len(self.frequencies_hz)
        if self.n_observations is None:
            self.n_observations = np.zeros(n, dtype=np.int32)
        if self.residual_scale_ex is None:
            self.residual_scale_ex = np.full(n, np.nan, dtype=np.float64)
        if self.residual_scale_ey is None:
            self.residual_scale_ey = np.full(n, np.nan, dtype=np.float64)
        if self.magnetic_condition_number is None:
            self.magnetic_condition_number = np.full(n, np.nan, dtype=np.float64)
        self.estimates = [SimpleNamespace(segment_count=self.segment_count)]

    @property
    def periods(self) -> NDArray[np.float64]:
        periods = np.full(len(self.frequencies_hz), np.nan, dtype=np.float64)
        valid = self.frequencies_hz > 0
        periods[valid] = 1.0 / self.frequencies_hz[valid]
        return periods

    def to_bands(self, band_configs: Sequence[BandConfig]):
        """Aggregate this result into the bundled EMTF export adapter."""
        from emtf.io import BandedResult

        periods = self.periods
        rows = []
        for band in band_configs:
            lo = min(float(band.min_period), float(band.max_period))
            hi = max(float(band.min_period), float(band.max_period))
            mask = np.isfinite(periods) & (periods >= lo) & (periods < hi)
            if not mask.any():
                continue
            rows.append(mask)

        if not rows:
            empty_f = np.zeros(0, dtype=np.float64)
            empty_c = np.zeros(0, dtype=np.complex128)
            return BandedResult(
                periods=empty_f,
                zxx=empty_c,
                zxy=empty_c,
                zyx=empty_c,
                zyy=empty_c,
                zxx_se=empty_f,
                zxy_se=empty_f,
                zyx_se=empty_f,
                zyy_se=empty_f,
                station_name=self.station_name,
                mode=self.mode,
            )

        def mean_complex(values: NDArray[np.complex128], mask: NDArray[np.bool_]) -> complex:
            selected = np.asarray(values[mask], dtype=np.complex128)
            finite = np.isfinite(selected.real) & np.isfinite(selected.imag)
            return complex(np.mean(selected[finite])) if finite.any() else complex(np.nan, np.nan)

        def combine_se(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
            selected = np.asarray(values[mask], dtype=np.float64)
            finite = np.isfinite(selected)
            if not finite.any():
                return float("nan")
            return float(np.sqrt(np.mean(np.square(selected[finite]))) / np.sqrt(finite.sum()))

        def mean_float(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
            selected = np.asarray(values[mask], dtype=np.float64)
            finite = np.isfinite(selected)
            return float(np.mean(selected[finite])) if finite.any() else float("nan")

        out_periods = np.array([mean_float(periods, mask) for mask in rows], dtype=np.float64)
        n_obs = np.array(
            [
                int(np.sum(np.asarray(self.n_observations)[mask]))
                for mask in rows
            ],
            dtype=np.int32,
        )

        kwargs: dict[str, Any] = {}
        if self.tzx is not None and self.tzy is not None:
            kwargs["tzx"] = np.array([mean_complex(self.tzx, mask) for mask in rows], dtype=np.complex128)
            kwargs["tzy"] = np.array([mean_complex(self.tzy, mask) for mask in rows], dtype=np.complex128)
            if self.tzx_se is not None:
                kwargs["tzx_se"] = np.array([combine_se(self.tzx_se, mask) for mask in rows], dtype=np.float64)
            if self.tzy_se is not None:
                kwargs["tzy_se"] = np.array([combine_se(self.tzy_se, mask) for mask in rows], dtype=np.float64)

        return BandedResult(
            periods=out_periods,
            zxx=np.array([mean_complex(self.zxx, mask) for mask in rows], dtype=np.complex128),
            zxy=np.array([mean_complex(self.zxy, mask) for mask in rows], dtype=np.complex128),
            zyx=np.array([mean_complex(self.zyx, mask) for mask in rows], dtype=np.complex128),
            zyy=np.array([mean_complex(self.zyy, mask) for mask in rows], dtype=np.complex128),
            zxx_se=np.array([combine_se(self.zxx_se, mask) for mask in rows], dtype=np.float64),
            zxy_se=np.array([combine_se(self.zxy_se, mask) for mask in rows], dtype=np.float64),
            zyx_se=np.array([combine_se(self.zyx_se, mask) for mask in rows], dtype=np.float64),
            zyy_se=np.array([combine_se(self.zyy_se, mask) for mask in rows], dtype=np.float64),
            n_observations=n_obs,
            station_name=self.station_name,
            mode=self.mode,
            **kwargs,
        )


def estimate_single_station_spectrum(
    ex: NDArray[np.float64],
    ey: NDArray[np.float64],
    hx: NDArray[np.float64],
    hy: NDArray[np.float64],
    sample_rate_hz: float,
    **kwargs: Any,
) -> SpectrumResult:
    from emtf.estimate import estimate_single_station as _estimate

    station_name = str(kwargs.get("station_name", "STATION"))
    zdata = _estimate(
        ex,
        ey,
        hx,
        hy,
        sample_rate_hz,
        hz=kwargs.get("hz"),
        station_name=station_name,
        window_size=kwargs.get("window_size"),
        step_size=kwargs.get("step_size"),
        max_missing_per_window=kwargs.get("max_missing_per_window", 0),
        max_gap_fill=kwargs.get("max_gap_fill", 0),
        huber_threshold=kwargs.get("huber_threshold", 1.5),
        max_iterations=kwargs.get("max_iterations", 25),
        tolerance=kwargs.get("tolerance", 1e-8),
        decimation_levels=kwargs.get("decimation_levels"),
        bands=kwargs.get("bands"),
        cleanup=kwargs.get("cleanup", True),
    )
    return _from_zfile_data(
        zdata,
        station_name=station_name,
        mode="ss",
        sample_rate_hz=sample_rate_hz,
    )


def estimate_remote_reference_spectrum(
    ex: NDArray[np.float64],
    ey: NDArray[np.float64],
    local_hx: NDArray[np.float64],
    local_hy: NDArray[np.float64],
    remote_hx: NDArray[np.float64],
    remote_hy: NDArray[np.float64],
    sample_rate_hz: float,
    **kwargs: Any,
) -> SpectrumResult:
    from emtf.estimate import estimate_remote_reference as _estimate

    station_name = str(kwargs.get("station_name", "STATION"))
    zdata = _estimate(
        ex,
        ey,
        local_hx,
        local_hy,
        remote_hx,
        remote_hy,
        sample_rate_hz,
        hz=kwargs.get("hz"),
        station_name=station_name,
        window_size=kwargs.get("window_size"),
        step_size=kwargs.get("step_size"),
        max_missing_per_window=kwargs.get("max_missing_per_window", 0),
        max_gap_fill=kwargs.get("max_gap_fill", 0),
        huber_threshold=kwargs.get("huber_threshold", 1.5),
        max_iterations=kwargs.get("max_iterations", 25),
        tolerance=kwargs.get("tolerance", 1e-8),
        decimation_levels=kwargs.get("decimation_levels"),
        bands=kwargs.get("bands"),
        cleanup=kwargs.get("cleanup", True),
    )
    return _from_zfile_data(
        zdata,
        station_name=station_name,
        mode="rr",
        sample_rate_hz=sample_rate_hz,
    )


def _from_zfile_data(
    zdata: Any,
    *,
    station_name: str,
    mode: str,
    sample_rate_hz: float = 1.0,
) -> SpectrumResult:
    from emtf.io import to_banded_result

    banded = to_banded_result(zdata, station_name=station_name, mode=mode)
    _scale_banded_periods_from_samples(banded, sample_rate_hz)
    return _from_banded_result(banded, segment_count=_segment_count(banded.n_observations))


def _scale_banded_periods_from_samples(banded: Any, sample_rate_hz: float) -> None:
    """Convert EMTF Fortran sample-period output to physical seconds.

    The bundled Fortran Z-file reports period values in sample intervals for
    non-1 Hz input.  The public emtf-rs-compatible API uses seconds/Hz, so
    convert once at the Z-file boundary before UI display, banding, and export.
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    periods = np.asarray(getattr(banded, "periods", []), dtype=np.float64)
    if periods.size == 0:
        return
    banded.periods = periods / float(sample_rate_hz)


def _from_banded_result(banded: Any, *, segment_count: int = 0) -> SpectrumResult:
    freqs = np.asarray(banded.center_frequencies_hz, dtype=np.float64)
    return SpectrumResult(
        frequencies_hz=freqs,
        zxx=np.asarray(banded.zxx, dtype=np.complex128),
        zxy=np.asarray(banded.zxy, dtype=np.complex128),
        zyx=np.asarray(banded.zyx, dtype=np.complex128),
        zyy=np.asarray(banded.zyy, dtype=np.complex128),
        zxx_se=np.asarray(banded.zxx_se, dtype=np.float64),
        zxy_se=np.asarray(banded.zxy_se, dtype=np.float64),
        zyx_se=np.asarray(banded.zyx_se, dtype=np.float64),
        zyy_se=np.asarray(banded.zyy_se, dtype=np.float64),
        rho_xy=np.asarray(banded.rho_xy, dtype=np.float64),
        rho_yx=np.asarray(banded.rho_yx, dtype=np.float64),
        phi_xy=np.asarray(banded.phi_xy, dtype=np.float64),
        phi_yx=np.asarray(banded.phi_yx, dtype=np.float64),
        rho_xy_se=np.asarray(banded.rho_xy_se, dtype=np.float64),
        rho_yx_se=np.asarray(banded.rho_yx_se, dtype=np.float64),
        phi_xy_se=np.asarray(banded.phi_xy_se, dtype=np.float64),
        phi_yx_se=np.asarray(banded.phi_yx_se, dtype=np.float64),
        n_observations=np.asarray(banded.n_observations, dtype=np.int32)
        if banded.n_observations is not None
        else None,
        segment_count=segment_count,
        tzx=np.asarray(banded.tzx, dtype=np.complex128) if banded.tzx is not None else None,
        tzy=np.asarray(banded.tzy, dtype=np.complex128) if banded.tzy is not None else None,
        tzx_se=np.asarray(banded.tzx_se, dtype=np.float64) if banded.tzx_se is not None else None,
        tzy_se=np.asarray(banded.tzy_se, dtype=np.float64) if banded.tzy_se is not None else None,
        station_name=str(getattr(banded, "station_name", "unknown")),
        mode=str(getattr(banded, "mode", "ss")),
    )


def _segment_count(values: Iterable[int] | None) -> int:
    if values is None:
        return 0
    arr = np.asarray(list(values), dtype=np.float64)
    finite = np.isfinite(arr)
    return int(np.max(arr[finite])) if finite.any() else 0


__all__ = [
    "BandConfig",
    "SpectrumResult",
    "estimate_single_station_spectrum",
    "estimate_remote_reference_spectrum",
]
