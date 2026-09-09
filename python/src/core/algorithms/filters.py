"""Small, deterministic finite-run filters used by processing plugins."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import ndimage, signal


@dataclass(frozen=True)
class FiniteRunFilterResult:
    cleaned: pd.Series
    processed_segments: tuple[tuple[int, int], ...]
    skipped_segments: tuple[tuple[int, int], ...]


def detrend_series(
    series: pd.Series,
    kind: str = "linear",
    min_segment_points: int = 4,
) -> FiniteRunFilterResult:
    """Remove a constant or linear trend without interpolating across NaNs."""

    if kind not in {"linear", "constant"}:
        raise ValueError("kind must be 'linear' or 'constant'")
    return _filter_finite_runs(
        series,
        min_segment_points,
        lambda values: signal.detrend(values, type=kind),
    )


def median_filter_series(
    series: pd.Series,
    window_points: int = 5,
    min_segment_points: int | None = None,
) -> FiniteRunFilterResult:
    """Apply a centered median filter inside every finite data run."""

    _validate_odd_window(window_points)
    return _filter_finite_runs(
        series,
        window_points if min_segment_points is None else int(min_segment_points),
        lambda values: ndimage.median_filter(values, size=window_points, mode="nearest"),
    )


def morphological_filter_series(
    series: pd.Series,
    window_points: int = 21,
    operation: str = "open_close",
    min_segment_points: int | None = None,
) -> FiniteRunFilterResult:
    """Smooth short positive/negative excursions by flat-structure morphology."""

    _validate_odd_window(window_points)
    if operation not in {"opening", "closing", "open_close", "close_open"}:
        raise ValueError("unsupported morphology operation")

    def apply(values: np.ndarray) -> np.ndarray:
        if operation == "opening":
            return ndimage.grey_opening(values, size=window_points, mode="nearest")
        if operation == "closing":
            return ndimage.grey_closing(values, size=window_points, mode="nearest")
        if operation == "open_close":
            opened = ndimage.grey_opening(values, size=window_points, mode="nearest")
            return ndimage.grey_closing(opened, size=window_points, mode="nearest")
        closed = ndimage.grey_closing(values, size=window_points, mode="nearest")
        return ndimage.grey_opening(closed, size=window_points, mode="nearest")

    return _filter_finite_runs(
        series,
        window_points if min_segment_points is None else int(min_segment_points),
        apply,
    )


def iir_filter_series(
    series: pd.Series,
    filter_kind: str = "bandstop",
    low_fraction: float = 0.1,
    high_fraction: float = 0.3,
    notch_fraction: float = 0.5,
    quality_factor: float = 30.0,
    order: int = 4,
    min_segment_points: int = 16,
) -> FiniteRunFilterResult:
    """Apply a zero-phase Butterworth or notch filter using Nyquist fractions.

    ``*_fraction`` parameters are ratios to Nyquist, in ``(0, 1)``.  This
    keeps a serialized recipe independent from the channel sampling rate.
    """

    if filter_kind not in {"lowpass", "highpass", "bandpass", "bandstop", "notch"}:
        raise ValueError("unsupported filter kind")
    if order <= 0:
        raise ValueError("order must be positive")
    if not np.isfinite(quality_factor) or quality_factor <= 0.0:
        raise ValueError("quality_factor must be finite and positive")
    for value in (low_fraction, high_fraction, notch_fraction):
        if not np.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError("cutoff fractions must be within (0, 1) of Nyquist")
    if filter_kind in {"bandpass", "bandstop"} and low_fraction >= high_fraction:
        raise ValueError("low_fraction must be lower than high_fraction")

    if filter_kind == "notch":
        numerator, denominator = signal.iirnotch(notch_fraction, quality_factor)
        sos = signal.tf2sos(numerator, denominator)
    elif filter_kind == "lowpass":
        sos = signal.butter(order, high_fraction, btype="lowpass", output="sos")
    elif filter_kind == "highpass":
        sos = signal.butter(order, low_fraction, btype="highpass", output="sos")
    else:
        sos = signal.butter(
            order,
            (low_fraction, high_fraction),
            btype=filter_kind,
            output="sos",
        )

    # scipy's default SOS pad length depends on the section layout.  Require
    # at least that many values so a short finite run is preserved untouched.
    zeros_at_origin = int(np.count_nonzero(sos[:, 2] == 0.0))
    poles_at_origin = int(np.count_nonzero(sos[:, 5] == 0.0))
    pad_length = 3 * (2 * len(sos) + 1 - min(zeros_at_origin, poles_at_origin))
    required_points = max(int(min_segment_points), pad_length + 1)
    return _filter_finite_runs(
        series,
        required_points,
        lambda values: signal.sosfiltfilt(sos, values),
    )


def _filter_finite_runs(
    series: pd.Series,
    min_segment_points: int,
    transform: Callable[[np.ndarray], np.ndarray],
) -> FiniteRunFilterResult:
    if min_segment_points < 2:
        raise ValueError("min_segment_points must be at least two")
    values = series.to_numpy(dtype=np.float64, copy=True)
    cleaned = values.copy()
    processed: list[tuple[int, int]] = []
    skipped: list[tuple[int, int]] = []
    for start, stop in _finite_runs(np.isfinite(values)):
        if stop - start < min_segment_points:
            skipped.append((start, stop))
            continue
        cleaned[start:stop] = transform(values[start:stop])
        processed.append((start, stop))
    return FiniteRunFilterResult(
        cleaned=pd.Series(cleaned, index=series.index, name=series.name),
        processed_segments=tuple(processed),
        skipped_segments=tuple(skipped),
    )


def _finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in zip(changes[::2], changes[1::2])]


def _validate_odd_window(window_points: int) -> None:
    if window_points < 3 or window_points % 2 == 0:
        raise ValueError("window_points must be an odd integer of at least three")


__all__ = [
    "FiniteRunFilterResult",
    "detrend_series",
    "median_filter_series",
    "morphological_filter_series",
    "iir_filter_series",
]
