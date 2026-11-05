"""Shared interpolation helpers for denoisers that require dense input."""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MAX_CONSECUTIVE_NANS = 10


def interpolate_nonfinite(
    values: np.ndarray,
    max_consecutive_nans: int = DEFAULT_MAX_CONSECUTIVE_NANS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a linearly interpolated dense copy and the original finite mask.

    Only interior gaps no longer than ``max_consecutive_nans`` are interpolated.
    Edge gaps and longer gaps remain non-finite so callers can trim edges or
    treat long gaps as processing boundaries.
    """
    if int(max_consecutive_nans) < 0:
        raise ValueError("max_consecutive_nans must be non-negative")
    source = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(source)
    dense = source.copy()
    if finite.any() and not finite.all():
        interpolation_source = source.copy()
        interpolation_source[~finite] = np.nan
        dense = pd.Series(interpolation_source).interpolate(
            method="linear",
            limit_area="inside",
        ).to_numpy(dtype=np.float64, copy=True)
        gap_edges = np.flatnonzero(np.diff(np.r_[False, ~finite, False].astype(np.int8)))
        for gap_start, gap_stop in zip(gap_edges[::2], gap_edges[1::2]):
            if gap_stop - gap_start > int(max_consecutive_nans):
                dense[gap_start:gap_stop] = np.nan
    return dense, finite


def restore_nonfinite(values: np.ndarray, original: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """Restore every originally non-finite sample after a dense transform."""
    result = np.asarray(values, dtype=np.float64).copy()
    source = np.asarray(original, dtype=np.float64)
    result[~finite] = source[~finite]
    return result


def finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open contiguous true runs from a one-dimensional mask."""
    values = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], values, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in zip(edges[::2], edges[1::2])]


__all__ = [
    "DEFAULT_MAX_CONSECUTIVE_NANS",
    "finite_runs",
    "interpolate_nonfinite",
    "restore_nonfinite",
]
