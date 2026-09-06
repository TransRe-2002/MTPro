"""Smooth long-period background (secular-variation) subtraction.

This implements the ``TS.removeSV(nSmooth, nOffset)`` workflow used by the
legacy MATLAB tools: estimate local means at regularly-spaced anchors, smooth
those estimates with the original seven-point kernel, interpolate a smooth
baseline, and subtract it.  Missing samples stay missing in the output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


_ANCHOR_SMOOTHING_KERNEL = np.array(
    [0.0705, 0.1307, 0.1911, 0.2154, 0.1911, 0.1307, 0.0705],
    dtype=np.float64,
)


@dataclass(frozen=True)
class SecularVariationResult:
    """Result of estimating and subtracting a slowly varying baseline."""

    cleaned: pd.Series
    estimated_baseline: pd.Series
    anchor_estimates: pd.Series
    anchor_count: int
    valid_anchor_count: int
    status: str


def remove_secular_variation_series(
    series: pd.Series,
    smoothing_window_points: int = 8_760,
    anchor_spacing_points: int = 4_380,
    min_valid_fraction: float = 0.25,
) -> SecularVariationResult:
    """Subtract a cubic-spline estimate of the long-period background.

    Parameters are sample counts, matching MATLAB ``TS.removeSV``.  Each
    anchor uses a centred local-mean window; non-edge anchors with too few
    finite samples are rejected.  The first and last anchors are retained when
    they have at least one finite sample, as in the original method.

    The fitted baseline may span a missing interval, but input NaNs are never
    replaced: they remain NaN after subtraction.
    """

    _validate_parameters(
        smoothing_window_points,
        anchor_spacing_points,
        min_valid_fraction,
    )
    values = series.to_numpy(dtype=np.float64, copy=True)
    size = values.size
    anchor_positions = _anchor_positions(size, anchor_spacing_points)
    anchors = np.full(anchor_positions.size, np.nan, dtype=np.float64)

    half_window = int(round(smoothing_window_points / 2.0))
    required_samples = smoothing_window_points * float(min_valid_fraction)
    for anchor_id, position in enumerate(anchor_positions):
        start = max(0, int(position) - half_window)
        stop = min(size, int(position) + half_window + 1)
        local = values[start:stop]
        finite = local[np.isfinite(local)]
        is_edge = anchor_id == 0 or anchor_id == anchor_positions.size - 1
        if finite.size and (is_edge or finite.size > required_samples):
            anchors[anchor_id] = float(np.mean(finite))

    smoothed_anchors = _smooth_anchor_estimates(anchors)
    valid = np.isfinite(smoothed_anchors)
    baseline = np.full(size, np.nan, dtype=np.float64)
    if int(valid.sum()) >= 2:
        baseline = _interpolate_baseline(
            anchor_positions[valid],
            smoothed_anchors[valid],
            size,
        )
        cleaned = values - baseline
        cleaned[~np.isfinite(values)] = np.nan
        status = "applied"
    else:
        cleaned = values.copy()
        status = "insufficient_anchor_estimates"

    anchor_index = series.index.take(anchor_positions)
    return SecularVariationResult(
        cleaned=pd.Series(cleaned, index=series.index, name=series.name),
        estimated_baseline=pd.Series(baseline, index=series.index, name=f"{series.name}_sv"),
        anchor_estimates=pd.Series(smoothed_anchors, index=anchor_index, name=f"{series.name}_sv_anchors"),
        anchor_count=int(anchor_positions.size),
        valid_anchor_count=int(valid.sum()),
        status=status,
    )


def _anchor_positions(size: int, spacing: int) -> np.ndarray:
    if size == 0:
        return np.empty(0, dtype=np.int64)
    positions = np.arange(0, size, spacing, dtype=np.int64)
    if positions[-1] != size - 1:
        positions = np.append(positions, size - 1)
    return positions


def _smooth_anchor_estimates(anchors: np.ndarray) -> np.ndarray:
    """Apply MATLAB's normalized seven-point convolution without filling gaps."""

    if anchors.size == 0:
        return anchors.copy()
    valid = np.isfinite(anchors)
    numerator = _same_length_convolution(np.where(valid, anchors, 0.0), _ANCHOR_SMOOTHING_KERNEL)
    denominator = _same_length_convolution(valid.astype(np.float64), _ANCHOR_SMOOTHING_KERNEL)
    smoothed = np.divide(
        numerator,
        denominator,
        out=np.full(anchors.size, np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )
    smoothed[~valid] = np.nan
    return smoothed


def _same_length_convolution(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Return the central convolution part with ``len(values)`` entries."""

    full = np.convolve(values, kernel, mode="full")
    first = (kernel.size - 1) // 2
    return full[first:first + values.size]


def _interpolate_baseline(
    positions: np.ndarray,
    estimates: np.ndarray,
    size: int,
) -> np.ndarray:
    target = np.arange(size, dtype=np.float64)
    if positions.size == 2:
        return np.interp(target, positions, estimates)
    return CubicSpline(positions, estimates, extrapolate=True)(target)


def _validate_parameters(
    smoothing_window_points: int,
    anchor_spacing_points: int,
    min_valid_fraction: float,
) -> None:
    if smoothing_window_points < 1:
        raise ValueError("smoothing_window_points must be positive")
    if anchor_spacing_points < 1:
        raise ValueError("anchor_spacing_points must be positive")
    if not np.isfinite(min_valid_fraction) or not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be within [0, 1]")


__all__ = ["SecularVariationResult", "remove_secular_variation_series"]
