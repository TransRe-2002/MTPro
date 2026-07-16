from __future__ import annotations

import numpy as np


def compute_time_range_intervals(
    time_ranges: list[tuple[float, float]],
    npts: int,
    start_ts: float,
    dt_sec: float,
) -> list[tuple[int, int]]:
    """Compute sample-index intervals from UTC timestamp ranges for one dataset.

    Mirrors the per-dataset index math used by RemoveRegion cross-dataset sync:
    each ``(t_start, t_end)`` pair (UTC seconds) maps to ``(left, right)`` sample
    indices, clamped to ``[0, npts]``. Empty/degenerate intervals are dropped.
    """
    intervals: list[tuple[int, int]] = []
    for t_start, t_end in time_ranges:
        left = int((t_start - start_ts) / dt_sec)
        right = int((t_end - start_ts) / dt_sec)
        left = max(0, min(left, npts))
        right = max(0, min(right, npts))
        if right > left:
            intervals.append((left, right))
    return intervals


def apply_nan_intervals(values: np.ndarray, intervals: list[tuple[int, int]]) -> np.ndarray:
    """Return a copy of ``values`` with each ``[left:right]`` slice set to NaN."""
    out = values.copy()
    for left, right in intervals:
        out[left:right] = np.nan
    return out