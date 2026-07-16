from __future__ import annotations

import numpy as np

from core.region_sync import apply_nan_intervals, compute_time_range_intervals


def test_compute_intervals_maps_timestamps_to_sample_indices():
    # 10 samples, 1s spacing, starting at epoch 0.
    intervals = compute_time_range_intervals(
        time_ranges=[(2.0, 5.0)],
        npts=10,
        start_ts=0.0,
        dt_sec=1.0,
    )
    assert intervals == [(2, 5)]


def test_compute_intervals_clamps_left_to_zero():
    intervals = compute_time_range_intervals(
        time_ranges=[(-5.0, 3.0)],
        npts=10,
        start_ts=0.0,
        dt_sec=1.0,
    )
    assert intervals == [(0, 3)]


def test_compute_intervals_clamps_right_to_npts():
    intervals = compute_time_range_intervals(
        time_ranges=[(8.0, 50.0)],
        npts=10,
        start_ts=0.0,
        dt_sec=1.0,
    )
    assert intervals == [(8, 10)]


def test_compute_intervals_drops_degenerate_ranges():
    intervals = compute_time_range_intervals(
        time_ranges=[(2.0, 2.0), (3.0, 2.0)],
        npts=10,
        start_ts=0.0,
        dt_sec=1.0,
    )
    assert intervals == []


def test_compute_intervals_handles_nonzero_start_offset():
    # Dataset starts at t=100s, 0.5s spacing.
    intervals = compute_time_range_intervals(
        time_ranges=[(101.0, 102.5)],
        npts=10,
        start_ts=100.0,
        dt_sec=0.5,
    )
    assert intervals == [(2, 5)]


def test_compute_intervals_returns_empty_for_no_overlap():
    # Dataset spans t=0..10s; range is entirely after.
    intervals = compute_time_range_intervals(
        time_ranges=[(20.0, 30.0)],
        npts=10,
        start_ts=0.0,
        dt_sec=1.0,
    )
    assert intervals == []


def test_apply_nan_intervals_sets_slices_to_nan():
    values = np.arange(10, dtype=float)
    result = apply_nan_intervals(values, [(2, 5)])
    assert np.all(np.isnan(result[2:5]))
    assert np.array_equal(result[:2], values[:2])
    assert np.array_equal(result[5:], values[5:])


def test_apply_nan_intervals_does_not_mutate_input():
    values = np.arange(10, dtype=float)
    apply_nan_intervals(values, [(2, 5)])
    assert np.array_equal(values, np.arange(10, dtype=float))


def test_apply_nan_intervals_multiple_intervals():
    values = np.zeros(10)
    result = apply_nan_intervals(values, [(1, 3), (7, 9)])
    assert np.all(np.isnan(result[[1, 2, 7, 8]]))
    assert not np.any(np.isnan(result[[0, 3, 4, 5, 6, 9]]))


def test_apply_nan_intervals_empty_intervals_leaves_values_unchanged():
    values = np.arange(5, dtype=float)
    result = apply_nan_intervals(values, [])
    assert np.array_equal(result, values)
