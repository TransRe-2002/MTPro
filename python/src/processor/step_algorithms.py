from __future__ import annotations

from typing import Iterable

import numpy as np


DEFAULT_AVG_WINDOW = 1000
DEFAULT_MAX_STEPS = 20


def detect_step_indices(
    diff_abs: np.ndarray,
    min_offset: float,
    max_steps: int,
) -> np.ndarray:
    if diff_abs.size == 0 or max_steps <= 0:
        return np.array([], dtype=int)

    working = np.array(diff_abs, copy=True)
    step_indices: list[int] = []
    for _ in range(min(max_steps, working.size)):
        index = int(np.argmax(working))
        if float(working[index]) <= min_offset:
            break
        step_indices.append(index)
        working[index] = 0.0

    if not step_indices:
        return np.array([], dtype=int)
    return np.array(sorted(step_indices), dtype=int)


def zero_diff_indices(
    samples: np.ndarray,
    diff_indices: Iterable[int],
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    data = np.asarray(samples, dtype=np.float64)
    finite_mask = np.isfinite(data)
    compact = data[finite_mask]
    if compact.size < 2:
        return data.copy(), []

    diff = np.diff(compact)
    removed_steps: list[tuple[int, float]] = []
    for raw_index in sorted({int(index) for index in diff_indices}):
        if raw_index < 0 or raw_index >= diff.size:
            continue
        step_value = float(diff[raw_index])
        if step_value == 0.0:
            continue
        removed_steps.append((raw_index, step_value))
        diff[raw_index] = 0.0

    if not removed_steps:
        return data.copy(), []

    rebuilt = np.cumsum(np.concatenate(([float(compact[0])], diff)))
    corrected = data.copy()
    corrected[finite_mask] = rebuilt
    return corrected, removed_steps


def remove_diff_steps_by_count(
    samples: np.ndarray,
    n_steps: int,
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    data = np.asarray(samples, dtype=np.float64)
    finite_mask = np.isfinite(data)
    compact = data[finite_mask]
    if compact.size < 2 or n_steps <= 0:
        return data.copy(), []

    diff = np.diff(compact)
    if diff.size == 0:
        return data.copy(), []

    limit = min(int(n_steps), diff.size)
    working = np.abs(diff).astype(np.float64, copy=True)
    selected: list[int] = []
    for _ in range(limit):
        index = int(np.argmax(working))
        if float(working[index]) == 0.0:
            break
        selected.append(index)
        working[index] = 0.0

    return zero_diff_indices(data, selected)


def remove_diff_steps_by_threshold(
    samples: np.ndarray,
    min_offset: float,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    data = np.asarray(samples, dtype=np.float64)
    finite_mask = np.isfinite(data)
    compact = data[finite_mask]
    if compact.size < 2 or min_offset <= 0:
        return data.copy(), []

    step_indices = detect_step_indices(np.abs(np.diff(compact)), min_offset, max(1, int(max_steps)))
    return zero_diff_indices(data, step_indices.tolist())


def windowed_mean_destep(
    samples: np.ndarray,
    min_offset: float,
    avg_window: int | None = None,
    max_steps: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    data = np.asarray(samples, dtype=np.float64)
    finite_mask = np.isfinite(data)
    compact = data[finite_mask]
    if compact.size < 2:
        return data.copy(), []

    window = DEFAULT_AVG_WINDOW if avg_window is None else int(avg_window)
    window = max(1, window)
    max_steps = DEFAULT_MAX_STEPS if max_steps is None else int(max_steps)
    max_steps = max(1, max_steps)

    diff_abs = np.abs(np.diff(compact))
    step_indices = detect_step_indices(diff_abs, min_offset, max_steps)
    if step_indices.size == 0:
        return data.copy(), []

    nd = diff_abs.size
    start_indices = np.maximum(step_indices - window, 0)
    end_indices = np.minimum(step_indices + window + 1, nd)
    if step_indices.size > 1:
        start_indices[1:] = np.maximum(start_indices[1:], step_indices[:-1] + 1)
        end_indices[:-1] = np.minimum(end_indices[:-1], step_indices[1:])

    shift = np.zeros(compact.size, dtype=np.float64)
    applied_steps: list[tuple[int, float]] = []
    for step_index, start_index, end_index in zip(step_indices, start_indices, end_indices):
        before = compact[start_index:step_index + 1]
        after = compact[step_index + 1:end_index + 1]
        if before.size == 0 or after.size == 0:
            continue

        step_value = float(np.mean(after) - np.mean(before))
        if abs(step_value) < min_offset:
            continue

        shift[step_index + 1:] -= step_value
        applied_steps.append((int(step_index), step_value))

    corrected = data.copy()
    corrected[finite_mask] = compact + shift
    return corrected, applied_steps
