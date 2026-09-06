"""JSON-friendly exact edit encoding used by migrated interactive editors."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def encode_edit(before, after) -> dict[str, Any]:
    source = np.asarray(before, dtype=np.float64)
    result = np.asarray(after, dtype=np.float64)
    if source.ndim != 1 or result.shape != source.shape:
        raise ValueError("editor result must be a one-dimensional equal-length array")

    same = (source == result) | (np.isnan(source) & np.isnan(result))
    changed = ~same
    new_nan = changed & np.isnan(result)
    finite_delta = changed & np.isfinite(source) & np.isfinite(result)
    absolute = changed & ~new_nan & ~finite_delta

    delta_runs: list[list[Any]] = []
    indices = np.flatnonzero(finite_delta)
    if indices.size:
        run_start = 0
        for position in range(1, indices.size + 1):
            if position == indices.size or indices[position] != indices[position - 1] + 1:
                selected = indices[run_start:position]
                values = (result[selected] - source[selected]).tolist()
                delta_runs.append([int(selected[0]), values])
                run_start = position

    absolute_values = [
        [int(index), float(result[index])]
        for index in np.flatnonzero(absolute)
    ]
    return {
        "nan_indices": np.flatnonzero(new_nan).astype(int).tolist(),
        "delta_runs": delta_runs,
        "absolute": absolute_values,
    }


def apply_edit(samples, edit: Mapping[str, Any]):
    values = np.asarray(samples, dtype=np.float64).copy()
    for raw_index in edit.get("nan_indices", []):
        index = int(raw_index)
        if 0 <= index < values.size:
            values[index] = np.nan
    for raw_start, raw_deltas in edit.get("delta_runs", []):
        start = int(raw_start)
        deltas = np.asarray(raw_deltas, dtype=np.float64)
        stop = min(values.size, start + deltas.size)
        if start < 0 or start >= stop:
            continue
        values[start:stop] += deltas[:stop - start]
    for raw_index, raw_value in edit.get("absolute", []):
        index = int(raw_index)
        if 0 <= index < values.size:
            values[index] = float(raw_value)
    return values


def apply_operations(samples, operations):
    values = np.asarray(samples, dtype=np.float64).copy()
    for edit in operations or []:
        values = apply_edit(values, edit)
    return values
