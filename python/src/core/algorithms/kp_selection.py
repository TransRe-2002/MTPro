"""Kp threshold based time selection shared by immediate and Pipeline processing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


KP_INTERVAL = pd.Timedelta(hours=3)

DEFAULT_KP_SELECTION_PARAMS: dict[str, Any] = {
    "lower_enabled": False,
    "lower_value": 0.0,
    "lower_inclusive": True,
    "upper_enabled": True,
    "upper_value": 4.0,
    "upper_inclusive": False,
    "minimum_duration_hours": 0.0,
    "trim_start_hours": 0.0,
    "trim_end_hours": 0.0,
    "missing_policy": "exclude",
}


@dataclass(frozen=True)
class KpSelectionResult:
    keep_mask: np.ndarray
    known_mask: np.ndarray
    raw_intervals: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    selected_intervals: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    raw_duration_hours: float
    selected_duration_hours: float

    @property
    def selected_samples(self) -> int:
        return int(np.count_nonzero(self.keep_mask))

    @property
    def known_samples(self) -> int:
        return int(np.count_nonzero(self.known_mask))


def normalize_kp_selection_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(DEFAULT_KP_SELECTION_PARAMS)
    if params:
        values.update(dict(params))

    values["lower_enabled"] = bool(values["lower_enabled"])
    values["upper_enabled"] = bool(values["upper_enabled"])
    values["lower_inclusive"] = bool(values["lower_inclusive"])
    values["upper_inclusive"] = bool(values["upper_inclusive"])
    for name in (
        "lower_value",
        "upper_value",
        "minimum_duration_hours",
        "trim_start_hours",
        "trim_end_hours",
    ):
        values[name] = float(values[name])

    if not values["lower_enabled"] and not values["upper_enabled"]:
        raise ValueError("Kp 下限和上限至少启用一个")
    if values["lower_enabled"] and not 0.0 <= values["lower_value"] <= 9.0:
        raise ValueError("Kp 下限必须在 0 到 9 之间")
    if values["upper_enabled"] and not 0.0 <= values["upper_value"] <= 9.0:
        raise ValueError("Kp 上限必须在 0 到 9 之间")
    if values["lower_enabled"] and values["upper_enabled"]:
        lower = values["lower_value"]
        upper = values["upper_value"]
        if lower > upper:
            raise ValueError("Kp 下限不能大于上限")
        if lower == upper and not (
            values["lower_inclusive"] and values["upper_inclusive"]
        ):
            raise ValueError("上下限相等时必须同时包含等号")
    for name, label in (
        ("minimum_duration_hours", "最短连续时长"),
        ("trim_start_hours", "区间起点裁去时长"),
        ("trim_end_hours", "区间终点裁去时长"),
    ):
        if not np.isfinite(values[name]) or values[name] < 0.0:
            raise ValueError(f"{label}不能为负数")
    if values["missing_policy"] not in {"exclude", "keep"}:
        raise ValueError("Kp 缺失时段策略必须是 exclude 或 keep")
    return values


def format_kp_condition(params: Mapping[str, Any]) -> str:
    values = normalize_kp_selection_params(params)
    parts: list[str] = []
    if values["lower_enabled"]:
        operator = "≤" if values["lower_inclusive"] else "<"
        parts.append(f"{values['lower_value']:g} {operator}")
    parts.append("Kp")
    if values["upper_enabled"]:
        operator = "≤" if values["upper_inclusive"] else "<"
        parts.append(f"{operator} {values['upper_value']:g}")
    return " ".join(parts)


def select_kp_samples(
    index: pd.DatetimeIndex,
    kp_data: pd.DataFrame | None,
    params: Mapping[str, Any] | None = None,
) -> KpSelectionResult:
    """Return the samples retained by Kp records ending each three-hour interval."""

    values = normalize_kp_selection_params(params)
    target_index = pd.DatetimeIndex(index)
    if target_index.hasnans:
        raise ValueError("数据时间轴包含无效时间")
    if not target_index.is_monotonic_increasing:
        raise ValueError("数据时间轴必须单调递增")
    if kp_data is None or kp_data.empty:
        raise ValueError("当前数据集没有 Kp 数据")
    if "Kp_datetime" not in kp_data.columns or "Kp" not in kp_data.columns:
        raise ValueError("Kp 数据缺少 Kp_datetime 或 Kp 列")

    records = pd.DataFrame({
        "end": pd.to_datetime(kp_data["Kp_datetime"], errors="coerce", utc=True),
        "value": pd.to_numeric(kp_data["Kp"], errors="coerce"),
    }).dropna(subset=["end"])
    records = records.sort_values("end").drop_duplicates("end", keep="last")

    known_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    qualifying: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for row in records.itertuples(index=False):
        try:
            kp_value = float(row.value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(kp_value):
            continue
        end = _timestamp_for_index(pd.Timestamp(row.end), target_index)
        start = end - KP_INTERVAL
        known_intervals.append((start, end))
        if _kp_value_matches(kp_value, values):
            qualifying.append((start, end))

    merged_known = _merge_intervals(known_intervals)
    raw_intervals = _merge_intervals(qualifying)
    start_trim = pd.Timedelta(hours=values["trim_start_hours"])
    end_trim = pd.Timedelta(hours=values["trim_end_hours"])
    minimum = pd.Timedelta(hours=values["minimum_duration_hours"])
    selected_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in raw_intervals:
        trimmed_start = start + start_trim
        trimmed_end = end - end_trim
        if trimmed_end <= trimmed_start:
            continue
        if trimmed_end - trimmed_start < minimum:
            continue
        selected_intervals.append((trimmed_start, trimmed_end))

    known_mask = _interval_mask(target_index, merged_known)
    keep_mask = _interval_mask(target_index, selected_intervals)
    if values["missing_policy"] == "keep":
        keep_mask |= ~known_mask

    return KpSelectionResult(
        keep_mask=keep_mask,
        known_mask=known_mask,
        raw_intervals=tuple(raw_intervals),
        selected_intervals=tuple(selected_intervals),
        raw_duration_hours=_duration_hours(raw_intervals),
        selected_duration_hours=_duration_hours(selected_intervals),
    )


def _kp_value_matches(value: float, params: Mapping[str, Any]) -> bool:
    if params["lower_enabled"]:
        lower = float(params["lower_value"])
        if params["lower_inclusive"]:
            if value < lower:
                return False
        elif value <= lower:
            return False
    if params["upper_enabled"]:
        upper = float(params["upper_value"])
        if params["upper_inclusive"]:
            if value > upper:
                return False
        elif value >= upper:
            return False
    return True


def _timestamp_for_index(timestamp: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    if index.tz is None:
        return timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.tz_convert(index.tz)


def _merge_intervals(
    intervals: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in sorted(intervals, key=lambda item: (item[0], item[1])):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _interval_mask(
    index: pd.DatetimeIndex,
    intervals: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
) -> np.ndarray:
    size = len(index)
    if size == 0 or not intervals:
        return np.zeros(size, dtype=bool)
    delta = np.zeros(size + 1, dtype=np.int32)
    for start, end in intervals:
        left = int(index.searchsorted(start, side="left"))
        right = int(index.searchsorted(end, side="left"))
        if right <= left:
            continue
        delta[left] += 1
        delta[right] -= 1
    return np.cumsum(delta[:-1]) > 0


def _duration_hours(intervals: Sequence[tuple[pd.Timestamp, pd.Timestamp]]) -> float:
    return float(sum((end - start).total_seconds() for start, end in intervals) / 3600.0)
