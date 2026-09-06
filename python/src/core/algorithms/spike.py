from __future__ import annotations

"""鲁棒窗口跳点筛选。

本模块只处理短促的脉冲/飞点，不处理持续台阶或长时段干扰：

* 每个重叠窗口以中位数作为局部基线，以 MAD 推导鲁棒标准差；
* 只将连续长度不超过 ``max_pulse_width`` 的候选段作为自动删除目标；
* 候选段两侧基线不连续时，将其视为潜在台阶并保留给去阶跃工具处理。

删除语义由调用方完成：算法仅返回索引，插件将目标位置设为 ``NaN``，
从而始终保留原始时间轴与数组长度。
"""

from dataclasses import dataclass
import math

import numpy as np


MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class AutoSpikeDetection:
    """一次自动跳点检测的结果。

    ``candidate_indices`` 包含任一窗口中超过阈值的全部点；
    ``removed_indices`` 是最终可安全置为 NaN 的短脉冲点；其余候选以
    ``long_runs`` 或 ``protected_runs`` 的形式保留给界面诊断。
    """

    candidate_indices: np.ndarray
    candidate_scores: np.ndarray
    removed_indices: np.ndarray
    long_runs: tuple[tuple[int, int], ...]
    protected_runs: tuple[tuple[int, int], ...]
    window_count: int


def detect_auto_spikes(
    samples: np.ndarray,
    *,
    window_points: int = 101,
    overlap_percent: float = 50.0,
    threshold_sigma: float = 10.0,
    max_pulse_width: int = 5,
) -> AutoSpikeDetection:
    """找出相对于重叠窗口局部基线的极端短脉冲。

    ``window_points`` 使用采样点数而非物理时间；相邻窗口的步长由
    ``overlap_percent`` 决定。一个点若在多个窗口中被识别，保留最大的
    鲁棒 Z 分数。窗口内有效样本不足一半时跳过，已有 ``NaN`` 始终不参与
    基线计算，也不会被返回为候选。
    """

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    if window_points < 3:
        raise ValueError("window_points must be >= 3")
    if not 0.0 <= overlap_percent <= 90.0:
        raise ValueError("overlap_percent must be within [0, 90]")
    if threshold_sigma <= 0.0 or not math.isfinite(threshold_sigma):
        raise ValueError("threshold_sigma must be finite and > 0")
    if max_pulse_width < 1:
        raise ValueError("max_pulse_width must be >= 1")

    size = values.size
    if size == 0:
        return _empty_detection()

    actual_window = min(int(window_points), size)
    hop = max(1, int(round(actual_window * (1.0 - overlap_percent / 100.0))))
    starts = list(range(0, size - actual_window + 1, hop))
    final_start = size - actual_window
    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    best_scores = np.full(size, -np.inf, dtype=np.float64)
    for start in starts:
        stop = start + actual_window
        segment = values[start:stop]
        finite = np.isfinite(segment)
        finite_values = segment[finite]
        if finite_values.size < max(3, int(math.ceil(actual_window * 0.5))):
            continue

        baseline = float(np.median(finite_values))
        deviations = np.abs(finite_values - baseline)
        mad = float(np.median(deviations))
        scale = MAD_TO_SIGMA * mad
        scale_floor = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(finite_values))))

        if scale <= scale_floor:
            # 完全平坦或量化到同一个值的窗口中，任意超过机器精度的偏离都
            # 属于相对局部零方差的极端点。使用 inf 分数避免除以极小数。
            local_scores = np.where(
                np.abs(segment - baseline) > scale_floor,
                np.inf,
                0.0,
            )
        else:
            local_scores = np.abs(segment - baseline) / scale
        local_scores[~finite] = -np.inf
        best_scores[start:stop] = np.maximum(best_scores[start:stop], local_scores)

    candidate_indices = np.flatnonzero(best_scores > threshold_sigma)
    if candidate_indices.size == 0:
        return AutoSpikeDetection(
            candidate_indices=np.array([], dtype=np.intp),
            candidate_scores=np.array([], dtype=np.float64),
            removed_indices=np.array([], dtype=np.intp),
            long_runs=(),
            protected_runs=(),
            window_count=len(starts),
        )

    removed: list[int] = []
    long_runs: list[tuple[int, int]] = []
    protected_runs: list[tuple[int, int]] = []
    for start, stop in _contiguous_runs(candidate_indices):
        width = stop - start + 1
        if width > max_pulse_width:
            long_runs.append((start, stop))
            continue
        if _has_continuous_neighbours(values, start, stop, max_pulse_width):
            removed.extend(range(start, stop + 1))
        else:
            # 两侧基线出现明显跳变，或候选位于记录边界。此时不自动删除，
            # 由手动删除/去阶跃工具判断，避免把真实台阶误认为飞点。
            protected_runs.append((start, stop))

    return AutoSpikeDetection(
        candidate_indices=candidate_indices.astype(np.intp, copy=False),
        candidate_scores=best_scores[candidate_indices],
        removed_indices=np.asarray(removed, dtype=np.intp),
        long_runs=tuple(long_runs),
        protected_runs=tuple(protected_runs),
        window_count=len(starts),
    )


def _empty_detection() -> AutoSpikeDetection:
    return AutoSpikeDetection(
        candidate_indices=np.array([], dtype=np.intp),
        candidate_scores=np.array([], dtype=np.float64),
        removed_indices=np.array([], dtype=np.intp),
        long_runs=(),
        protected_runs=(),
        window_count=0,
    )


def _contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    """把严格相邻的索引归并为闭区间。"""

    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = start
    for raw_index in indices[1:]:
        index = int(raw_index)
        if index != previous + 1:
            runs.append((start, previous))
            start = index
        previous = index
    runs.append((start, previous))
    return runs


def _has_continuous_neighbours(
    values: np.ndarray,
    start: int,
    stop: int,
    max_pulse_width: int,
) -> bool:
    """确认短候选段两侧处于同一基线，防止误删台阶边缘。"""

    radius = max(3, max_pulse_width)
    left = values[max(0, start - radius):start]
    right = values[stop + 1:min(values.size, stop + 1 + radius)]
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size == 0 or right.size == 0:
        return False

    left_baseline = float(np.median(left))
    right_baseline = float(np.median(right))
    support = np.concatenate((left, right))
    support_median = float(np.median(support))
    support_scale = MAD_TO_SIGMA * float(np.median(np.abs(support - support_median)))
    scale_floor = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(support))))
    if support_scale <= scale_floor:
        return abs(left_baseline - right_baseline) <= scale_floor
    return abs(left_baseline - right_baseline) <= 3.0 * support_scale
