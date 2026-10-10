from __future__ import annotations

"""处理器共用的底层算法函数。

这里放的是“纯数值逻辑”：
- 差分去阶跃
- 窗口均值去阶跃
- 鲁棒基线去阶跃

上层各个 QWidget 工具主要负责：
- 收集参数
- 调用这些算法
- 画图、日志和确认写回
"""

from dataclasses import dataclass
import logging
from collections.abc import Iterable

import numpy as np
from scipy.optimize import least_squares


logger = logging.getLogger(__name__)


DEFAULT_AVG_WINDOW = 1000
DEFAULT_MAX_STEPS = 20


try:
    import rust_lib
    _HAS_RUST_STEP = hasattr(rust_lib, "detect_step_indices_rs")
except ImportError:
    _HAS_RUST_STEP = False
    rust_lib = None  # type: ignore

if _HAS_RUST_STEP:
    _HAS_RUST_WINDOW = hasattr(rust_lib, "windowed_mean_destep_rs")
else:
    _HAS_RUST_WINDOW = False


@dataclass(frozen=True)
class WindowedStepCandidate:
    """窗口均值去阶跃时，一个候选台阶的诊断信息。"""
    index: int
    raw_diff: float
    step_value: float | None
    before_count: int
    after_count: int
    applied: bool


@dataclass(frozen=True)
class RobustBaselineStepResult:
    """鲁棒基线去阶跃的完整拟合结果。"""
    corrected: np.ndarray
    correction: np.ndarray
    baseline: np.ndarray
    fit_window: tuple[int, int]
    region: tuple[int, int]
    baseline_coeffs: np.ndarray
    transition_coeffs: np.ndarray
    step_height: float
    success: bool
    cost: float
    message: str


def detect_step_indices(
    diff_abs: np.ndarray,
    min_offset: float,
    max_steps: int,
) -> np.ndarray:
    """在差分绝对值序列中找出最显著的若干台阶位置。

    算法很直接：每次取当前最大差分，如果它大于阈值就记录下来，
    然后把该位置置零，继续找下一个，直到达到数量上限或没有超过阈值的点。

    当 ``rust_lib`` 可用时自动调用 Rust 实现（同样的 argmax+neutralize 循环
    但无 Python 解释器开销，性能显著提升）。
    """
    if diff_abs.size == 0 or max_steps <= 0:
        return np.array([], dtype=int)

    if _HAS_RUST_STEP:
        try:
            result = rust_lib.detect_step_indices_rs(
                np.ascontiguousarray(diff_abs, dtype=np.float64),
                float(min_offset), int(max_steps),
            )
            return np.asarray(result, dtype=np.intp)
        except Exception:
            logger.debug("Rust detect_step_indices_rs failed, falling back to Python", exc_info=True)

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
    """把给定的一阶差分位置强制置零，并重建原序列。

    这相当于假设这些差分尖峰对应的是“台阶跳变”，
    去掉差分后再做累积和，就能得到消除了台阶的新序列。
    """
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
    """按“最大差分的前 n 个”去除台阶。"""
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
    """按差分阈值去除台阶。"""
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
    """窗口均值去阶跃的简化入口。

    这里只返回校正后的结果和实际应用的台阶；
    更详细的候选诊断信息由带 diagnostics 的版本返回。
    """
    corrected, applied_steps, _ = windowed_mean_destep_with_diagnostics(
        samples,
        min_offset,
        avg_window=avg_window,
        max_steps=max_steps,
    )
    return corrected, applied_steps


def windowed_mean_destep_with_diagnostics(
    samples: np.ndarray,
    min_offset: float,
    avg_window: int | None = None,
    max_steps: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, float]], list[WindowedStepCandidate]]:
    """利用台阶前后窗口均值差估计台阶幅度，并返回诊断信息。

    核心思想：
    1. 先通过差分绝对值找候选台阶位置。
    2. 对每个候选点，分别在其前后取局部窗口。
    3. 用“后窗均值 - 前窗均值”估计台阶高度。
    4. 只有当估计出的台阶高度仍然超过阈值时才真正应用。
    """
    data = np.asarray(samples, dtype=np.float64)
    finite_mask = np.isfinite(data)
    compact = data[finite_mask]
    if compact.size < 2:
        return data.copy(), [], []

    window = DEFAULT_AVG_WINDOW if avg_window is None else int(avg_window)
    window = max(1, window)
    max_steps = DEFAULT_MAX_STEPS if max_steps is None else int(max_steps)
    max_steps = max(1, max_steps)

    diff_abs = np.abs(np.diff(compact))
    step_indices = detect_step_indices(diff_abs, min_offset, max_steps)
    if step_indices.size == 0:
        return data.copy(), [], []

    # 相邻台阶过近时，前后窗口会互相挤占，这里对窗口范围做裁剪，
    # 尽量避免一个台阶的窗口跨到另一个台阶附近。
    nd = diff_abs.size
    start_indices = np.maximum(step_indices - window, 0)
    end_indices = np.minimum(step_indices + window + 1, nd)
    if step_indices.size > 1:
        start_indices[1:] = np.maximum(start_indices[1:], step_indices[:-1] + 1)
        end_indices[:-1] = np.minimum(end_indices[:-1], step_indices[1:])

    # Rust 加速路径：O(N+K) 单趟前缀和构造 shift，避免 Python 循环的 O(N·K) 尾写。
    if _HAS_RUST_WINDOW:
        try:
            shift_np, applied_steps, candidates_rs = rust_lib.windowed_mean_destep_rs(
                np.ascontiguousarray(compact, dtype=np.float64),
                np.ascontiguousarray(diff_abs, dtype=np.float64),
                np.ascontiguousarray(step_indices, dtype=np.int64),
                np.ascontiguousarray(start_indices, dtype=np.int64),
                np.ascontiguousarray(end_indices, dtype=np.int64),
                float(min_offset),
            )
            shift = np.asarray(shift_np, dtype=np.float64)
            candidates = [
                WindowedStepCandidate(
                    index=int(c.index), raw_diff=float(c.raw_diff),
                    step_value=c.step_value, before_count=int(c.before_count),
                    after_count=int(c.after_count), applied=bool(c.applied),
                )
                for c in candidates_rs
            ]
            corrected = data.copy()
            corrected[finite_mask] = compact + shift
            return corrected, applied_steps, candidates
        except Exception:
            logger.debug("Rust windowed_mean_destep_rs failed, falling back to Python", exc_info=True)

    # shift 表示从某个台阶位置之后整段都要一起平移的补偿量。
    shift = np.zeros(compact.size, dtype=np.float64)
    applied_steps: list[tuple[int, float]] = []
    candidates: list[WindowedStepCandidate] = []
    for step_index, start_index, end_index in zip(step_indices, start_indices, end_indices):
        before = compact[start_index:step_index + 1]
        after = compact[step_index + 1:end_index + 1]
        if before.size == 0 or after.size == 0:
            candidates.append(
                WindowedStepCandidate(
                    index=int(step_index),
                    raw_diff=float(diff_abs[step_index]),
                    step_value=None,
                    before_count=int(before.size),
                    after_count=int(after.size),
                    applied=False,
                )
            )
            continue

        step_value = float(np.mean(after) - np.mean(before))
        if abs(step_value) < min_offset:
            candidates.append(
                WindowedStepCandidate(
                    index=int(step_index),
                    raw_diff=float(diff_abs[step_index]),
                    step_value=step_value,
                    before_count=int(before.size),
                    after_count=int(after.size),
                    applied=False,
                )
            )
            continue

        shift[step_index + 1:] -= step_value
        applied_steps.append((int(step_index), step_value))
        candidates.append(
            WindowedStepCandidate(
                index=int(step_index),
                raw_diff=float(diff_abs[step_index]),
                step_value=step_value,
                before_count=int(before.size),
                after_count=int(after.size),
                applied=True,
            )
        )

    corrected = data.copy()
    corrected[finite_mask] = compact + shift
    return corrected, applied_steps, candidates


def _normalize_index(index: np.ndarray, start: int, stop: int) -> np.ndarray:
    """把索引区间归一化到近似 [-1, 1]，便于多项式拟合数值稳定。"""
    center = 0.5 * (start + stop)
    scale = max(1.0, 0.5 * (stop - start))
    return (index - center) / scale


def _poly_design(index: np.ndarray, start: int, stop: int, degree: int) -> np.ndarray:
    """构造多项式设计矩阵。"""
    x = _normalize_index(index.astype(np.float64), start, stop)
    cols = [x ** power for power in range(degree + 1)]
    return np.column_stack(cols)


def _saturated_transition(index: np.ndarray, left: int, right: int, degree: int) -> np.ndarray:
    if degree <= 0:
        raise ValueError("transition degree must be positive")

    if right <= left:
        t = (index >= left).astype(np.float64)
    else:
        t = np.zeros_like(index, dtype=np.float64)
        inside = (index >= left) & (index <= right)
        t[index > right] = 1.0
        t[inside] = (index[inside] - left) / float(right - left)

    cols = [t ** power for power in range(1, degree + 1)]
    return np.column_stack(cols)


def _transition_smoothness_design(degree: int, resolution: int = 64) -> np.ndarray:
    if degree <= 1:
        return np.zeros((0, max(1, degree)), dtype=np.float64)

    resolution = max(3, int(resolution))
    t = np.linspace(0.0, 1.0, num=resolution, dtype=np.float64)
    cols = [t ** power for power in range(1, degree + 1)]
    design = np.column_stack(cols)
    smoothness_design = np.diff(design, n=2, axis=0)

    step = float(t[1] - t[0]) if resolution > 1 else 1.0
    if step > 0.0:
        smoothness_design /= step ** 2
    return smoothness_design


def robust_baseline_destep(
    samples: np.ndarray,
    region: tuple[int, int],
    *,
    baseline_degree: int = 1,
    transition_degree: int = 2,
    context_points: int = 200,
    loss: str = "soft_l1",
    f_scale: float | None = None,
    transition_smoothness: float | None = None,
) -> RobustBaselineStepResult:
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError("samples must be a 1D array")
    if data.size < 3:
        raise ValueError("samples is too short")

    baseline_degree = min(1, max(0, int(baseline_degree)))
    transition_degree = min(10, max(1, int(transition_degree)))
    context_points = max(1, int(context_points))

    left, right = sorted((int(region[0]), int(region[1])))
    left = max(0, left)
    right = min(data.size - 1, right)
    if left == right:
        right = min(data.size - 1, left + 1)

    fit_start = max(0, left - context_points)
    fit_stop = min(data.size, right + context_points + 1)
    fit_index = np.arange(fit_start, fit_stop, dtype=np.int64)
    if fit_index.size < 3:
        raise ValueError("fit window is too short")

    fit_values = data[fit_start:fit_stop]
    finite_mask = np.isfinite(fit_values)
    pre_mask = finite_mask & (fit_index < left)
    if int(np.count_nonzero(pre_mask)) < 2:
        raise ValueError("not enough finite pre-step samples for baseline fit")

    max_pre_degree = max(0, int(np.count_nonzero(pre_mask)) - 1)
    effective_baseline_degree = min(baseline_degree, max_pre_degree)
    baseline_design = _poly_design(
        fit_index, fit_start, max(fit_start, left - 1), effective_baseline_degree,
    )
    baseline_fit_design = baseline_design[pre_mask]
    baseline_target = fit_values[pre_mask]
    baseline_initial, *_ = np.linalg.lstsq(baseline_fit_design, baseline_target, rcond=None)
    if f_scale is None:
        baseline_diff = np.diff(baseline_target)
        if baseline_diff.size:
            mad = np.median(np.abs(baseline_diff - np.median(baseline_diff)))
            robust_scale = mad * 1.4826
        else:
            robust_scale = 0.0
        f_scale = max(float(robust_scale), 1e-3)

    baseline_result = least_squares(
        lambda coeffs: baseline_fit_design @ coeffs - baseline_target,
        baseline_initial,
        loss=loss,
        f_scale=float(f_scale),
    )
    baseline_coeffs = baseline_result.x
    baseline_fit_values = baseline_design @ baseline_coeffs

    # Fit the signed step correction against the residual relative to the pre-step baseline.
    transition_design = _saturated_transition(fit_index, left, right, transition_degree)
    transition_mask = finite_mask & (fit_index >= left)
    transition_parameter_count = transition_design.shape[1]
    if int(np.count_nonzero(transition_mask)) < transition_parameter_count:
        raise ValueError("not enough finite post-step samples for transition fit")

    transition_fit_design = transition_design[transition_mask]
    transition_target = fit_values[transition_mask] - baseline_fit_values[transition_mask]
    transition_initial, *_ = np.linalg.lstsq(
        transition_fit_design, transition_target, rcond=None,
    )
    transition_diff = np.diff(transition_target)
    if transition_diff.size:
        mad_t = np.median(np.abs(transition_diff - np.median(transition_diff)))
        transition_scale = mad_t * 1.4826
    else:
        transition_scale = 0.0
    transition_f_scale = max(float(transition_scale), 1e-3)
    if transition_smoothness is None:
        if transition_degree <= 2:
            transition_smoothness = 0.0
        else:
            transition_smoothness = 1e-5 * float((transition_degree - 1) ** 2)
    transition_smoothness = max(0.0, float(transition_smoothness))
    smoothness_design = _transition_smoothness_design(transition_degree)

    def _transition_residuals(coeffs: np.ndarray) -> np.ndarray:
        residuals = transition_fit_design @ coeffs - transition_target
        if transition_smoothness <= 0.0 or smoothness_design.size == 0:
            return residuals
        smoothness_residuals = np.sqrt(transition_smoothness) * (smoothness_design @ coeffs)
        return np.concatenate((residuals, smoothness_residuals))

    transition_result = least_squares(
        _transition_residuals,
        transition_initial,
        loss=loss,
        f_scale=transition_f_scale,
    )
    transition_coeffs = transition_result.x

    all_index = np.arange(data.size, dtype=np.int64)
    baseline = np.full(data.size, np.nan, dtype=np.float64)
    baseline[fit_start:fit_stop] = baseline_fit_values
    correction = _saturated_transition(
        all_index, left, right, transition_degree,
    ) @ transition_coeffs
    correction[:left] = 0.0
    if right + 1 < correction.size:
        correction[right + 1:] = correction[right]

    corrected = data.copy()
    finite_all = np.isfinite(corrected)
    corrected[finite_all] = corrected[finite_all] - correction[finite_all]

    before_value = float(correction[left - 1]) if left > 0 else 0.0
    after_index = min(data.size - 1, right + 1)
    step_height = float(correction[after_index] - before_value)

    return RobustBaselineStepResult(
        corrected=corrected,
        correction=correction,
        baseline=baseline,
        fit_window=(fit_start, fit_stop - 1),
        region=(left, right),
        baseline_coeffs=np.asarray(baseline_coeffs, dtype=np.float64),
        transition_coeffs=np.asarray(transition_coeffs, dtype=np.float64),
        step_height=step_height,
        success=bool(baseline_result.success and transition_result.success),
        cost=float(baseline_result.cost + transition_result.cost),
        message=f"baseline: {baseline_result.message}; transition: {transition_result.message}",
    )
