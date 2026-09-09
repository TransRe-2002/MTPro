from __future__ import annotations

"""处理器共用的底层算法函数。

这里放的是“纯数值逻辑”：
- 差分去阶跃
- 窗口均值去阶跃
- 鲁棒基线去阶跃
- 小波多尺度去噪

上层各个 QWidget 工具主要负责：
- 收集参数
- 调用这些算法
- 画图、日志和确认写回
"""

from dataclasses import dataclass
import logging
import math
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


@dataclass(frozen=True)
class WaveletDenoiseResult:
    """小波去噪一次处理后的输出结果。"""
    corrected: np.ndarray
    correction: np.ndarray
    local_window: tuple[int, int]
    region: tuple[int, int]
    local_original: np.ndarray
    local_reconstructed: np.ndarray
    detail_energy_before: np.ndarray
    detail_energy_after: np.ndarray
    band_labels: list[str]
    band_ranges_hz: list[tuple[float, float]]
    wavelet_name: str
    transform_kind: str
    level: int
    threshold_scale: float
    threshold_method: str
    level_policy: str
    protect_period_seconds: float | None
    padding_points: int
    thresholds: np.ndarray
    detail_gains: np.ndarray
    mode: str
    changed_coefficients: int
    success: bool
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


def _import_pywt():
    try:
        import pywt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("当前环境未安装 PyWavelets，无法执行小波多尺度去噪") from exc
    return pywt


def _format_band_range(low_hz: float, high_hz: float) -> str:
    return f"{low_hz:.4g}-{high_hz:.4g} Hz"


def _dwt_band_ranges(level: int, sample_rate_hz: float) -> list[tuple[float, float]]:
    nyquist = 0.5 * sample_rate_hz
    ranges: list[tuple[float, float]] = []
    for detail_level in range(1, level + 1):
        high = nyquist / (2 ** (detail_level - 1))
        low = nyquist / (2 ** detail_level)
        ranges.append((low, high))
    return ranges


def _wp_band_ranges(level: int, sample_rate_hz: float) -> list[tuple[float, float]]:
    nyquist = 0.5 * sample_rate_hz
    band_count = 2 ** level
    width = nyquist / float(band_count)
    return [(i * width, (i + 1) * width) for i in range(band_count)]


def _mad_noise_sigma(coefficients: np.ndarray) -> float:
    """Estimate Gaussian noise scale from wavelet coefficients with MAD.

    The median is removed before computing the MAD.  That matters for local
    windows whose selected event leaks into a coarse detail band.
    """
    values = np.asarray(coefficients, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1e-12

    median = float(np.median(values))
    sigma = float(np.median(np.abs(values - median)) / 0.67448975)
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(values))
    return max(sigma, 1e-12)


def _universal_threshold(sigma: float, coefficient_count: int) -> float:
    return float(sigma) * math.sqrt(2.0 * math.log(max(2, int(coefficient_count))))


def _sure_threshold(coefficients: np.ndarray, sigma: float) -> float:
    """Return the SureShrink soft-threshold for one coefficient band.

    SURE is evaluated over the finite reference coefficients (normally the
    context around the selected region).  The result is capped by the
    universal threshold, as in SureShrink, so the adaptive rule never becomes
    more aggressive merely because a short context happened to be unusual.
    """
    values = np.asarray(coefficients, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0

    normalized = np.sort(np.abs(values) / max(float(sigma), 1e-12))
    squared = normalized * normalized
    counts = np.arange(1, normalized.size + 1, dtype=np.float64)
    risks = (
        float(normalized.size)
        - 2.0 * counts
        + np.cumsum(squared)
        + (float(normalized.size) - counts) * squared
    )
    # Threshold zero is a valid SURE candidate as well.
    candidate_values = np.concatenate((np.array([0.0]), normalized))
    candidate_risks = np.concatenate((np.array([float(normalized.size)]), risks))
    sure_value = float(candidate_values[int(np.argmin(candidate_risks))] * sigma)
    return min(sure_value, _universal_threshold(sigma, normalized.size))


def _threshold_for_band(
    reference_coefficients: np.ndarray,
    *,
    threshold_scale: float,
    threshold_method: str,
) -> float:
    sigma = _mad_noise_sigma(reference_coefficients)
    if threshold_method == "universal":
        base_threshold = _universal_threshold(sigma, reference_coefficients.size)
    elif threshold_method == "sure":
        base_threshold = _sure_threshold(reference_coefficients, sigma)
    else:  # checked by the caller; keeps this helper safe as a public-test seam.
        raise ValueError(f"unsupported wavelet threshold method: {threshold_method}")
    return float(threshold_scale) * base_threshold


def _level_for_protected_period(
    sample_rate_hz: float,
    protect_period_seconds: float,
) -> int:
    """Choose a DWT/SWT level whose approximation retains periods at least P.

    After level ``J`` the approximation approximately covers
    ``[0, fs / 2**(J + 1)]``.  Choosing ``J`` from this relation lets the UI
    ask for a physical protection period instead of a sampling-rate-dependent
    fixed level.
    """
    period = float(protect_period_seconds)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("protect_period_seconds must be a positive finite number")
    samples_per_period = max(2.0, float(sample_rate_hz) * period)
    return max(1, int(math.ceil(math.log2(samples_per_period))) - 1)


def wavelet_multiscale_denoise(
    samples: np.ndarray,
    region: tuple[int, int],
    *,
    transform_kind: str = "dwt",
    wavelet_name: str = "db4",
    level: int = 6,
    sample_rate_hz: float | None = None,
    threshold_scale: float = 1.0,
    padding_points: int = 256,
    padding_seconds: float | None = None,
    mode: str = "soft",
    threshold_method: str = "universal",
    level_policy: str = "manual",
    protect_period_seconds: float | None = None,
    detail_gains: Iterable[float] | None = None,
) -> WaveletDenoiseResult:
    pywt = _import_pywt()

    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError("samples must be a 1D array")
    if data.size < 16:
        raise ValueError("samples is too short")

    left, right = sorted((int(region[0]), int(region[1])))
    left = max(0, left)
    right = min(data.size - 1, right)
    if right <= left:
        right = min(data.size - 1, left + 1)

    threshold_scale = float(threshold_scale)
    if threshold_scale < 0.0:
        raise ValueError("threshold_scale must be >= 0")

    if mode not in {"soft", "hard", "zero", "manual"}:
        raise ValueError(f"unsupported wavelet denoise mode: {mode}")
    if transform_kind not in {"dwt", "swt", "wp"}:
        raise ValueError(f"unsupported transform kind: {transform_kind}")
    if threshold_method not in {"universal", "sure"}:
        raise ValueError(f"unsupported wavelet threshold method: {threshold_method}")
    if level_policy not in {"manual", "auto_by_period"}:
        raise ValueError(f"unsupported wavelet level policy: {level_policy}")
    if level_policy == "auto_by_period" and transform_kind == "wp":
        raise ValueError("automatic protection-period levels are only available for DWT or SWT")
    if sample_rate_hz is None or not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        sample_rate_hz = 1.0

    selected_width = right - left + 1
    if padding_seconds is not None:
        padding_seconds = float(padding_seconds)
        if not np.isfinite(padding_seconds) or padding_seconds < 0.0:
            raise ValueError("padding_seconds must be a non-negative finite number")
        padding_points = int(math.ceil(padding_seconds * float(sample_rate_hz)))
    padding_points = max(int(padding_points), 8)
    half_window = max(padding_points, int(np.ceil(selected_width * 0.5)))
    local_start = max(0, left - half_window)
    local_stop = min(data.size, right + half_window + 1)
    original_local_segment = np.array(data[local_start:local_stop], copy=True, dtype=np.float64)
    if original_local_segment.size < 16:
        raise ValueError("local window is too short for wavelet processing")

    finite_mask = np.isfinite(original_local_segment)
    local_segment = np.array(original_local_segment, copy=True)
    if not np.all(finite_mask):
        finite_indices = np.flatnonzero(finite_mask)
        if finite_indices.size < 2:
            raise ValueError("not enough finite samples in local window")
        local_segment = np.interp(
            np.arange(local_segment.size, dtype=np.float64),
            finite_indices.astype(np.float64),
            local_segment[finite_mask],
        )

    wavelet = pywt.Wavelet(wavelet_name)
    if transform_kind == "swt":
        max_level = int(math.floor(math.log2(local_segment.size)))
    else:
        max_level = pywt.dwt_max_level(local_segment.size, wavelet.dec_len)
    if max_level < 1:
        raise ValueError("local window is too short for the requested wavelet")
    if level_policy == "auto_by_period":
        if protect_period_seconds is None:
            raise ValueError("protect_period_seconds is required for automatic levels")
        level = _level_for_protected_period(float(sample_rate_hz), protect_period_seconds)
    level = max(1, min(int(level), int(max_level)))

    changed_coefficients = 0
    local_left = left - local_start
    local_right = right - local_start

    if transform_kind == "dwt":
        coeffs = pywt.wavedec(local_segment, wavelet, mode="symmetric", level=level)
        modified_coeffs = [np.array(coeffs[0], copy=True)]
        detail_energy_before_desc: list[float] = []
        detail_energy_after_desc: list[float] = []
        thresholds_desc: list[float] = []
        band_ranges_hz = _dwt_band_ranges(level, float(sample_rate_hz))
        band_labels = [
            f"D{i} [{_format_band_range(*band_ranges_hz[i - 1])}]"
            for i in range(1, level + 1)
        ]
        gains_array = np.ones(level, dtype=np.float64)
        if detail_gains is not None:
            requested_gains = np.asarray(list(detail_gains), dtype=np.float64)
            if requested_gains.size > 0:
                usable = min(level, requested_gains.size)
                gains_array[:usable] = requested_gains[:usable]

        for coeff_index in range(1, len(coeffs)):
            detail_level = level - coeff_index + 1
            detail = np.array(coeffs[coeff_index], copy=True, dtype=np.float64)
            scale = 2 ** detail_level
            guard = scale
            coeff_left = max(0, int(np.floor((local_left - guard) / float(scale))))
            coeff_right = min(
                detail.size,
                int(np.ceil((local_right + 1 + guard) / float(scale))),
            )

            detail_energy_before_desc.append(float(np.sum(detail * detail)))
            if mode == "manual":
                ui_level = len(coeffs) - coeff_index
                if 1 <= ui_level <= gains_array.size:
                    gain = float(gains_array[ui_level - 1])
                else:
                    gain = 1.0
                updated = detail * gain
                threshold = 0.0
            else:
                background = np.concatenate((detail[:coeff_left], detail[coeff_right:]))
                if background.size == 0:
                    background = detail
                threshold = _threshold_for_band(
                    background,
                    threshold_scale=threshold_scale,
                    threshold_method=threshold_method,
                )
                if mode == "zero":
                    updated = np.zeros_like(detail)
                else:
                    updated = pywt.threshold(detail, value=float(threshold), mode=mode)

            changed_coefficients += int(np.count_nonzero(np.abs(updated - detail) > 1e-12))
            detail = np.asarray(updated, dtype=np.float64)
            modified_coeffs.append(detail)
            detail_energy_after_desc.append(float(np.sum(detail * detail)))
            thresholds_desc.append(float(threshold))

        reconstructed_local = pywt.waverec(modified_coeffs, wavelet, mode="symmetric")
        detail_energy_before = np.asarray(detail_energy_before_desc[::-1], dtype=np.float64)
        detail_energy_after = np.asarray(detail_energy_after_desc[::-1], dtype=np.float64)
        thresholds = np.asarray(thresholds_desc[::-1], dtype=np.float64)
    elif transform_kind == "swt":
        # SWT requires a length divisible by 2**level.  Symmetric padding keeps
        # the selected-window edges from becoming artificial discontinuities.
        divisor = 2 ** level
        padded_length = int(math.ceil(local_segment.size / float(divisor)) * divisor)
        padded_segment = np.pad(local_segment, (0, padded_length - local_segment.size), mode="symmetric")
        coeffs = pywt.swt(padded_segment, wavelet, level=level, trim_approx=False)
        modified_coeffs = []
        detail_energy_before_desc = []
        detail_energy_after_desc = []
        thresholds_desc = []
        band_ranges_hz = _dwt_band_ranges(level, float(sample_rate_hz))
        band_labels = [
            f"D{i} [{_format_band_range(*band_ranges_hz[i - 1])}]"
            for i in range(1, level + 1)
        ]
        gains_array = np.ones(level, dtype=np.float64)
        if detail_gains is not None:
            requested_gains = np.asarray(list(detail_gains), dtype=np.float64)
            if requested_gains.size > 0:
                usable = min(level, requested_gains.size)
                gains_array[:usable] = requested_gains[:usable]

        for coeff_index, (approximation, detail_values) in enumerate(coeffs):
            detail_level = level - coeff_index
            detail = np.array(detail_values, copy=True, dtype=np.float64)
            guard = 2 ** detail_level
            coeff_left = max(0, local_left - guard)
            coeff_right = min(detail.size, local_right + 1 + guard)
            detail_energy_before_desc.append(float(np.sum(detail * detail)))
            if mode == "manual":
                gain = float(gains_array[detail_level - 1])
                updated = detail * gain
                threshold = 0.0
            else:
                background = np.concatenate((detail[:coeff_left], detail[coeff_right:]))
                if background.size == 0:
                    background = detail
                threshold = _threshold_for_band(
                    background,
                    threshold_scale=threshold_scale,
                    threshold_method=threshold_method,
                )
                if mode == "zero":
                    updated = np.zeros_like(detail)
                else:
                    updated = pywt.threshold(detail, value=float(threshold), mode=mode)
            changed_coefficients += int(np.count_nonzero(np.abs(updated - detail) > 1e-12))
            modified_coeffs.append((np.asarray(approximation, dtype=np.float64), np.asarray(updated, dtype=np.float64)))
            detail_energy_after_desc.append(float(np.sum(np.asarray(updated, dtype=np.float64) ** 2)))
            thresholds_desc.append(float(threshold))

        reconstructed_local = pywt.iswt(modified_coeffs, wavelet)[:local_segment.size]
        detail_energy_before = np.asarray(detail_energy_before_desc[::-1], dtype=np.float64)
        detail_energy_after = np.asarray(detail_energy_after_desc[::-1], dtype=np.float64)
        thresholds = np.asarray(thresholds_desc[::-1], dtype=np.float64)
    else:
        wp = pywt.WaveletPacket(
            data=local_segment, wavelet=wavelet, mode="symmetric", maxlevel=level,
        )
        nodes = wp.get_level(level, order="freq")
        band_ranges_hz = _wp_band_ranges(level, float(sample_rate_hz))
        band_labels = [
            f"{(node.path or 'a')} [{_format_band_range(*band_ranges_hz[i])}]"
            for i, node in enumerate(nodes)
        ]
        gains_array = np.ones(len(nodes), dtype=np.float64)
        if detail_gains is not None:
            requested_gains = np.asarray(list(detail_gains), dtype=np.float64)
            if requested_gains.size > 0:
                usable = min(len(nodes), requested_gains.size)
                gains_array[:usable] = requested_gains[:usable]

        detail_energy_before_list: list[float] = []
        detail_energy_after_list: list[float] = []
        thresholds_list: list[float] = []
        for band_index, node in enumerate(nodes):
            data_band = np.array(node.data, copy=True, dtype=np.float64)
            detail_energy_before_list.append(float(np.sum(data_band * data_band)))
            if mode == "manual":
                updated = data_band * float(gains_array[band_index])
                threshold = 0.0
            else:
                threshold = _threshold_for_band(
                    data_band,
                    threshold_scale=threshold_scale,
                    threshold_method=threshold_method,
                )
                if mode == "zero":
                    updated = np.zeros_like(data_band)
                else:
                    updated = pywt.threshold(data_band, value=float(threshold), mode=mode)
            changed_coefficients += int(np.count_nonzero(np.abs(updated - data_band) > 1e-12))
            wp[node.path] = np.asarray(updated, dtype=np.float64)
            energy_val = float(np.sum(
                np.asarray(updated, dtype=np.float64) ** 2,
            ))
            detail_energy_after_list.append(energy_val)
            thresholds_list.append(float(threshold))

        reconstructed_local = wp.reconstruct(update=True)
        detail_energy_before = np.asarray(detail_energy_before_list, dtype=np.float64)
        detail_energy_after = np.asarray(detail_energy_after_list, dtype=np.float64)
        thresholds = np.asarray(thresholds_list, dtype=np.float64)

    reconstructed_local = np.asarray(reconstructed_local[:local_segment.size], dtype=np.float64)

    blend_width = min(
        max(4, selected_width),
        local_left,
        max(0, local_segment.size - local_right - 1),
        half_window,
    )
    weights = np.ones(local_segment.size, dtype=np.float64)
    if blend_width > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, blend_width, dtype=np.float64))
        weights[:blend_width] = ramp
        weights[-blend_width:] = ramp[::-1]

    corrected_local = local_segment + weights * (reconstructed_local - local_segment)
    correction_local = local_segment - corrected_local
    # Numerical processing uses interpolation only internally.  Missing input
    # samples are data-quality information and must never silently turn into
    # denoised observations in a batch/plugin invocation.
    corrected_local[~finite_mask] = np.nan
    correction_local[~finite_mask] = 0.0

    corrected = np.array(data, copy=True)
    corrected[local_start:local_stop] = corrected_local
    correction = np.zeros_like(corrected)
    correction[local_start:local_stop] = correction_local

    return WaveletDenoiseResult(
        corrected=corrected,
        correction=correction,
        local_window=(local_start, local_stop - 1),
        region=(left, right),
        local_original=original_local_segment,
        local_reconstructed=corrected_local,
        detail_energy_before=detail_energy_before,
        detail_energy_after=detail_energy_after,
        band_labels=band_labels,
        band_ranges_hz=band_ranges_hz,
        wavelet_name=wavelet_name,
        transform_kind=transform_kind,
        level=level,
        threshold_scale=float(threshold_scale),
        threshold_method=threshold_method,
        level_policy=level_policy,
        protect_period_seconds=(
            float(protect_period_seconds) if protect_period_seconds is not None else None
        ),
        padding_points=padding_points,
        thresholds=thresholds,
        detail_gains=np.asarray(gains_array, dtype=np.float64),
        mode=mode,
        changed_coefficients=changed_coefficients,
        success=True,
        message="wavelet multiscale denoise finished",
    )


WaveletPulseResult = WaveletDenoiseResult


def wavelet_local_depulse(
    samples: np.ndarray,
    region: tuple[int, int],
    *,
    wavelet_name: str = "db4",
    level: int = 6,
    sample_rate_hz: float | None = None,
    threshold_scale: float = 1.0,
    padding_points: int = 256,
    mode: str = "soft",
) -> WaveletDenoiseResult:
    return wavelet_multiscale_denoise(
        samples,
        region,
        transform_kind="dwt",
        wavelet_name=wavelet_name,
        level=level,
        sample_rate_hz=sample_rate_hz,
        threshold_scale=threshold_scale,
        padding_points=padding_points,
        mode=mode,
    )
