"""NaN 感知的整数倍信号降采样。

实现 ``matlab/TS.m::decimate`` 默认 ``boxcar`` 策略的 Python 版本：
每个输出样点对应一个输入窗口，对窗口内有限样点求平均；仅当缺失
比例超过阈值时输出 NaN。窗口平均同时构成基础的 boxcar FIR 低通，
避免直接抽点造成最明显的混叠。

与旧实现不同，这里不先跨越任意长度的 NaN 缺口做线性插值，也不再
因为窗口内仅有一个 NaN 就废弃整个输出点。
"""

from __future__ import annotations

import logging

import numpy as np


# ``TS.m::decimate`` 的 boxcar 分支直接比较 ``nGood < .25``，其中
# nGood 是窗口内有效样点的整数个数。因此 MATLAB 的实际默认行为是：
# 只要窗口里至少有一个有效样点就保留。用缺失比例表达时等价于 1.0；
# 全缺失窗口仍由 good_count == 0 单独输出 NaN。
DEFAULT_MAX_MISSING_FRACTION = 1.0
logger = logging.getLogger(__name__)

try:
    import rust_lib
    _HAS_RUST_BOXCAR = hasattr(rust_lib, "nan_boxcar_decimate_rs")
except ImportError:
    rust_lib = None  # type: ignore
    _HAS_RUST_BOXCAR = False


def decimated_length(npts: int, factor: int) -> int:
    """返回含末尾不足整窗部分时的输出长度。"""
    if npts <= 0:
        return 0
    return (npts - 1) // factor + 1


def decimate_nan_aware(
    y: np.ndarray,
    factor: int,
    new_npts: int,
    max_missing_fraction: float = DEFAULT_MAX_MISSING_FRACTION,
) -> np.ndarray:
    """按 boxcar 窗口对一维序列做 NaN 感知降采样。

    每 ``factor`` 个输入样点对应一个输出样点。有限值参与窗口平均，
    NaN/正负无穷视为缺失；窗口缺失比例不超过
    ``max_missing_fraction`` 时保留平均值，否则输出 NaN。末尾不足一个
    完整窗口时按实际窗口长度计算比例。

    Args:
        y: 一维数值序列。
        factor: 整数降采样因子，必须 >= 2。
        new_npts: 期望输出长度，必须等于
            ``decimated_length(len(y), factor)``。
        max_missing_fraction: 每窗口最大允许缺失比例，范围 [0, 1]；
            默认 1.0，与 ``TS.m`` boxcar 分支的实际行为一致：窗口只要
            至少含一个有效样点就求平均。全缺失窗口始终输出 NaN。

    Returns:
        长度为 ``new_npts`` 的 float64 数组。
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {y.shape}")
    if factor < 2:
        raise ValueError(f"factor must be >= 2, got {factor}")
    expected_npts = decimated_length(len(y), factor)
    if new_npts != expected_npts:
        raise ValueError(
            f"new_npts must be {expected_npts} for {len(y)} samples "
            f"and factor {factor}, got {new_npts}"
        )
    if not 0.0 <= max_missing_fraction <= 1.0:
        raise ValueError(
            "max_missing_fraction must be between 0 and 1, "
            f"got {max_missing_fraction}"
        )

    if _HAS_RUST_BOXCAR:
        try:
            contiguous = np.ascontiguousarray(y, dtype=np.float64)
            result = rust_lib.nan_boxcar_decimate_rs(
                contiguous,
                factor,
                max_missing_fraction,
            )
            return np.asarray(result, dtype=np.float64)
        except Exception:
            logger.debug(
                "Rust nan_boxcar_decimate failed, falling back to NumPy",
                exc_info=True,
            )

    return _decimate_nan_aware_numpy(y, factor, new_npts, max_missing_fraction)


def _decimate_nan_aware_numpy(
    y: np.ndarray,
    factor: int,
    new_npts: int,
    max_missing_fraction: float,
) -> np.ndarray:
    """NumPy fallback; inputs have already been validated by the public API."""

    result = np.full(new_npts, np.nan, dtype=np.float64)
    full_windows = len(y) // factor
    if full_windows:
        windows = y[:full_windows * factor].reshape(full_windows, factor)
        finite = np.isfinite(windows)
        good_counts = finite.sum(axis=1)
        accepted = (
            (good_counts > 0)
            & ((factor - good_counts) / factor <= max_missing_fraction)
        )
        sums = np.sum(windows, axis=1, where=finite, initial=0.0)
        accepted_indices = np.flatnonzero(accepted)
        result[accepted_indices] = (
            sums[accepted_indices] / good_counts[accepted_indices]
        )

    if full_windows < new_npts:
        window = y[full_windows * factor:]
        finite = np.isfinite(window)
        good_count = int(finite.sum())
        missing_fraction = 1.0 - good_count / len(window)
        if good_count > 0 and missing_fraction <= max_missing_fraction:
            result[full_windows] = float(window[finite].mean())
    return result
