"""信号处理级整数倍抽取（降采样）。

与 ``utils/decimate.py``（显示用 min/max 包络 LOD 金字塔）不同，
本模块做真正的抗混叠重采样，用于数据管理器内的就地降采样预处理：
先低通滤波再按整数因子抽点，结果会写回数据集并随保存持久化。

NaN 间隙保持
------------
MT 时序用 NaN 标记数据缺口，而 ``scipy.signal.decimate`` 的零相位
IIR 滤波（filtfilt）会把 NaN 涂抹到整条序列。处理流程：

1. 记录 NaN 掩码；
2. 按 q 窗口降采样掩码——窗口内**任一**采样为 NaN，输出即为 NaN
   （保守策略：滤波窗口内的缺口会污染该输出点）；
3. 线性插值填补原序列的 NaN，得到干净的滤波输入；
4. 对插值序列做 ``scipy.signal.decimate``；
5. 把降采样后的 NaN 掩码重新贴回输出。
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import decimate as _scipy_decimate

logger = logging.getLogger(__name__)

# 过短序列无法通过 8 阶 IIR 零相位滤波的边界填充（filtfilt 要求
# 长度 > 3×滤波器阶数相关填充），低于该阈值退化为直接抽点。
_MIN_POINTS_FOR_FILTER = 32


def decimated_length(npts: int, factor: int) -> int:
    """整数倍抽取后的输出长度（与 ``scipy.signal.decimate`` 一致）。"""
    if npts <= 0:
        return 0
    return (npts - 1) // factor + 1


def decimate_nan_aware(y: np.ndarray, factor: int, new_npts: int) -> np.ndarray:
    """按整数因子对一维序列做抗混叠抽取，保持 NaN 缺口。

    Args:
        y: 一维浮点序列（允许含 NaN）。
        factor: 抽取因子，整数 ≥ 2。
        new_npts: 期望输出长度，应等于 ``decimated_length(len(y), factor)``。

    Returns:
        长度为 ``new_npts`` 的 float64 数组；缺口窗口处为 NaN。
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if factor < 2:
        raise ValueError(f"factor must be >= 2, got {factor}")

    nan_mask = np.isnan(y)

    # ── NaN 掩码按 q 窗口降采样：窗口内任一 NaN → 输出 NaN ──
    down_mask = np.zeros(new_npts, dtype=bool)
    full_windows = n // factor
    if full_windows > 0:
        down_mask[:full_windows] = (
            nan_mask[: full_windows * factor]
            .reshape(full_windows, factor)
            .any(axis=1)
        )
    remainder = n - full_windows * factor
    if remainder > 0 and full_windows < new_npts:
        down_mask[full_windows] = nan_mask[full_windows * factor :].any()

    # ── 全 NaN：直接返回正确长度的 NaN ──
    if nan_mask.all():
        return np.full(new_npts, np.nan, dtype=np.float64)

    # ── 无 NaN：直接抽取 ──
    if not nan_mask.any():
        return _decimate_clean(y, factor)[:new_npts].copy()

    # ── 有 NaN：线性插值填补后抽取，再贴回降采样掩码 ──
    y_interp = y.copy()
    indices = np.arange(n)
    valid = ~nan_mask
    y_interp[nan_mask] = np.interp(
        indices[nan_mask], indices[valid], y[valid],
    )
    result = _decimate_clean(y_interp, factor)[:new_npts].copy()
    result[down_mask] = np.nan
    return result


def _decimate_clean(y: np.ndarray, factor: int) -> np.ndarray:
    """对不含 NaN 的序列做零相位抗混叠抽取。

    序列过短（低于 :data:`_MIN_POINTS_FOR_FILTER`）时无法通过
    filtfilt 的边界填充，退化为直接抽点（无滤波）并记录警告。
    """
    if len(y) < _MIN_POINTS_FOR_FILTER:
        logger.warning(
            "序列过短（%d 点），无法抗混叠滤波，退化为直接抽点", len(y),
        )
        return y[::factor]
    return _scipy_decimate(y, factor)
