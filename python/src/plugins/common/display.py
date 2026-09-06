"""显示降采样工具。

供 RegionPlotWidget / SpikePlotWidget 共用，将千万级数据降采样到
屏幕可分辨点数后传给 pyqtgraph，大幅降低渲染开销。

部分函数有 Rust 加速实现，当 ``rust_lib`` 可用时自动调用。
"""

from __future__ import annotations

import logging

import numpy as np

from utils.decimate import build_lod_pyramid, minmax_decimate

logger = logging.getLogger(__name__)

_DISPLAY_MAX_POINTS = 100_000

try:
    import rust_lib
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

# minmax_decimate / build_lod_pyramid 已移至 utils.decimate，
# 此处保留重导出以兼容旧引用路径。
__all__ = [
    "apply_nan_intervals",
    "build_lod_pyramid",
    "decimate_for_display",
    "minmax_decimate",
]


def decimate_for_display(
    x: np.ndarray, y: np.ndarray, max_points: int = _DISPLAY_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """等步长降采样（极简，用于 pyqtgraph 渲染）。

    对数据做等步长抽取，配合 pyqtgraph 内置的 ``autoDownsample='peak'``
    使用，既保证渲染速度，又通过 pyqtgraph 的峰值保留避免丢失尖峰。

    Args:
        x: X 轴数据。
        y: Y 轴数据。
        max_points: 最大保留点数，默认 100000。

    Returns:
        (x_downsampled, y_downsampled)。
    """
    n = len(x)
    if n <= max_points:
        return x, y
    stride = max(1, n // max_points)
    return x[::stride].copy(), y[::stride].copy()


def apply_nan_intervals(
    y: np.ndarray, intervals: list[tuple[int, int]],
) -> None:
    """将指定区间内的值原地置为 NaN。

    当 ``rust_lib`` 可用时调用 Rust 实现，消除中间索引数组分配。

    Args:
        y: 要修改的数组（原地操作）。
        intervals: (start, end) 索引区间列表。
    """
    if _HAS_RUST:
        try:
            rust_lib.apply_nan_intervals_rs(y, intervals)
            return
        except Exception:
            logger.debug(
                "Rust apply_nan_intervals failed, falling back to Python",
                exc_info=True,
            )

    for start, end in intervals:
        y[start:end] = np.nan
