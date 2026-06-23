"""显示降采样工具。

供 RegionPlotWidget / SpikePlotWidget 共用，将千万级数据降采样到
屏幕可分辨点数后传给 pyqtgraph，大幅降低渲染开销。

部分函数有 Rust 加速实现，当 ``rust_lib`` 可用时自动调用。
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

_DISPLAY_MAX_POINTS = 100_000

try:
    import rust_lib
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def decimate_for_display(
    x: np.ndarray, y: np.ndarray, max_points: int = _DISPLAY_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """等步长降采样（极简，用于 pyqtgraph 渲染）。

    对数据做等步长抽取，配合 pyqtgraph 内置的 ``autoDownsample='peak'``
    使用，既保证渲染速度，又通过 pyqtgraph 的峰值保留避免丢失尖峰。
    """
    n = len(x)
    if n <= max_points:
        return x, y
    stride = max(1, n // max_points)
    return x[::stride].copy(), y[::stride].copy()


def minmax_decimate(
    x: np.ndarray, y: np.ndarray, target_bins: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Min/max 包络线降采样。

    将数据分成 target_bins 个桶，每桶取 min + max 各一点交错排列。
    保证任何尖峰都不会在视觉上丢失，输出点数 ≤ 2 × target_bins。

    当 ``rust_lib`` 可用时自动调用 Rust 并行实现。
    """
    if _HAS_RUST:
        try:
            return rust_lib.minmax_decimate_rs(x, y, target_bins)
        except Exception:
            logger.debug("Rust minmax_decimate failed, falling back to Python",
                         exc_info=True)

    n = len(x)
    if n <= target_bins * 2:
        return x, y

    bins = max(1, min(target_bins, n // 2))
    x_chunks = np.array_split(x, bins)
    y_chunks = np.array_split(y, bins)

    rx_parts, ry_parts = [], []
    for xc, yc in zip(x_chunks, y_chunks):
        if len(xc) <= 1:
            rx_parts.append(xc)
            ry_parts.append(yc)
            continue
        imin = int(np.argmin(yc))
        imax = int(np.argmax(yc))
        if imin < imax:
            rx_parts.append(np.array([xc[imin], xc[imax]]))
            ry_parts.append(np.array([yc[imin], yc[imax]]))
        elif imax < imin:
            rx_parts.append(np.array([xc[imax], xc[imin]]))
            ry_parts.append(np.array([yc[imax], yc[imin]]))
        else:
            rx_parts.append(xc[imin:imin + 1])
            ry_parts.append(yc[imin:imin + 1])

    return np.concatenate(rx_parts), np.concatenate(ry_parts)


def build_lod_pyramid(
    x: np.ndarray, y: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """构建 LOD 金字塔。

    返回从粗到精的多层降采样数据。当 ``rust_lib`` 可用时调用 Rust 实现。
    """
    if _HAS_RUST:
        try:
            return list(rust_lib.build_lod_pyramid_rs(x, y))
        except Exception:
            logger.debug("Rust build_lod_pyramid failed, falling back to Python",
                         exc_info=True)

    levels: list[tuple[np.ndarray, np.ndarray]] = []
    cur_x, cur_y = x, y
    target = max(256, len(x) // 5)

    while target >= 256 and len(cur_x) > target * 2:
        dx, dy = minmax_decimate(cur_x, cur_y, target)
        levels.append((dx, dy))
        cur_x, cur_y = dx, dy
        target = max(256, len(cur_x) // 5)

    return levels


def apply_nan_intervals(
    y: np.ndarray, intervals: list[tuple[int, int]],
) -> None:
    """将指定区间内的值原地置为 NaN。

    当 ``rust_lib`` 可用时调用 Rust 实现，消除中间索引数组分配。
    """
    if _HAS_RUST:
        try:
            rust_lib.apply_nan_intervals_rs(y, intervals)
            return
        except Exception:
            logger.debug("Rust apply_nan_intervals failed, falling back to Python",
                         exc_info=True)

    for start, end in intervals:
        y[start:end] = np.nan
