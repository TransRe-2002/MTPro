"""时序数据降采样工具。

提供像素级 min/max 包络线降采样和 LOD 金字塔预计算，
用于百万级数据点的波形流畅浏览。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import rust_lib
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def minmax_decimate(
    x: np.ndarray, y: np.ndarray, target_bins: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Min/max 包络线降采样。

    将数据分成 target_bins 个桶，每桶取 min + max 各一点交错排列。
    保证任何尖峰都不会在视觉上丢失，输出点数 ≤ 2 × target_bins。

    当 ``rust_lib`` 可用时自动调用 Rust 并行实现。

    Args:
        x: X 轴数据。
        y: Y 轴数据。
        target_bins: 目标桶数，默认 1000。

    Returns:
        (x_downsampled, y_downsampled) 降采样后的数据。
    """
    if _HAS_RUST:
        try:
            return rust_lib.minmax_decimate_rs(x, y, target_bins)
        except Exception:
            logger.debug(
                "Rust minmax_decimate failed, falling back to Python",
                exc_info=True,
            )

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

    Args:
        x: X 轴数据。
        y: Y 轴数据。

    Returns:
        LOD 层级列表，每层为 (x, y) 元组，从粗到精排列。
    """
    if _HAS_RUST:
        try:
            return list(rust_lib.build_lod_pyramid_rs(x, y))
        except Exception:
            logger.debug(
                "Rust build_lod_pyramid failed, falling back to Python",
                exc_info=True,
            )

    levels: list[tuple[np.ndarray, np.ndarray]] = []
    cur_x, cur_y = x, y
    target = max(256, len(x) // 5)

    while target >= 256 and len(cur_x) > target * 2:
        dx, dy = minmax_decimate(cur_x, cur_y, target)
        levels.append((dx, dy))
        cur_x, cur_y = dx, dy
        target = max(256, len(cur_x) // 5)

    return levels


def lod_levels(
    max_points: int, min_points: int = 256,
) -> list[int]:
    """根据原始数据点数生成 LOD 金字塔各级的目标点数。

    每一级约为上一级的 1/5，直到低于 min_points。

    Args:
        max_points: 原始数据点数。
        min_points: 最小目标点数，默认 256。

    Returns:
        各级目标点数列表 [L1_target, L2_target, ...]。
    """
    levels: list[int] = []
    current = max_points
    while current > min_points:
        current = max(min_points, current // 5)
        levels.append(current)
    return levels


class LODPyramid:
    """预计算的 LOD 金字塔。

    在数据加载时预先计算多个下采样层级，查询时根据可见范围
    和目标像素宽度返回最合适的一层。

    用法::

        pyramid = LODPyramid(x_data, y_data)
        x_lod, y_lod = pyramid.get(x_range_start, x_range_end, pixel_width)
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x_raw = np.asarray(x, dtype=np.float64)
        self.y_raw = np.asarray(y, dtype=np.float64)
        self._levels: list[tuple[np.ndarray, np.ndarray]] = []
        self._build()

    def _build(self) -> None:
        self._levels = build_lod_pyramid(self.x_raw, self.y_raw)

    @property
    def level_count(self) -> int:
        return len(self._levels)

    def get(
        self, x_start: float, x_end: float, pixel_width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """获取适合当前视口的数据。

        Args:
            x_start: 视口 X 轴起始（数据坐标）。
            x_end: 视口 X 轴结束（数据坐标）。
            pixel_width: 视口水平像素数。

        Returns:
            (x_slice, y_slice)，可能来自某个 LOD 层或原始数据切片。
        """
        if pixel_width <= 0:
            pixel_width = 1

        mask = (self.x_raw >= x_start) & (self.x_raw <= x_end)
        visible_count = int(np.sum(mask))

        if visible_count <= pixel_width * 2:
            return self.x_raw[mask], self.y_raw[mask]

        for dx, dy in self._levels:
            if len(dx) <= pixel_width * 2:
                submask = (dx >= x_start) & (dx <= x_end)
                return dx[submask], dy[submask]

        dx, dy = self._levels[-1]
        submask = (dx >= x_start) & (dx <= x_end)
        return dx[submask], dy[submask]

    def get_full_overview(
        self, pixel_width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """获取全量数据的概览（用于 overview 窗口）。

        Args:
            pixel_width: overview 窗口的像素宽度。

        Returns:
            适合该像素宽度的降采样后全量数据。
        """
        target = max(256, pixel_width * 2)
        for dx, dy in self._levels:
            if len(dx) <= target:
                return dx, dy
        return self._levels[-1]


def slice_visible(
    x: np.ndarray, y: np.ndarray,
    x_start: float, x_end: float,
    margin_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """从全量数据中切片出可见范围（含边距）。

    边距用于支持小幅平移时的 pixmap 缓存。

    Args:
        x: X 轴数据。
        y: Y 轴数据。
        x_start: 视口起始。
        x_end: 视口结束。
        margin_ratio: 边距比例，默认 0.1 (10%)。

    Returns:
        (x_slice, y_slice)。
    """
    margin = (x_end - x_start) * margin_ratio
    lo = x_start - margin
    hi = x_end + margin
    mask = (x >= lo) & (x <= hi)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return x[:0], y[:0]
    return x[indices[0]:indices[-1] + 1], y[indices[0]:indices[-1] + 1]
