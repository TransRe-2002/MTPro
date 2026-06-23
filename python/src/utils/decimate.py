"""时序数据降采样工具。

提供像素级 min/max 包络线降采样和 LOD 金字塔预计算，
用于百万级数据点的波形流畅浏览。
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from processor._display_utils import minmax_decimate as _minmax_decimate_rs
from processor._display_utils import build_lod_pyramid as _build_lod_pyramid_rs


def minmax_decimate(x: np.ndarray, y: np.ndarray,
                    target_bins: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """委托到 _display_utils（自动选择 Rust 或 numpy 实现）。"""
    return _minmax_decimate_rs(x, y, target_bins)


def lod_levels(max_points: int,
               min_points: int = 256) -> list[int]:
    """根据原始数据点数生成 LOD 金字塔各级的目标点数。

    每一级约为上一级的 1/5，直到低于 min_points。
    返回 [L1_target, L2_target, L3_target, ...]。
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

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x_raw = np.asarray(x, dtype=np.float64)
        self.y_raw = np.asarray(y, dtype=np.float64)
        self._levels: list[tuple[np.ndarray, np.ndarray]] = []
        self._build()

    def _build(self) -> None:
        self._levels = _build_lod_pyramid_rs(self.x_raw, self.y_raw)

    @property
    def level_count(self) -> int:
        return len(self._levels)

    def get(self, x_start: float, x_end: float,
            pixel_width: int) -> tuple[np.ndarray, np.ndarray]:
        """获取适合当前视口的数据。

        x_start, x_end: 视口 X 轴范围（数据坐标）
        pixel_width: 视口水平像素数

        返回 (x_slice, y_slice)，可能来自某个 LOD 层或原始数据切片。
        """
        if pixel_width <= 0:
            pixel_width = 1

        # 找到视口内的数据范围
        mask = (self.x_raw >= x_start) & (self.x_raw <= x_end)
        visible_count = int(np.sum(mask))

        if visible_count <= pixel_width * 2:
            # 可见点不多，直接用原始数据
            return self.x_raw[mask], self.y_raw[mask]

        # 遍历 LOD 层级，找到第一个 target <= pixel_width * 2 的
        for (dx, dy) in self._levels:
            if len(dx) <= pixel_width * 2:
                submask = (dx >= x_start) & (dx <= x_end)
                return dx[submask], dy[submask]

        # 最粗一层仍不够，使用最后一层
        dx, dy = self._levels[-1]
        submask = (dx >= x_start) & (dx <= x_end)
        return dx[submask], dy[submask]

    def get_full_overview(self, pixel_width: int) -> tuple[np.ndarray, np.ndarray]:
        """获取全量数据的概览（用于 overview 窗口）。

        pixel_width: overview 窗口的像素宽度

        返回适合该像素宽度的降采样后全量数据。
        """
        target = max(256, pixel_width * 2)
        for (dx, dy) in self._levels:
            if len(dx) <= target:
                return dx, dy
        return self._levels[-1]


def slice_visible(x: np.ndarray, y: np.ndarray,
                  x_start: float, x_end: float,
                  margin_ratio: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """从全量数据中切片出可见范围（含边距）。

    边距用于支持小幅平移时的 pixmap 缓存。
    """
    margin = (x_end - x_start) * margin_ratio
    lo = x_start - margin
    hi = x_end + margin
    mask = (x >= lo) & (x <= hi)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return x[:0], y[:0]
    return x[indices[0]:indices[-1] + 1], y[indices[0]:indices[-1] + 1]
