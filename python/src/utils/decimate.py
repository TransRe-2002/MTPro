"""时序数据降采样工具。

提供像素级 min/max 包络线降采样和 LOD 金字塔预计算，
用于百万级数据点的波形流畅浏览。
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import rust_lib
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

# 金字塔截顶上限：点数超过该值的顶层不存储（链式推导照常进行，
# 更粗层的精度不受影响）。只对超大数据集生效（如 5s 采样年度
# 文件，顶层可达 250 万点）；60s 采样数据的顶层远小于该值，
# 截顶不会触发。压低的是金字塔内存，不影响构建耗时。
PYRAMID_MAX_FINEST_POINTS = 1_500_000


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
            # MATLAB channel rows can be strided views of a Fortran-order
            # matrix. PyO3 as_slice requires contiguous, aligned f64 input;
            # normalize only the native-call buffers, not the retained data.
            return rust_lib.minmax_decimate_rs(
                np.require(x, dtype=np.float64, requirements=["C", "A"]),
                np.require(y, dtype=np.float64, requirements=["C", "A"]),
                target_bins,
            )
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
        try:
            imin = int(np.nanargmin(yc))
            imax = int(np.nanargmax(yc))
        except ValueError:
            # All-NaN bin: retain one NaN sample so gaps survive coarser LODs.
            imin = imax = 0
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
    max_finest_points: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """构建 LOD 金字塔。

    返回从粗到精的多层降采样数据。当 ``rust_lib`` 可用时调用 Rust 实现。

    Args:
        x: X 轴数据。
        y: Y 轴数据。
        max_finest_points: 截顶上限——点数超过该值的顶层不存储，
            但链式推导照常进行（更粗层精度不受影响）。None = 不截顶。

    Returns:
        LOD 层级列表，每层为 (x, y) 元组，从粗到精排列。
    """
    if _HAS_RUST:
        try:
            return list(rust_lib.build_lod_pyramid_rs(
                np.require(x, dtype=np.float64, requirements=["C", "A"]),
                np.require(y, dtype=np.float64, requirements=["C", "A"]),
                max_finest_points,
            ))
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
        if max_finest_points is None or len(dx) <= max_finest_points:
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

    def __init__(
        self, x: np.ndarray, y: np.ndarray,
        max_finest_points: int | None = None,
    ) -> None:
        self.x_raw = np.asarray(x, dtype=np.float64)
        self.y_raw = np.asarray(y, dtype=np.float64)
        self._max_finest_points = max_finest_points
        self._levels: list[tuple[np.ndarray, np.ndarray]] = []
        self._build()

    def _build(self) -> None:
        self._levels = build_lod_pyramid(
            self.x_raw, self.y_raw, self._max_finest_points,
        )

    @property
    def level_count(self) -> int:
        return len(self._levels)

    def get(
        self, x_start: float, x_end: float, pixel_width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """获取适合当前视口的数据。

        选择标准是**视口内可见点数**（而非某层的总点数）不超过
        ``pixel_width * 4`` 的最精细层级，因此放大到很小视口时仍能
        拿到足够密的数据，不会出现"选中的层在视口内没有点"的情况。

        要求 x 数据升序排列（时间序列天然满足），用二分查找定位
        视口边界，查询代价与数据总量无关（O(log n)）。

        Args:
            x_start: 视口 X 轴起始（数据坐标）。
            x_end: 视口 X 轴结束（数据坐标）。
            pixel_width: 视口水平像素数。

        Returns:
            (x_slice, y_slice)，可能来自某个 LOD 层或原始数据切片。
        """
        if pixel_width <= 0:
            pixel_width = 1
        if x_start > x_end:
            x_start, x_end = x_end, x_start
        target = pixel_width * 4

        lo = int(np.searchsorted(self.x_raw, x_start, side="left"))
        hi = int(np.searchsorted(self.x_raw, x_end, side="right"))
        if hi - lo <= target:
            return self.x_raw[lo:hi], self.y_raw[lo:hi]

        # levels 从精细到粗糙排列；取第一个"可见点数达标"的层即最精细可用层
        for dx, dy in self._levels:
            lo = int(np.searchsorted(dx, x_start, side="left"))
            hi = int(np.searchsorted(dx, x_end, side="right"))
            if hi - lo <= target:
                return dx[lo:hi], dy[lo:hi]

        dx, dy = self._levels[-1]
        lo = int(np.searchsorted(dx, x_start, side="left"))
        hi = int(np.searchsorted(dx, x_end, side="right"))
        return dx[lo:hi], dy[lo:hi]

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
        if not self._levels:
            return self.x_raw, self.y_raw
        return self._levels[-1]


def slice_visible(
    x: np.ndarray, y: np.ndarray,
    x_start: float, x_end: float,
    margin_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """从全量数据中切片出可见范围（含边距）。

    边距用于支持小幅平移时的 pixmap 缓存。要求 x 升序排列，
    用二分查找定位边界（O(log n)），与数据总量无关。

    Args:
        x: X 轴数据（升序）。
        y: Y 轴数据。
        x_start: 视口起始。
        x_end: 视口结束。
        margin_ratio: 边距比例，默认 0.1 (10%)。

    Returns:
        (x_slice, y_slice)。
    """
    margin = (x_end - x_start) * margin_ratio
    lo = int(np.searchsorted(x, x_start - margin, side="left"))
    hi = int(np.searchsorted(x, x_end + margin, side="right"))
    return x[lo:hi], y[lo:hi]
