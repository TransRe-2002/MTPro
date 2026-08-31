"""应用级数据量阈值与相关工具。

``MAX_POINTS_PER_CHANNEL`` 原为 ``io_utils/xlsx_io.py`` 的导入把关阈值，
降采样对话框（``ui/dialogs/decimate_dialog.py``）也需要同一阈值做预览提示，
故提升为共享模块；``xlsx_io`` 从此处 re-export 保持兼容。
"""

from __future__ import annotations

import math

# 单通道点数建议上限：超过后波形浏览（即便有 LOD 降采样）与
# 处理工具（全序列拷贝、逐窗拟合）的内存/时间开销都会明显上升。
MAX_POINTS_PER_CHANNEL = 5_000_000


def suggested_decimate_factor(
    npts: int,
    max_points: int = MAX_POINTS_PER_CHANNEL,
) -> int | None:
    """把 ``npts`` 压到 ``max_points`` 以内的最小整数抽取因子。

    与 ``utils.signal_decimate.decimated_length`` 的输出长度约定反解：
    ``(n-1)//f + 1 <= m`` 等价于 ``n <= f*m``，故 ``f >= n/m``；
    ``npts > max_points`` 时该值必然 ≥ 2。``npts`` 已在上限以内时
    返回 ``None``。
    """
    if npts <= max_points:
        return None
    if max_points < 2:
        max_points = 2
    return max(2, math.ceil(npts / max_points))


def format_points(n: int) -> str:
    """点数的中文紧凑显示：1576.8万 / 1.58亿 / 9,999。"""
    if n >= 100_000_000:
        return f"{n / 1e8:.2f}亿"
    if n >= 10_000:
        return f"{n / 1e4:.1f}万"
    return f"{n:,}"
