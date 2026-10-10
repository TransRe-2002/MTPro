from __future__ import annotations

"""与 Qt 无关的 Pipeline 时间轴布局和联动组标签计算。"""

import math


def linked_group_label(index: int) -> str:
    """Return compact spreadsheet-style labels: A..Z, AA..AZ, ..."""
    index = max(0, int(index))
    chars: list[str] = []
    while True:
        index, remainder = divmod(index, 26)
        chars.append(chr(ord("A") + remainder))
        if index == 0:
            return "".join(reversed(chars))
        index -= 1


def linked_group_label_index(label: str) -> int:
    """Return the zero-based color slot source for a group display label."""
    value = 0
    for char in label.upper():
        if not "A" <= char <= "Z":
            continue
        value = value * 26 + ord(char) - ord("A") + 1
    return max(0, value - 1)


def px_per_point(zoom: float) -> float:
    return max(0.02, zoom * 8 / 1000)


def points_to_pixels(points: int, zoom: float) -> float:
    return points * px_per_point(zoom)


def fit_timeline_px_per_point(zoom: float, data_len: int, viewport_width: int) -> float:
    """Return a timeline scale where zoom 1 fits the full channel width."""
    return max(1e-9, float(viewport_width) * max(1.0, zoom) / max(1, data_len))


def _nice_tick_multiple(minimum: float) -> int:
    """Return a 1/2/5-based integer multiple no smaller than ``minimum``."""
    if minimum <= 1.0:
        return 1
    magnitude = 10 ** math.floor(math.log10(minimum))
    for factor in (1, 2, 5, 10):
        candidate = int(factor * magnitude)
        if candidate >= minimum:
            return candidate
    return int(10 * magnitude)


def ruler_tick_multipliers(
    grid_points: int,
    zoom: float,
    *,
    min_tick_pixels: float = 8.0,
    min_label_pixels: float = 48.0,
    pixels_per_point: float | None = None,
) -> tuple[int, int]:
    """Choose minor and labeled-tick strides that stay legible at ``zoom``."""
    scale = px_per_point(zoom) if pixels_per_point is None else pixels_per_point
    grid_pixels = max(max(1, grid_points) * scale, 1e-6)
    tick_multiple = _nice_tick_multiple(min_tick_pixels / grid_pixels)
    label_multiple = _nice_tick_multiple(min_label_pixels / grid_pixels)
    return tick_multiple, max(tick_multiple, label_multiple)
