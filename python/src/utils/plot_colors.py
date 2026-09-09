"""Shared distinct-colour allocation for plots."""

from __future__ import annotations

import colorsys
import random

CURVE_COLOR_CANDIDATES = 48
CURVE_COLOR_LIGHTNESS = 0.42
CURVE_COLOR_SATURATION = 0.78


class DistinctColorAllocator:
    """Allocate and retain visually separated RGB colours for stable keys."""

    def __init__(self, seed: int | None = None) -> None:
        self.colors: dict[str, tuple[int, int, int]] = {}
        self.hues: dict[str, float] = {}
        self.rng = random.Random(seed)

    def color_for(self, key: str) -> tuple[int, int, int]:
        key = str(key)
        if key in self.colors:
            return self.colors[key]

        active_hues = list(self.hues.values())
        candidates = [self.rng.random() for _ in range(CURVE_COLOR_CANDIDATES)]
        if active_hues:
            hue = max(
                candidates,
                key=lambda candidate: min(
                    min(
                        abs(candidate - existing),
                        1.0 - abs(candidate - existing),
                    )
                    for existing in active_hues
                ),
            )
        else:
            hue = candidates[0]

        red, green, blue = colorsys.hls_to_rgb(
            hue,
            CURVE_COLOR_LIGHTNESS,
            CURVE_COLOR_SATURATION,
        )
        color = (int(red * 255), int(green * 255), int(blue * 255))
        self.hues[key] = hue
        self.colors[key] = color
        return color

    def clear(self) -> None:
        self.colors.clear()
        self.hues.clear()
