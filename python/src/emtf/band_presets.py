"""Frequency-band presets for EMTF ``bs.cfg`` generation."""

from __future__ import annotations

import math
from collections.abc import Sequence

from emtf.types import DecimationLevel, FrequencyBand

DEFAULT_NUM_LEVELS = 6


def standard_bands(num_levels: int = DEFAULT_NUM_LEVELS) -> list[FrequencyBand]:
    """Return the default EMTF band table."""
    bands: list[FrequencyBand] = []
    for level in range(1, min(num_levels + 1, 6)):
        if level == 1:
            ranges = [(8, 9), (6, 7), (5, 5)]
        else:
            ranges = [(14, 17), (11, 13), (9, 10), (7, 8), (6, 6), (5, 5)]
        bands.extend(FrequencyBand(level, start, end) for start, end in ranges)
    return bands


def fine_bands(num_levels: int = DEFAULT_NUM_LEVELS) -> list[FrequencyBand]:
    """Return mostly one-Fourier-coefficient bands for denser output."""
    bands: list[FrequencyBand] = []
    for level in range(1, num_levels + 1):
        fc_values = range(5, 31) if level == 1 else range(5, 18)
        bands.extend(FrequencyBand(level, fc, fc) for fc in reversed(list(fc_values)))
    return bands


def half_decade_bands(num_levels: int = DEFAULT_NUM_LEVELS) -> list[FrequencyBand]:
    """Return a sparse half-decade-style band table."""
    bands = [
        FrequencyBand(1, 12, 14),
        FrequencyBand(2, 16, 18),
        FrequencyBand(3, 19, 22),
        FrequencyBand(4, 25, 28),
        FrequencyBand(5, 25, 30),
    ]
    return [band for band in bands if band.decimation_level <= num_levels]


def preset_bands(name: str, num_levels: int = DEFAULT_NUM_LEVELS) -> list[FrequencyBand]:
    """Return a named EMTF frequency-band preset."""
    key = name.strip().lower().replace("_", "-")
    if key == "standard":
        return standard_bands(num_levels)
    if key == "fine":
        return fine_bands(num_levels)
    if key in {"half-decade", "halfdecade"}:
        return half_decade_bands(num_levels)
    raise ValueError(f"unknown EMTF band preset: {name}")


def period_grid_bands(
    sample_rate_hz: float,
    levels: Sequence[DecimationLevel],
    *,
    min_period: float,
    max_period: float,
    n_edges: int,
    min_fc: int = 5,
) -> list[FrequencyBand]:
    """Map logarithmic period bins to EMTF ``bs.cfg`` Fourier-coefficient bands.

    EMTF's ``bs.cfg`` does not accept periods directly.  It accepts a
    decimation level and an inclusive FC-index range.  For a given level,
    EMTF reports approximately ``period = nwin / (fc * level_sample_rate)``.
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if min_period <= 0 or max_period <= 0:
        raise ValueError("period range must be positive")
    if max_period <= min_period:
        raise ValueError("max_period must be greater than min_period")
    if n_edges < 2:
        raise ValueError("n_edges must be at least 2")
    if not levels:
        raise ValueError("at least one decimation level is required")

    log_min = math.log(min_period)
    log_max = math.log(max_period)
    edges = [
        math.exp(log_min + (log_max - log_min) * i / (n_edges - 1))
        for i in range(n_edges)
    ]

    bands: list[FrequencyBand] = []
    seen: set[tuple[int, int, int]] = set()
    cumulative_decimation = 1.0

    level_candidates: list[tuple[int, DecimationLevel, float, range]] = []
    for level_index, level in enumerate(levels, start=1):
        cumulative_decimation *= max(1.0, float(level.decimation_factor))
        level_sample_rate = sample_rate_hz / cumulative_decimation
        max_fc = _max_fc_for_level(level_index, level, min_fc)
        if max_fc >= min_fc:
            level_candidates.append((level_index, level, level_sample_rate, range(min_fc, max_fc + 1)))

    for edge_index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        include_hi = edge_index == len(edges) - 2
        groups: list[tuple[float, FrequencyBand]] = []
        for level_index, level, level_sample_rate, fc_values in level_candidates:
            selected = [
                fc for fc in fc_values
                if _period_in_bin(
                    _fc_period(level, level_sample_rate, fc),
                    lo,
                    hi,
                    include_hi,
                )
            ]
            for start, end in _contiguous_ranges(selected):
                key = (level_index, start, end)
                if key in seen:
                    continue
                seen.add(key)
                center_period = _fc_period(level, level_sample_rate, (start + end) / 2.0)
                groups.append((center_period, FrequencyBand(level_index, start, end)))
        groups.sort(key=lambda item: item[0])
        bands.extend(band for _, band in groups)

    if not bands:
        raise ValueError(
            f"period range {min_period:g}-{max_period:g}s did not map to any EMTF FC bands"
        )
    return bands


def _max_fc_for_level(level_index: int, level: DecimationLevel, min_fc: int) -> int:
    nfft = max(1, int(level.nfft))
    if level_index == 1:
        return max(min_fc, nfft - 2)
    return max(min_fc, nfft // 2 + 1)


def _fc_period(level: DecimationLevel, level_sample_rate: float, fc: float) -> float:
    return float(level.nwin) / (float(fc) * level_sample_rate)


def _period_in_bin(period: float, lo: float, hi: float, include_hi: bool) -> bool:
    if include_hi:
        return lo <= period <= hi
    return lo <= period < hi


def _contiguous_ranges(values: Sequence[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    ordered = sorted(values)
    ranges: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))
    return ranges


__all__ = [
    "DEFAULT_NUM_LEVELS",
    "standard_bands",
    "fine_bands",
    "half_decade_bands",
    "period_grid_bands",
    "preset_bands",
]
