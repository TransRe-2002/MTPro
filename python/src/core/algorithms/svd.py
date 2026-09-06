"""Hankel low-rank SVD denoising for finite time-series segments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SVDSegmentDiagnostic:
    start: int
    stop: int
    embedding_points: int
    rank: int
    singular_values: tuple[float, ...]


@dataclass(frozen=True)
class SVDDenoiseResult:
    cleaned: pd.Series
    processed_segments: tuple[SVDSegmentDiagnostic, ...]
    skipped_segments: tuple[tuple[int, int], ...]


def svd_denoise_series(
    series: pd.Series,
    embedding_points: int = 101,
    rank: int = 8,
    min_segment_points: int = 128,
) -> SVDDenoiseResult:
    """Denoise finite runs through Hankel rank truncation and diagonal averaging."""

    if embedding_points < 2:
        raise ValueError("embedding_points must be at least two")
    if rank < 1:
        raise ValueError("rank must be positive")
    if min_segment_points < 4:
        raise ValueError("min_segment_points must be at least four")

    values = series.to_numpy(dtype=np.float64, copy=True)
    cleaned = values.copy()
    diagnostics: list[SVDSegmentDiagnostic] = []
    skipped: list[tuple[int, int]] = []
    for start, stop in _finite_runs(np.isfinite(values)):
        size = stop - start
        if size < min_segment_points:
            skipped.append((start, stop))
            continue
        actual_embedding = min(int(embedding_points), size // 2)
        if actual_embedding < 2:
            skipped.append((start, stop))
            continue
        denoised, actual_rank, singular_values = _hankel_svd(values[start:stop], actual_embedding, rank)
        cleaned[start:stop] = denoised
        diagnostics.append(
            SVDSegmentDiagnostic(
                start=start,
                stop=stop,
                embedding_points=actual_embedding,
                rank=actual_rank,
                singular_values=tuple(float(value) for value in singular_values[:16]),
            )
        )
    return SVDDenoiseResult(
        cleaned=pd.Series(cleaned, index=series.index, name=series.name),
        processed_segments=tuple(diagnostics),
        skipped_segments=tuple(skipped),
    )


def _hankel_svd(values: np.ndarray, embedding_points: int, requested_rank: int) -> tuple[np.ndarray, int, np.ndarray]:
    columns = values.size - embedding_points + 1
    hankel = np.lib.stride_tricks.sliding_window_view(values, embedding_points).T
    left, singular_values, right = np.linalg.svd(hankel, full_matrices=False)
    actual_rank = min(int(requested_rank), len(singular_values))
    reconstructed = (left[:, :actual_rank] * singular_values[:actual_rank]) @ right[:actual_rank]

    sums = np.zeros(values.size, dtype=np.float64)
    counts = np.zeros(values.size, dtype=np.float64)
    for row in range(embedding_points):
        sums[row:row + columns] += reconstructed[row]
        counts[row:row + columns] += 1.0
    return sums / counts, actual_rank, singular_values


def _finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in zip(changes[::2], changes[1::2])]


__all__ = ["SVDDenoiseResult", "SVDSegmentDiagnostic", "svd_denoise_series"]
