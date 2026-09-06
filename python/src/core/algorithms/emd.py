"""NaN-safe empirical mode decomposition helpers.

The Rust kernel performs extrema detection, spline envelopes, and sifting.
This module adds the application-level rules: preserve missing samples, run
each finite segment independently, and reconstruct only explicitly selected
IMFs.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


try:
    import rust_lib
    _HAS_RUST_EMD = hasattr(rust_lib, "emd_decompose_rs")
except ImportError:
    rust_lib = None  # type: ignore
    _HAS_RUST_EMD = False


@dataclass(frozen=True)
class EMDConfig:
    max_imfs: int = 8
    max_siftings: int = 100
    sift_tolerance: float = 0.2
    min_segment_points: int = 32

    def __post_init__(self) -> None:
        if self.max_imfs <= 0:
            raise ValueError("max_imfs must be positive")
        if self.max_siftings <= 0:
            raise ValueError("max_siftings must be positive")
        if not np.isfinite(self.sift_tolerance) or self.sift_tolerance <= 0.0:
            raise ValueError("sift_tolerance must be finite and positive")
        if self.min_segment_points < 4:
            raise ValueError("min_segment_points must be at least four")


@dataclass(frozen=True)
class EMDDecomposition:
    """IMFs are ordered from high-frequency to low-frequency extraction order."""

    imfs: np.ndarray
    residue: np.ndarray
    siftings: tuple[int, ...]
    used_rust: bool


@dataclass(frozen=True)
class EMDSegmentDiagnostic:
    start: int
    stop: int
    selected_imfs: tuple[int, ...]
    imf_energies: tuple[float, ...]
    residue_energy: float
    siftings: tuple[int, ...]
    reconstruction_residual_rms: float
    used_rust: bool


@dataclass(frozen=True)
class EMDDenoiseResult:
    cleaned: pd.Series
    processed_segments: tuple[EMDSegmentDiagnostic, ...]
    skipped_segments: tuple[tuple[int, int], ...]


def emd_decompose(values: Iterable[float] | np.ndarray, config: EMDConfig | None = None) -> EMDDecomposition:
    """Decompose one finite 1-D signal into high-to-low empirical IMFs."""

    actual = config or EMDConfig()
    source = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if source.ndim != 1:
        raise ValueError("EMD input must be one-dimensional")
    if source.size < 4:
        raise ValueError("EMD requires at least four samples")
    if not np.isfinite(source).all():
        raise ValueError("EMD input must be finite; segment NaN-containing data first")

    if _HAS_RUST_EMD:
        try:
            imfs, residue, siftings = rust_lib.emd_decompose_rs(
                source,
                actual.max_imfs,
                actual.max_siftings,
                actual.sift_tolerance,
            )
            return EMDDecomposition(
                imfs=np.asarray(imfs, dtype=np.float64),
                residue=np.asarray(residue, dtype=np.float64),
                siftings=tuple(int(value) for value in siftings),
                used_rust=True,
            )
        except Exception:
            logger.warning("Rust EMD failed; falling back to NumPy", exc_info=True)

    imfs, residue, siftings = _emd_decompose_numpy(source, actual)
    return EMDDecomposition(imfs=imfs, residue=residue, siftings=siftings, used_rust=False)


def emd_denoise_series(
    series: pd.Series,
    config: EMDConfig | None = None,
    keep_imfs: str = "all",
    keep_residue: bool = True,
) -> EMDDenoiseResult:
    """Run EMD over each finite segment and reconstruct selected components."""

    actual = config or EMDConfig()
    values = series.to_numpy(dtype=np.float64, copy=True)
    cleaned = values.copy()
    diagnostics: list[EMDSegmentDiagnostic] = []
    skipped: list[tuple[int, int]] = []
    for start, stop in _finite_runs(np.isfinite(values)):
        if stop - start < actual.min_segment_points:
            skipped.append((start, stop))
            continue
        source = values[start:stop]
        decomposition = emd_decompose(source, actual)
        selected = _parse_keep_imfs(keep_imfs, decomposition.imfs.shape[0])
        reconstructed = np.zeros_like(source)
        if selected:
            reconstructed += decomposition.imfs[list(selected)].sum(axis=0)
        if keep_residue:
            reconstructed += decomposition.residue
        cleaned[start:stop] = reconstructed
        full_reconstruction = decomposition.imfs.sum(axis=0) + decomposition.residue
        energies = np.mean(np.square(decomposition.imfs), axis=1)
        diagnostics.append(
            EMDSegmentDiagnostic(
                start=start,
                stop=stop,
                selected_imfs=selected,
                imf_energies=tuple(float(value) for value in energies),
                residue_energy=float(np.mean(np.square(decomposition.residue))),
                siftings=decomposition.siftings,
                reconstruction_residual_rms=float(
                    np.sqrt(np.mean(np.square(source - full_reconstruction)))
                ),
                used_rust=decomposition.used_rust,
            )
        )

    return EMDDenoiseResult(
        cleaned=pd.Series(cleaned, index=series.index, name=series.name),
        processed_segments=tuple(diagnostics),
        skipped_segments=tuple(skipped),
    )


def _emd_decompose_numpy(values: np.ndarray, config: EMDConfig) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    residue = values.copy()
    imfs: list[np.ndarray] = []
    siftings: list[int] = []
    for _ in range(config.max_imfs):
        maxima, minima = _extrema_indices(residue)
        if not maxima or not minima:
            break
        imf, count = _sift(residue, config.max_siftings, config.sift_tolerance)
        imfs.append(imf)
        siftings.append(count)
        residue -= imf
        maxima, minima = _extrema_indices(residue)
        if not maxima or not minima:
            break
    stacked = np.stack(imfs) if imfs else np.empty((0, values.size), dtype=np.float64)
    return stacked, residue, tuple(siftings)


def _sift(source: np.ndarray, max_siftings: int, tolerance: float) -> tuple[np.ndarray, int]:
    component = source.copy()
    for iteration in range(max_siftings):
        maxima, minima = _extrema_indices(component)
        if not maxima or not minima:
            return component, iteration
        upper = _spline_envelope(component, maxima)
        lower = _spline_envelope(component, minima)
        next_component = component - 0.5 * (upper + lower)
        relative_change = float(
            np.sum(np.square(component - next_component))
            / (np.sum(np.square(component)) + np.finfo(np.float64).eps)
        )
        component = next_component
        next_maxima, next_minima = _extrema_indices(component)
        if relative_change <= tolerance and abs(len(next_maxima) + len(next_minima) - _zero_crossings(component)) <= 1:
            return component, iteration + 1
    return component, max_siftings


def _extrema_indices(values: np.ndarray) -> tuple[list[int], list[int]]:
    maxima: list[int] = []
    minima: list[int] = []
    index = 1
    while index + 1 < values.size:
        start = index
        while index + 1 < values.size - 1 and values[index] == values[index + 1]:
            index += 1
        end = index
        before = values[start] - values[start - 1]
        after = values[end + 1] - values[end]
        midpoint = (start + end) // 2
        if before > 0.0 and after < 0.0:
            maxima.append(midpoint)
        elif before < 0.0 and after > 0.0:
            minima.append(midpoint)
        index += 1
    return maxima, minima


def _zero_crossings(values: np.ndarray) -> int:
    signs = np.sign(values)
    nonzero = signs[signs != 0.0]
    return int(np.count_nonzero(nonzero[1:] != nonzero[:-1])) if nonzero.size > 1 else 0


def _spline_envelope(values: np.ndarray, extrema: list[int]) -> np.ndarray:
    nodes = [(0, float(values[0]))]
    nodes.extend((index, float(values[index])) for index in extrema if 0 < index < values.size - 1)
    nodes.append((values.size - 1, float(values[-1])))
    nodes = sorted(dict(nodes).items())
    if len(nodes) == 2:
        return np.interp(np.arange(values.size), *zip(*nodes))

    x = np.asarray([item[0] for item in nodes], dtype=np.float64)
    a = np.asarray([item[1] for item in nodes], dtype=np.float64)
    h = np.diff(x)
    alpha = np.zeros(len(nodes), dtype=np.float64)
    alpha[1:-1] = 3.0 / h[1:] * np.diff(a[1:]) - 3.0 / h[:-1] * np.diff(a[:-1])
    lower = np.ones(len(nodes), dtype=np.float64)
    mu = np.zeros(len(nodes), dtype=np.float64)
    z = np.zeros(len(nodes), dtype=np.float64)
    for index in range(1, len(nodes) - 1):
        lower[index] = 2.0 * (x[index + 1] - x[index - 1]) - h[index - 1] * mu[index - 1]
        if abs(lower[index]) <= np.finfo(np.float64).eps:
            return np.interp(np.arange(values.size), x, a)
        mu[index] = h[index] / lower[index]
        z[index] = (alpha[index] - h[index - 1] * z[index - 1]) / lower[index]
    c = np.zeros(len(nodes), dtype=np.float64)
    b = np.zeros(len(nodes) - 1, dtype=np.float64)
    d = np.zeros(len(nodes) - 1, dtype=np.float64)
    for index in range(len(nodes) - 2, -1, -1):
        c[index] = z[index] - mu[index] * c[index + 1]
        b[index] = (a[index + 1] - a[index]) / h[index] - h[index] * (c[index + 1] + 2.0 * c[index]) / 3.0
        d[index] = (c[index + 1] - c[index]) / (3.0 * h[index])

    samples = np.arange(values.size, dtype=np.float64)
    intervals = np.searchsorted(x[1:], samples, side="left")
    delta = samples - x[intervals]
    return a[intervals] + b[intervals] * delta + c[intervals] * delta**2 + d[intervals] * delta**3


def _finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in zip(changes[::2], changes[1::2])]


def _parse_keep_imfs(value: str, imf_count: int) -> tuple[int, ...]:
    text = str(value).strip().casefold()
    if text in {"", "all", "*"}:
        return tuple(range(imf_count))
    try:
        parsed = tuple(sorted({int(part.strip()) for part in text.split(",") if part.strip()}))
    except ValueError as exc:
        raise ValueError("keep_imfs must be 'all' or comma-separated IMF indices") from exc
    if not parsed:
        raise ValueError("keep_imfs must select at least one IMF")
    if imf_count == 0 or parsed[0] < 0 or parsed[-1] >= imf_count:
        raise ValueError(f"keep_imfs indices must be within [0, {imf_count - 1}]")
    return parsed


__all__ = [
    "EMDConfig",
    "EMDDecomposition",
    "EMDDenoiseResult",
    "EMDSegmentDiagnostic",
    "emd_decompose",
    "emd_denoise_series",
]
