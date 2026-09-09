"""NaN-safe Variational Mode Decomposition helpers.

The numerical ADMM kernel lives in ``rust_lib`` when the PyO3 extension is
available.  A NumPy implementation remains as a deterministic development and
test fallback, so this module can be imported before a local Rust wheel exists.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

try:
    import rust_lib
    _HAS_RUST_VMD = hasattr(rust_lib, "vmd_decompose_rs")
except ImportError:
    rust_lib = None  # type: ignore
    _HAS_RUST_VMD = False


@dataclass(frozen=True)
class VMDConfig:
    """Configuration shared by VMD decomposition and reconstruction."""

    mode_count: int = 4
    alpha: float = 2000.0
    tau: float = 0.0
    dc: bool = False
    init: str = "uniform"
    tolerance: float = 1.0e-7
    max_iter: int = 500
    min_segment_points: int = 32

    def __post_init__(self) -> None:
        if self.mode_count <= 0:
            raise ValueError("mode_count must be positive")
        if not np.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be finite and positive")
        if not np.isfinite(self.tau) or self.tau < 0.0:
            raise ValueError("tau must be finite and non-negative")
        if self.init not in {"uniform", "zero"}:
            raise ValueError("init must be 'uniform' or 'zero'")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.min_segment_points < 4:
            raise ValueError("min_segment_points must be at least four")


@dataclass(frozen=True)
class VMDDecomposition:
    """Modes are ordered by ascending center frequency in cycles per sample."""

    modes: np.ndarray
    center_frequencies: np.ndarray
    iterations: int
    convergence: float
    used_rust: bool


@dataclass(frozen=True)
class VMDSegmentDiagnostic:
    start: int
    stop: int
    center_frequencies_hz: tuple[float, ...]
    mode_energies: tuple[float, ...]
    reconstruction_residual_rms: float
    iterations: int
    convergence: float
    used_rust: bool


@dataclass(frozen=True)
class VMDDenoiseResult:
    cleaned: pd.Series
    selected_modes: tuple[int, ...]
    processed_segments: tuple[VMDSegmentDiagnostic, ...]
    skipped_segments: tuple[tuple[int, int], ...]


def vmd_decompose(values: Iterable[float] | np.ndarray, config: VMDConfig | None = None) -> VMDDecomposition:
    """Decompose one finite 1-D signal into narrow-band VMD modes."""

    actual = config or VMDConfig()
    source = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if source.ndim != 1:
        raise ValueError("VMD input must be one-dimensional")
    if source.size < 4:
        raise ValueError("VMD requires at least four samples")
    if not np.isfinite(source).all():
        raise ValueError("VMD input must be finite; segment NaN-containing data first")

    if _HAS_RUST_VMD:
        try:
            modes, frequencies, iterations, convergence = rust_lib.vmd_decompose_rs(
                source,
                actual.mode_count,
                actual.alpha,
                actual.tau,
                actual.dc,
                actual.init,
                actual.tolerance,
                actual.max_iter,
            )
            return VMDDecomposition(
                modes=np.asarray(modes, dtype=np.float64),
                center_frequencies=np.asarray(frequencies, dtype=np.float64),
                iterations=int(iterations),
                convergence=float(convergence),
                used_rust=True,
            )
        except Exception:
            # Source-tree tests and development builds must remain usable even
            # if an older installed extension lacks a compatible VMD symbol.
            logger.warning("Rust VMD failed; falling back to NumPy", exc_info=True)

    modes, frequencies, iterations, convergence = _vmd_decompose_numpy(source, actual)
    return VMDDecomposition(
        modes=modes,
        center_frequencies=frequencies,
        iterations=iterations,
        convergence=convergence,
        used_rust=False,
    )


def vmd_denoise_series(
    series: pd.Series,
    config: VMDConfig | None = None,
    keep_modes: str = "all",
    sample_rate_hz: float | None = None,
) -> VMDDenoiseResult:
    """VMD-decompose every finite run and reconstruct selected modes only."""

    actual = config or VMDConfig()
    values = series.to_numpy(dtype=np.float64, copy=True)
    selected = _parse_keep_modes(keep_modes, actual.mode_count)
    sample_rate = _infer_sample_rate(series.index) if sample_rate_hz is None else float(sample_rate_hz)
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        sample_rate = 0.0

    cleaned = values.copy()
    diagnostics: list[VMDSegmentDiagnostic] = []
    skipped: list[tuple[int, int]] = []
    for start, stop in _finite_runs(np.isfinite(values)):
        if stop - start < actual.min_segment_points:
            skipped.append((start, stop))
            continue
        decomposition = vmd_decompose(values[start:stop], actual)
        cleaned[start:stop] = decomposition.modes[list(selected)].sum(axis=0)
        frequencies_hz = decomposition.center_frequencies * sample_rate
        energies = np.mean(np.square(decomposition.modes), axis=1)
        reconstruction_residual = values[start:stop] - decomposition.modes.sum(axis=0)
        diagnostics.append(
            VMDSegmentDiagnostic(
                start=start,
                stop=stop,
                center_frequencies_hz=tuple(float(value) for value in frequencies_hz),
                mode_energies=tuple(float(value) for value in energies),
                reconstruction_residual_rms=float(np.sqrt(np.mean(np.square(reconstruction_residual)))),
                iterations=decomposition.iterations,
                convergence=decomposition.convergence,
                used_rust=decomposition.used_rust,
            )
        )

    return VMDDenoiseResult(
        cleaned=pd.Series(cleaned, index=series.index, name=series.name),
        selected_modes=selected,
        processed_segments=tuple(diagnostics),
        skipped_segments=tuple(skipped),
    )


def _vmd_decompose_numpy(values: np.ndarray, config: VMDConfig) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Reference implementation matching the Rust kernel's FFT convention."""

    sample_count = values.size
    fft_size = sample_count * 2
    half = fft_size // 2
    mirrored = np.concatenate((values[::-1], values))
    spectrum = np.fft.fft(mirrored)
    target = spectrum[:half + 1].copy()
    frequencies = np.arange(half + 1, dtype=np.float64) / float(fft_size)
    modes_hat = np.zeros((config.mode_count, half + 1), dtype=np.complex128)
    sum_modes = np.zeros(half + 1, dtype=np.complex128)
    lagrange = np.zeros(half + 1, dtype=np.complex128)
    if config.init == "uniform":
        omega = 0.5 * np.arange(config.mode_count, dtype=np.float64) / config.mode_count
    else:
        omega = np.zeros(config.mode_count, dtype=np.float64)
    if config.dc:
        omega[0] = 0.0

    convergence = float("inf")
    iterations = 0
    for iteration in range(config.max_iter):
        delta_norm = 0.0
        previous_norm = 0.0
        for mode in range(config.mode_count):
            old = modes_hat[mode].copy()
            without_current = sum_modes - old
            denominator = 1.0 + config.alpha * np.square(frequencies - omega[mode])
            new = (target - without_current - 0.5 * lagrange) / denominator
            modes_hat[mode] = new
            sum_modes = without_current + new
            delta_norm += float(np.vdot(new - old, new - old).real)
            previous_norm += float(np.vdot(old, old).real)
            if not (config.dc and mode == 0):
                energy = np.square(np.abs(new[1:])).sum()
                if energy > np.finfo(np.float64).eps:
                    omega[mode] = float(np.dot(frequencies[1:], np.square(np.abs(new[1:]))) / energy)
        if config.tau > 0.0:
            lagrange += config.tau * (sum_modes - target)
        convergence = float(np.sqrt(delta_norm / (previous_norm + np.finfo(np.float64).eps)))
        iterations = iteration + 1
        if convergence < config.tolerance:
            break

    full_spectra = np.zeros((config.mode_count, fft_size), dtype=np.complex128)
    full_spectra[:, :half + 1] = modes_hat
    full_spectra[:, half + 1:] = np.conj(modes_hat[:, 1:half][:, ::-1])
    modes = np.fft.ifft(full_spectra, axis=1).real[:, sample_count:]
    order = np.argsort(omega, kind="stable")
    return modes[order], omega[order], iterations, convergence


def _finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in zip(changes[::2], changes[1::2])]


def _parse_keep_modes(value: str, mode_count: int) -> tuple[int, ...]:
    text = str(value).strip().casefold()
    if text in {"", "all", "*"}:
        return tuple(range(mode_count))
    try:
        parsed = tuple(sorted({int(part.strip()) for part in text.split(",") if part.strip()}))
    except ValueError as exc:
        raise ValueError("keep_modes must be 'all' or comma-separated mode indices") from exc
    if not parsed:
        raise ValueError("keep_modes must select at least one mode")
    if parsed[0] < 0 or parsed[-1] >= mode_count:
        raise ValueError(f"keep_modes indices must be within [0, {mode_count - 1}]")
    return parsed


def _infer_sample_rate(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 0.0
    deltas_ns = np.diff(index.asi8)
    positive = deltas_ns[deltas_ns > 0]
    if positive.size == 0:
        return 0.0
    interval_seconds = float(np.median(positive)) / 1.0e9
    return 1.0 / interval_seconds if interval_seconds > 0.0 else 0.0


__all__ = [
    "VMDConfig",
    "VMDDecomposition",
    "VMDDenoiseResult",
    "VMDSegmentDiagnostic",
    "vmd_decompose",
    "vmd_denoise_series",
]
