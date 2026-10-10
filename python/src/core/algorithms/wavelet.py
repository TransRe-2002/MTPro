from __future__ import annotations

"""纯数值小波去噪算法，与阶跃检测和 Qt 编辑器解耦。"""

from dataclasses import dataclass
import math
from collections.abc import Iterable

import numpy as np


@dataclass(frozen=True)
class WaveletDenoiseResult:
    """小波去噪一次处理后的输出结果。"""
    corrected: np.ndarray
    correction: np.ndarray
    local_window: tuple[int, int]
    region: tuple[int, int]
    local_original: np.ndarray
    local_reconstructed: np.ndarray
    detail_energy_before: np.ndarray
    detail_energy_after: np.ndarray
    band_labels: list[str]
    band_ranges_hz: list[tuple[float, float]]
    wavelet_name: str
    transform_kind: str
    level: int
    threshold_scale: float
    threshold_method: str
    level_policy: str
    protect_period_seconds: float | None
    padding_points: int
    thresholds: np.ndarray
    detail_gains: np.ndarray
    mode: str
    changed_coefficients: int
    success: bool
    message: str



def _import_pywt():
    try:
        import pywt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("当前环境未安装 PyWavelets，无法执行小波多尺度去噪") from exc
    return pywt


def _format_band_range(low_hz: float, high_hz: float) -> str:
    return f"{low_hz:.4g}-{high_hz:.4g} Hz"


def _dwt_band_ranges(level: int, sample_rate_hz: float) -> list[tuple[float, float]]:
    nyquist = 0.5 * sample_rate_hz
    ranges: list[tuple[float, float]] = []
    for detail_level in range(1, level + 1):
        high = nyquist / (2 ** (detail_level - 1))
        low = nyquist / (2 ** detail_level)
        ranges.append((low, high))
    return ranges


def _wp_band_ranges(level: int, sample_rate_hz: float) -> list[tuple[float, float]]:
    nyquist = 0.5 * sample_rate_hz
    band_count = 2 ** level
    width = nyquist / float(band_count)
    return [(i * width, (i + 1) * width) for i in range(band_count)]


def _mad_noise_sigma(coefficients: np.ndarray) -> float:
    """Estimate Gaussian noise scale from wavelet coefficients with MAD.

    The median is removed before computing the MAD.  That matters for local
    windows whose selected event leaks into a coarse detail band.
    """
    values = np.asarray(coefficients, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1e-12

    median = float(np.median(values))
    sigma = float(np.median(np.abs(values - median)) / 0.67448975)
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(values))
    return max(sigma, 1e-12)


def _universal_threshold(sigma: float, coefficient_count: int) -> float:
    return float(sigma) * math.sqrt(2.0 * math.log(max(2, int(coefficient_count))))


def _sure_threshold(coefficients: np.ndarray, sigma: float) -> float:
    """Return the SureShrink soft-threshold for one coefficient band.

    SURE is evaluated over the finite reference coefficients (normally the
    context around the selected region).  The result is capped by the
    universal threshold, as in SureShrink, so the adaptive rule never becomes
    more aggressive merely because a short context happened to be unusual.
    """
    values = np.asarray(coefficients, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0

    normalized = np.sort(np.abs(values) / max(float(sigma), 1e-12))
    squared = normalized * normalized
    counts = np.arange(1, normalized.size + 1, dtype=np.float64)
    risks = (
        float(normalized.size)
        - 2.0 * counts
        + np.cumsum(squared)
        + (float(normalized.size) - counts) * squared
    )
    # Threshold zero is a valid SURE candidate as well.
    candidate_values = np.concatenate((np.array([0.0]), normalized))
    candidate_risks = np.concatenate((np.array([float(normalized.size)]), risks))
    sure_value = float(candidate_values[int(np.argmin(candidate_risks))] * sigma)
    return min(sure_value, _universal_threshold(sigma, normalized.size))


def _threshold_for_band(
    reference_coefficients: np.ndarray,
    *,
    threshold_scale: float,
    threshold_method: str,
) -> float:
    sigma = _mad_noise_sigma(reference_coefficients)
    if threshold_method == "universal":
        base_threshold = _universal_threshold(sigma, reference_coefficients.size)
    elif threshold_method == "sure":
        base_threshold = _sure_threshold(reference_coefficients, sigma)
    else:  # checked by the caller; keeps this helper safe as a public-test seam.
        raise ValueError(f"unsupported wavelet threshold method: {threshold_method}")
    return float(threshold_scale) * base_threshold


def _level_for_protected_period(
    sample_rate_hz: float,
    protect_period_seconds: float,
) -> int:
    """Choose a DWT/SWT level whose approximation retains periods at least P.

    After level ``J`` the approximation approximately covers
    ``[0, fs / 2**(J + 1)]``.  Choosing ``J`` from this relation lets the UI
    ask for a physical protection period instead of a sampling-rate-dependent
    fixed level.
    """
    period = float(protect_period_seconds)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("protect_period_seconds must be a positive finite number")
    samples_per_period = max(2.0, float(sample_rate_hz) * period)
    return max(1, int(math.ceil(math.log2(samples_per_period))) - 1)


def wavelet_multiscale_denoise(
    samples: np.ndarray,
    region: tuple[int, int],
    *,
    transform_kind: str = "dwt",
    wavelet_name: str = "db4",
    level: int = 6,
    sample_rate_hz: float | None = None,
    threshold_scale: float = 1.0,
    padding_points: int = 256,
    padding_seconds: float | None = None,
    mode: str = "soft",
    threshold_method: str = "universal",
    level_policy: str = "manual",
    protect_period_seconds: float | None = None,
    detail_gains: Iterable[float] | None = None,
) -> WaveletDenoiseResult:
    pywt = _import_pywt()

    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError("samples must be a 1D array")
    if data.size < 16:
        raise ValueError("samples is too short")

    left, right = sorted((int(region[0]), int(region[1])))
    left = max(0, left)
    right = min(data.size - 1, right)
    if right <= left:
        right = min(data.size - 1, left + 1)

    threshold_scale = float(threshold_scale)
    if threshold_scale < 0.0:
        raise ValueError("threshold_scale must be >= 0")

    if mode not in {"soft", "hard", "zero", "manual"}:
        raise ValueError(f"unsupported wavelet denoise mode: {mode}")
    if transform_kind not in {"dwt", "swt", "wp"}:
        raise ValueError(f"unsupported transform kind: {transform_kind}")
    if threshold_method not in {"universal", "sure"}:
        raise ValueError(f"unsupported wavelet threshold method: {threshold_method}")
    if level_policy not in {"manual", "auto_by_period"}:
        raise ValueError(f"unsupported wavelet level policy: {level_policy}")
    if level_policy == "auto_by_period" and transform_kind == "wp":
        raise ValueError("automatic protection-period levels are only available for DWT or SWT")
    if sample_rate_hz is None or not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        sample_rate_hz = 1.0

    selected_width = right - left + 1
    if padding_seconds is not None:
        padding_seconds = float(padding_seconds)
        if not np.isfinite(padding_seconds) or padding_seconds < 0.0:
            raise ValueError("padding_seconds must be a non-negative finite number")
        padding_points = int(math.ceil(padding_seconds * float(sample_rate_hz)))
    padding_points = max(int(padding_points), 8)
    half_window = max(padding_points, int(np.ceil(selected_width * 0.5)))
    local_start = max(0, left - half_window)
    local_stop = min(data.size, right + half_window + 1)
    original_local_segment = np.array(data[local_start:local_stop], copy=True, dtype=np.float64)
    if original_local_segment.size < 16:
        raise ValueError("local window is too short for wavelet processing")

    finite_mask = np.isfinite(original_local_segment)
    local_segment = np.array(original_local_segment, copy=True)
    if not np.all(finite_mask):
        finite_indices = np.flatnonzero(finite_mask)
        if finite_indices.size < 2:
            raise ValueError("not enough finite samples in local window")
        local_segment = np.interp(
            np.arange(local_segment.size, dtype=np.float64),
            finite_indices.astype(np.float64),
            local_segment[finite_mask],
        )

    wavelet = pywt.Wavelet(wavelet_name)
    if transform_kind == "swt":
        max_level = int(math.floor(math.log2(local_segment.size)))
    else:
        max_level = pywt.dwt_max_level(local_segment.size, wavelet.dec_len)
    if max_level < 1:
        raise ValueError("local window is too short for the requested wavelet")
    if level_policy == "auto_by_period":
        if protect_period_seconds is None:
            raise ValueError("protect_period_seconds is required for automatic levels")
        level = _level_for_protected_period(float(sample_rate_hz), protect_period_seconds)
    level = max(1, min(int(level), int(max_level)))

    changed_coefficients = 0
    local_left = left - local_start
    local_right = right - local_start

    if transform_kind == "dwt":
        coeffs = pywt.wavedec(local_segment, wavelet, mode="symmetric", level=level)
        modified_coeffs = [np.array(coeffs[0], copy=True)]
        detail_energy_before_desc: list[float] = []
        detail_energy_after_desc: list[float] = []
        thresholds_desc: list[float] = []
        band_ranges_hz = _dwt_band_ranges(level, float(sample_rate_hz))
        band_labels = [
            f"D{i} [{_format_band_range(*band_ranges_hz[i - 1])}]"
            for i in range(1, level + 1)
        ]
        gains_array = np.ones(level, dtype=np.float64)
        if detail_gains is not None:
            requested_gains = np.asarray(list(detail_gains), dtype=np.float64)
            if requested_gains.size > 0:
                usable = min(level, requested_gains.size)
                gains_array[:usable] = requested_gains[:usable]

        for coeff_index in range(1, len(coeffs)):
            detail_level = level - coeff_index + 1
            detail = np.array(coeffs[coeff_index], copy=True, dtype=np.float64)
            scale = 2 ** detail_level
            guard = scale
            coeff_left = max(0, int(np.floor((local_left - guard) / float(scale))))
            coeff_right = min(
                detail.size,
                int(np.ceil((local_right + 1 + guard) / float(scale))),
            )

            detail_energy_before_desc.append(float(np.sum(detail * detail)))
            if mode == "manual":
                ui_level = len(coeffs) - coeff_index
                if 1 <= ui_level <= gains_array.size:
                    gain = float(gains_array[ui_level - 1])
                else:
                    gain = 1.0
                updated = detail * gain
                threshold = 0.0
            else:
                background = np.concatenate((detail[:coeff_left], detail[coeff_right:]))
                if background.size == 0:
                    background = detail
                threshold = _threshold_for_band(
                    background,
                    threshold_scale=threshold_scale,
                    threshold_method=threshold_method,
                )
                if mode == "zero":
                    updated = np.zeros_like(detail)
                else:
                    updated = pywt.threshold(detail, value=float(threshold), mode=mode)

            changed_coefficients += int(np.count_nonzero(np.abs(updated - detail) > 1e-12))
            detail = np.asarray(updated, dtype=np.float64)
            modified_coeffs.append(detail)
            detail_energy_after_desc.append(float(np.sum(detail * detail)))
            thresholds_desc.append(float(threshold))

        reconstructed_local = pywt.waverec(modified_coeffs, wavelet, mode="symmetric")
        detail_energy_before = np.asarray(detail_energy_before_desc[::-1], dtype=np.float64)
        detail_energy_after = np.asarray(detail_energy_after_desc[::-1], dtype=np.float64)
        thresholds = np.asarray(thresholds_desc[::-1], dtype=np.float64)
    elif transform_kind == "swt":
        # SWT requires a length divisible by 2**level.  Symmetric padding keeps
        # the selected-window edges from becoming artificial discontinuities.
        divisor = 2 ** level
        padded_length = int(math.ceil(local_segment.size / float(divisor)) * divisor)
        padded_segment = np.pad(local_segment, (0, padded_length - local_segment.size), mode="symmetric")
        coeffs = pywt.swt(padded_segment, wavelet, level=level, trim_approx=False)
        modified_coeffs = []
        detail_energy_before_desc = []
        detail_energy_after_desc = []
        thresholds_desc = []
        band_ranges_hz = _dwt_band_ranges(level, float(sample_rate_hz))
        band_labels = [
            f"D{i} [{_format_band_range(*band_ranges_hz[i - 1])}]"
            for i in range(1, level + 1)
        ]
        gains_array = np.ones(level, dtype=np.float64)
        if detail_gains is not None:
            requested_gains = np.asarray(list(detail_gains), dtype=np.float64)
            if requested_gains.size > 0:
                usable = min(level, requested_gains.size)
                gains_array[:usable] = requested_gains[:usable]

        for coeff_index, (approximation, detail_values) in enumerate(coeffs):
            detail_level = level - coeff_index
            detail = np.array(detail_values, copy=True, dtype=np.float64)
            guard = 2 ** detail_level
            coeff_left = max(0, local_left - guard)
            coeff_right = min(detail.size, local_right + 1 + guard)
            detail_energy_before_desc.append(float(np.sum(detail * detail)))
            if mode == "manual":
                gain = float(gains_array[detail_level - 1])
                updated = detail * gain
                threshold = 0.0
            else:
                background = np.concatenate((detail[:coeff_left], detail[coeff_right:]))
                if background.size == 0:
                    background = detail
                threshold = _threshold_for_band(
                    background,
                    threshold_scale=threshold_scale,
                    threshold_method=threshold_method,
                )
                if mode == "zero":
                    updated = np.zeros_like(detail)
                else:
                    updated = pywt.threshold(detail, value=float(threshold), mode=mode)
            changed_coefficients += int(np.count_nonzero(np.abs(updated - detail) > 1e-12))
            modified_coeffs.append((np.asarray(approximation, dtype=np.float64), np.asarray(updated, dtype=np.float64)))
            detail_energy_after_desc.append(float(np.sum(np.asarray(updated, dtype=np.float64) ** 2)))
            thresholds_desc.append(float(threshold))

        reconstructed_local = pywt.iswt(modified_coeffs, wavelet)[:local_segment.size]
        detail_energy_before = np.asarray(detail_energy_before_desc[::-1], dtype=np.float64)
        detail_energy_after = np.asarray(detail_energy_after_desc[::-1], dtype=np.float64)
        thresholds = np.asarray(thresholds_desc[::-1], dtype=np.float64)
    else:
        wp = pywt.WaveletPacket(
            data=local_segment, wavelet=wavelet, mode="symmetric", maxlevel=level,
        )
        nodes = wp.get_level(level, order="freq")
        band_ranges_hz = _wp_band_ranges(level, float(sample_rate_hz))
        band_labels = [
            f"{(node.path or 'a')} [{_format_band_range(*band_ranges_hz[i])}]"
            for i, node in enumerate(nodes)
        ]
        gains_array = np.ones(len(nodes), dtype=np.float64)
        if detail_gains is not None:
            requested_gains = np.asarray(list(detail_gains), dtype=np.float64)
            if requested_gains.size > 0:
                usable = min(len(nodes), requested_gains.size)
                gains_array[:usable] = requested_gains[:usable]

        detail_energy_before_list: list[float] = []
        detail_energy_after_list: list[float] = []
        thresholds_list: list[float] = []
        for band_index, node in enumerate(nodes):
            data_band = np.array(node.data, copy=True, dtype=np.float64)
            detail_energy_before_list.append(float(np.sum(data_band * data_band)))
            if mode == "manual":
                updated = data_band * float(gains_array[band_index])
                threshold = 0.0
            else:
                threshold = _threshold_for_band(
                    data_band,
                    threshold_scale=threshold_scale,
                    threshold_method=threshold_method,
                )
                if mode == "zero":
                    updated = np.zeros_like(data_band)
                else:
                    updated = pywt.threshold(data_band, value=float(threshold), mode=mode)
            changed_coefficients += int(np.count_nonzero(np.abs(updated - data_band) > 1e-12))
            wp[node.path] = np.asarray(updated, dtype=np.float64)
            energy_val = float(np.sum(
                np.asarray(updated, dtype=np.float64) ** 2,
            ))
            detail_energy_after_list.append(energy_val)
            thresholds_list.append(float(threshold))

        reconstructed_local = wp.reconstruct(update=True)
        detail_energy_before = np.asarray(detail_energy_before_list, dtype=np.float64)
        detail_energy_after = np.asarray(detail_energy_after_list, dtype=np.float64)
        thresholds = np.asarray(thresholds_list, dtype=np.float64)

    reconstructed_local = np.asarray(reconstructed_local[:local_segment.size], dtype=np.float64)

    blend_width = min(
        max(4, selected_width),
        local_left,
        max(0, local_segment.size - local_right - 1),
        half_window,
    )
    weights = np.ones(local_segment.size, dtype=np.float64)
    if blend_width > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, blend_width, dtype=np.float64))
        weights[:blend_width] = ramp
        weights[-blend_width:] = ramp[::-1]

    corrected_local = local_segment + weights * (reconstructed_local - local_segment)
    correction_local = local_segment - corrected_local
    # Numerical processing uses interpolation only internally.  Missing input
    # samples are data-quality information and must never silently turn into
    # denoised observations in a batch/plugin invocation.
    corrected_local[~finite_mask] = np.nan
    correction_local[~finite_mask] = 0.0

    corrected = np.array(data, copy=True)
    corrected[local_start:local_stop] = corrected_local
    correction = np.zeros_like(corrected)
    correction[local_start:local_stop] = correction_local

    return WaveletDenoiseResult(
        corrected=corrected,
        correction=correction,
        local_window=(local_start, local_stop - 1),
        region=(left, right),
        local_original=original_local_segment,
        local_reconstructed=corrected_local,
        detail_energy_before=detail_energy_before,
        detail_energy_after=detail_energy_after,
        band_labels=band_labels,
        band_ranges_hz=band_ranges_hz,
        wavelet_name=wavelet_name,
        transform_kind=transform_kind,
        level=level,
        threshold_scale=float(threshold_scale),
        threshold_method=threshold_method,
        level_policy=level_policy,
        protect_period_seconds=(
            float(protect_period_seconds) if protect_period_seconds is not None else None
        ),
        padding_points=padding_points,
        thresholds=thresholds,
        detail_gains=np.asarray(gains_array, dtype=np.float64),
        mode=mode,
        changed_coefficients=changed_coefficients,
        success=True,
        message="wavelet multiscale denoise finished",
    )


WaveletPulseResult = WaveletDenoiseResult


def wavelet_local_depulse(
    samples: np.ndarray,
    region: tuple[int, int],
    *,
    wavelet_name: str = "db4",
    level: int = 6,
    sample_rate_hz: float | None = None,
    threshold_scale: float = 1.0,
    padding_points: int = 256,
    mode: str = "soft",
) -> WaveletDenoiseResult:
    return wavelet_multiscale_denoise(
        samples,
        region,
        transform_kind="dwt",
        wavelet_name=wavelet_name,
        level=level,
        sample_rate_hz=sample_rate_hz,
        threshold_scale=threshold_scale,
        padding_points=padding_points,
        mode=mode,
    )
