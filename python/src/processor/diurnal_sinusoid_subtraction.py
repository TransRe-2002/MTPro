from __future__ import annotations

"""日变/周期性窄带信号的正弦拟合与相减算法。

本模块只负责“数值处理”，不涉及任何界面逻辑。核心思路是：
1. 把整条时间序列按固定长度切成多个重叠时间窗。
2. 在每个时间窗内，用“趋势项 + 正弦/余弦谐波项”做最小二乘拟合。
3. 只提取拟合结果中的谐波部分，视为待去除的周期性噪声。
4. 将各个时间窗得到的噪声结果按窗函数加权拼接，再从原始序列中相减。

对于当前项目中的“去日变”工具，这里的基础周期默认是 24 小时，
常用谐波是 1 阶和 2 阶，也就是 24h 与 12h 两个周期分量。
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from core.em_data import EMData


SECONDS_PER_DAY = 24.0 * 60.0 * 60.0


@dataclass(frozen=True)
class PeriodicSubtractionConfig:
    """周期性信号相减算法的配置参数。

    这些参数决定：
    - 目标基础周期是什么
    - 要拟合到第几阶谐波
    - 每个拟合窗口多长、相邻窗口隔多远
    - 允许多高阶的低频趋势背景
    - 当前窗口至少要有多少有效数据才参与拟合
    """

    base_period_seconds: float = SECONDS_PER_DAY
    harmonics: tuple[int, ...] = (1, 2)
    window_duration_seconds: float = 3.0 * SECONDS_PER_DAY
    step_duration_seconds: float = 1.0 * SECONDS_PER_DAY
    trend_degree: int = 1
    min_valid_fraction: float = 0.6
    min_valid_samples: int = 128

    def __post_init__(self) -> None:
        """在 dataclass 构造完成后统一做参数合法性检查。"""
        harmonics = tuple(int(harmonic) for harmonic in self.harmonics)
        if self.base_period_seconds <= 0.0:
            raise ValueError("base_period_seconds must be positive")
        if not harmonics:
            raise ValueError("harmonics must not be empty")
        if any(harmonic <= 0 for harmonic in harmonics):
            raise ValueError("harmonics must contain positive integers")
        if self.window_duration_seconds <= 0.0:
            raise ValueError("window_duration_seconds must be positive")
        if self.step_duration_seconds <= 0.0:
            raise ValueError("step_duration_seconds must be positive")
        if self.trend_degree < 0:
            raise ValueError("trend_degree must be >= 0")
        if not 0.0 < self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be within (0, 1]")
        if self.min_valid_samples <= 0:
            raise ValueError("min_valid_samples must be positive")
        object.__setattr__(self, "harmonics", harmonics)


@dataclass
class PeriodicSubtractionResult:
    """一次周期相减处理后的输出结果。

    cleaned:
        去掉目标周期信号后的结果。
    estimated_noise:
        被拟合并准备减掉的周期性噪声。
    coverage_weight:
        每个采样点最终由多少窗口权重叠加而来，可用于诊断边缘区结果是否可靠。
    window_diagnostics:
        每个拟合窗口的状态、振幅、残差等诊断信息。
    """

    cleaned: pd.Series
    estimated_noise: pd.Series
    coverage_weight: pd.Series
    window_diagnostics: pd.DataFrame
    config: PeriodicSubtractionConfig


def subtract_periodic_signal_series(
    series: pd.Series,
    config: PeriodicSubtractionConfig | None = None,
) -> PeriodicSubtractionResult:
    """对单条等采样时间序列执行“周期拟合 + 相减”。

    这是本模块最核心的入口。它不关心具体是“日变”还是别的周期，
    只按照 config 中给定的基础周期与谐波集合去做。
    """

    if config is None:
        config = PeriodicSubtractionConfig()
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series index must be a pandas.DatetimeIndex")
    if series.empty:
        raise ValueError("series must not be empty")

    values = series.to_numpy(dtype=np.float64, copy=True)
    dt_seconds = _infer_sample_interval_seconds(series.index)
    sample_count = len(values)
    if sample_count < 2:
        raise ValueError("series must contain at least two samples")

    # 把“秒”为单位的窗口长度/步长换算成“样本点数”。
    window_size = min(sample_count, max(2, int(round(config.window_duration_seconds / dt_seconds))))
    step_size = max(1, int(round(config.step_duration_seconds / dt_seconds)))
    starts = _window_starts(sample_count, window_size, step_size)
    weights = _window_weights(window_size)

    # noise_sum / weight_sum 采用“先分别累计、最后统一归一化”的方式，
    # 避免各窗口拼接时在边界处出现生硬跳变。
    noise_sum = np.zeros(sample_count, dtype=np.float64)
    weight_sum = np.zeros(sample_count, dtype=np.float64)
    diagnostics_rows: list[dict[str, object]] = []

    for window_id, start in enumerate(starts):
        stop = start + window_size
        segment = values[start:stop]
        valid_mask = np.isfinite(segment)
        valid_samples = int(valid_mask.sum())
        valid_fraction = valid_samples / float(window_size)
        row: dict[str, object] = {
            "window_id": window_id,
            "start_time": series.index[start],
            "end_time": series.index[stop - 1],
            "valid_samples": valid_samples,
            "valid_fraction": valid_fraction,
            "status": "ok",
        }

        for harmonic in config.harmonics:
            row[f"harmonic_{harmonic}_amplitude"] = np.nan

        # 当前窗口有效样本不足时跳过，不强行拟合，以免在长缺测区得到不可信结果。
        if valid_samples < config.min_valid_samples or valid_fraction < config.min_valid_fraction:
            row["status"] = "skipped_insufficient_valid"
            diagnostics_rows.append(row)
            continue

        # design_matrix 包含“趋势项 + 全部谐波项”，
        # harmonic_matrix 只包含谐波项，便于后面单独取出“要减掉的噪声部分”。
        design_matrix, harmonic_matrix = _design_matrices(
            window_size=window_size,
            sample_interval_seconds=dt_seconds,
            config=config,
        )
        coefficients, *_ = np.linalg.lstsq(
            design_matrix[valid_mask],
            segment[valid_mask],
            rcond=None,
        )

        # 完整拟合 = 趋势背景 + 周期谐波。
        fitted_segment = design_matrix @ coefficients
        harmonic_offset = design_matrix.shape[1] - harmonic_matrix.shape[1]
        harmonic_coefficients = coefficients[harmonic_offset:]
        # 只用谐波系数重建周期噪声，不把低频趋势一并减掉。
        harmonic_signal = harmonic_matrix @ harmonic_coefficients

        noise_sum[start:stop] += harmonic_signal * weights
        weight_sum[start:stop] += weights

        residual = segment[valid_mask] - fitted_segment[valid_mask]
        row["residual_rms"] = float(np.sqrt(np.mean(residual ** 2)))
        row["harmonic_rms"] = float(np.sqrt(np.mean(harmonic_signal[valid_mask] ** 2)))
        for harmonic_index, harmonic in enumerate(config.harmonics):
            cos_coefficient = harmonic_coefficients[2 * harmonic_index]
            sin_coefficient = harmonic_coefficients[2 * harmonic_index + 1]
            row[f"harmonic_{harmonic}_amplitude"] = float(
                np.hypot(cos_coefficient, sin_coefficient)
            )
        diagnostics_rows.append(row)

    # 各窗口都算完后，再按累计权重归一化，得到最终的连续噪声估计。
    estimated_noise = np.divide(
        noise_sum,
        weight_sum,
        out=np.zeros_like(noise_sum),
        where=weight_sum > 0.0,
    )
    cleaned = values - estimated_noise
    cleaned[~np.isfinite(values)] = np.nan
    estimated_noise[~np.isfinite(values)] = np.nan

    return PeriodicSubtractionResult(
        cleaned=pd.Series(cleaned, index=series.index, name=series.name),
        estimated_noise=pd.Series(estimated_noise, index=series.index, name=series.name),
        coverage_weight=pd.Series(weight_sum, index=series.index, name=f"{series.name}_coverage"),
        window_diagnostics=pd.DataFrame(diagnostics_rows),
        config=config,
    )


def subtract_diurnal_signal_series(
    series: pd.Series,
    *,
    harmonics: Iterable[int] = (1, 2),
    window_days: float = 3.0,
    step_days: float = 1.0,
    trend_degree: int = 1,
    min_valid_fraction: float = 0.6,
    min_valid_samples: int = 128,
) -> PeriodicSubtractionResult:
    """去日变专用包装函数。

    这里只是把“以天为单位”的参数翻译成通用周期相减函数所需的秒单位配置。
    """

    return subtract_periodic_signal_series(
        series,
        config=PeriodicSubtractionConfig(
            base_period_seconds=SECONDS_PER_DAY,
            harmonics=tuple(harmonics),
            window_duration_seconds=window_days * SECONDS_PER_DAY,
            step_duration_seconds=step_days * SECONDS_PER_DAY,
            trend_degree=trend_degree,
            min_valid_fraction=min_valid_fraction,
            min_valid_samples=min_valid_samples,
        ),
    )


def subtract_periodic_signal_channel(
    em_data: EMData,
    channel_name: str,
    config: PeriodicSubtractionConfig | None = None,
    *,
    inplace: bool = False,
) -> PeriodicSubtractionResult:
    """直接对 EMData 中的某个通道执行周期相减。"""
    series = em_data.get_channel_series(channel_name)
    result = subtract_periodic_signal_series(series, config=config)
    if inplace:
        em_data.set_channel_series(channel_name, result.cleaned)
    return result


def subtract_diurnal_signal_channel(
    em_data: EMData,
    channel_name: str,
    *,
    harmonics: Iterable[int] = (1, 2),
    window_days: float = 3.0,
    step_days: float = 1.0,
    trend_degree: int = 1,
    min_valid_fraction: float = 0.6,
    min_valid_samples: int = 128,
    inplace: bool = False,
) -> PeriodicSubtractionResult:
    """直接对 EMData 中的某个通道执行去日变处理。"""
    return subtract_periodic_signal_channel(
        em_data,
        channel_name,
        config=PeriodicSubtractionConfig(
            base_period_seconds=SECONDS_PER_DAY,
            harmonics=tuple(harmonics),
            window_duration_seconds=window_days * SECONDS_PER_DAY,
            step_duration_seconds=step_days * SECONDS_PER_DAY,
            trend_degree=trend_degree,
            min_valid_fraction=min_valid_fraction,
            min_valid_samples=min_valid_samples,
        ),
        inplace=inplace,
    )


def _infer_sample_interval_seconds(index: pd.DatetimeIndex) -> float:
    """推断时间索引的采样间隔，并强制要求等采样。"""
    if len(index) < 2:
        raise ValueError("index must contain at least two samples")
    deltas = np.diff(index.asi8) / 1e9
    first_delta = float(deltas[0])
    if first_delta <= 0.0:
        raise ValueError("series index must be strictly increasing")
    if not np.allclose(deltas, first_delta, rtol=0.0, atol=1e-6):
        raise ValueError("series index must be uniformly sampled")
    return first_delta


def _window_starts(sample_count: int, window_size: int, step_size: int) -> list[int]:
    """生成所有滑动窗口的起点索引。

    末尾不足一个整步长时，会额外补一个“贴到序列末端”的窗口，
    这样最后一段数据也能参与拟合。
    """
    if sample_count <= window_size:
        return [0]
    starts = list(range(0, sample_count - window_size + 1, step_size))
    last_start = sample_count - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _window_weights(window_size: int) -> np.ndarray:
    """生成窗口内的拼接权重。

    这里使用 Hann 型权重，目的是降低窗口边缘在拼接时的接缝感。
    """
    if window_size <= 2:
        return np.ones(window_size, dtype=np.float64)
    return np.hanning(window_size + 2).astype(np.float64)[1:-1]


def _design_matrices(
    *,
    window_size: int,
    sample_interval_seconds: float,
    config: PeriodicSubtractionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """构造当前时间窗的设计矩阵。

    返回两个矩阵：
    - design_matrix: 趋势项 + 全部谐波项
    - harmonic_matrix: 仅谐波项
    """

    times = (np.arange(window_size, dtype=np.float64) - 0.5 * (window_size - 1)) * sample_interval_seconds
    scale = max(np.max(np.abs(times)), 1.0)
    normalized_time = times / scale

    # 趋势项使用归一化时间构造多项式基函数，
    # 这样可以避免原始秒数太大导致数值条件变差。
    trend_columns = [
        normalized_time ** degree
        for degree in range(config.trend_degree + 1)
    ]
    harmonic_columns = []
    angular_frequency = 2.0 * np.pi / config.base_period_seconds
    for harmonic in config.harmonics:
        # 每个谐波都放入 cos/sin 两列，便于自由表示任意相位。
        phase = angular_frequency * harmonic * times
        harmonic_columns.append(np.cos(phase))
        harmonic_columns.append(np.sin(phase))

    harmonic_matrix = np.column_stack(harmonic_columns)
    design_matrix = np.column_stack([*trend_columns, *harmonic_columns])
    return design_matrix, harmonic_matrix
