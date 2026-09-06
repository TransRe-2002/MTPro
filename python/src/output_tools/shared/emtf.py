from __future__ import annotations

"""MT 阻抗估计的共享逻辑：常量、通道检测、后端分派、频带构造、估计预处理。

被 ``robust_estimate.py`` (单站 Robust 估计) 和 ``remote_reference.py`` (远参考)
共用，消除两文件间的逐字重复。
"""

import importlib.util
import logging
from typing import Optional

import numpy as np
from scipy.signal import detrend
from PySide6.QtWidgets import QFormLayout, QWidget

from core.em_data import EMData


# ─── 估计参数默认值 ─────────────────────────────────────────────

DEFAULT_OPTION: dict = {
    "window_size": 1024,
    "step_pct": 50.0,
    "huber_threshold": 1.5,
    "max_iterations": 25,
    "tolerance": 1e-8,
}

BACKEND_EMTF_PYWRAP = "emtf_pywrap"
BACKEND_EMTF_RS = "emtf_rs"

WINDOW_VALUES: list[int] = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
FFT_WINDOW_VALUES: list[int] = [64, 128, 256]

# 各通道绑图颜色 (Ex/Ey 为电场用暖色，Hx/Hy/Hz 为磁场用冷色)
CH_COLOURS: dict[str, tuple[int, int, int]] = {
    "Ex": (220, 50, 50),   # 红
    "Ey": (200, 120, 0),   # 橙
    "Hx": (30, 80, 220),   # 蓝
    "Hy": (30, 160, 80),   # 绿
    "Hz": (140, 40, 180),  # 紫
}


# ─── 通道检测 ───────────────────────────────────────────────────

def detect_channels(channel_names: list[str]) -> dict[str, Optional[str]]:
    """根据通道名称的命名规则自动匹配 Ex/Ey/Hx/Hy/Hz 五道。

    优先匹配第一个符合规则且尚未被占用的通道，避免重复映射。
    """
    mapping: dict[str, Optional[str]] = {
        "Ex": None, "Ey": None, "Hx": None, "Hy": None, "Hz": None,
    }
    for ch in channel_names:
        lower = ch.lower()
        if lower.startswith("e") and "x" in lower and mapping["Ex"] is None:
            mapping["Ex"] = ch
        elif lower.startswith("e") and "y" in lower and mapping["Ey"] is None:
            mapping["Ey"] = ch
        elif (lower.startswith("h") or lower.startswith("b")) and "x" in lower and mapping["Hx"] is None:
            mapping["Hx"] = ch
        elif (lower.startswith("h") or lower.startswith("b")) and "y" in lower and mapping["Hy"] is None:
            mapping["Hy"] = ch
        elif (lower.startswith("h") or lower.startswith("b")) and "z" in lower and mapping["Hz"] is None:
            mapping["Hz"] = ch
    return mapping


def is_e_channel(name: str) -> bool:
    """判断通道是否属于电场分量（名称以 'e' 开头）。"""
    return name.lower().startswith("e")


def is_h_channel(name: str) -> bool:
    """判断通道是否属于磁场分量（名称以 'h' 或 'b' 开头）。"""
    return name.lower().startswith(("h", "b"))


def has_magnetic_channel(data: EMData) -> bool:
    """检查 EMData 是否包含至少一个磁场通道（名称以 'h' 或 'b' 开头）。"""
    return any(ch.lower().startswith(("h", "b")) for ch in data.channel_names())


def has_electric_channel(data: EMData) -> bool:
    """检查 EMData 是否包含至少一个电场通道（名称以 'e' 开头）。"""
    return any(ch.lower().startswith("e") for ch in data.channel_names())


# ─── 后端分派 ───────────────────────────────────────────────────

def native_emtf_rs_available() -> bool:
    """Rust 扩展 emtf_rs 是否可导入。"""
    return importlib.util.find_spec("emtf_rs") is not None


def load_emtf_rs_backend():
    """返回 emtf_rs 模块（原生 Rust 扩展优先，否则回退到 emtf Python 包）。"""
    if native_emtf_rs_available():
        import emtf_rs
        return emtf_rs
    import emtf as emtf_rs
    return emtf_rs


# ─── 频带构造 ───────────────────────────────────────────────────

def emtf_period_grid(
    sample_rate: float,
    window_size: int,
    fft_window_size: int,
    n_bands: int,
):
    """从采样率/窗口/FFT 窗口计算 emtf_rs 的 decimation levels + period-grid 频带。"""
    from emtf.band_presets import period_grid_bands
    from emtf.estimate import decimation_with_fft_window

    decimation_levels = decimation_with_fft_window(sample_rate, fft_window_size)
    dt = 1.0 / sample_rate
    bands = period_grid_bands(
        sample_rate,
        decimation_levels,
        min_period=2.0 * dt,
        max_period=float(window_size) * dt,
        n_edges=n_bands,
    )
    return decimation_levels, bands


def build_band_configs(periods: np.ndarray):
    """从 period 数组构造 emtf_rs.BandConfig 列表。"""
    backend = load_emtf_rs_backend()
    return [
        backend.BandConfig(
            name=f"band_{i+1}",
            min_period=float(periods[i]),
            max_period=float(periods[i+1]),
        )
        for i in range(len(periods) - 1)
    ]


# ─── Qt 辅助 ────────────────────────────────────────────────────

def set_form_row_visible(form: QFormLayout, field: QWidget, visible: bool) -> None:
    """显示/隐藏 QFormLayout 中某一行（label + field）。"""
    label = form.labelForField(field)
    if label is not None:
        label.setVisible(visible)
    field.setVisible(visible)


# ─── 估计预处理 ─────────────────────────────────────────────────

_mt_logger = logging.getLogger(__name__)


def preprocess_channel(arr: np.ndarray, name: str) -> np.ndarray:
    """对通道数据做去趋势（detrend），以移除仪器漂移等低频伪影。

    如果数据中存在 NaN 断点则跳过去趋势，避免 scipy.detrend 报错。
    emtf_rs 内部会自行处理 NaN 和去趋势。
    """
    n_nan = int(np.sum(~np.isfinite(arr)))
    if n_nan > 0:
        _mt_logger.warning(
            "%s: %d NaN in selected range (%.2f%%), detrend skipped",
            name, n_nan, 100.0 * n_nan / len(arr),
        )
        return arr
    return detrend(arr)


def build_spectrum_kwargs(
    *,
    sample_rate: float,
    window_size: int,
    step_pct: float,
    huber_threshold: float,
    max_iterations: int,
    tolerance: float,
    hz_arr: Optional[np.ndarray],
    backend: str,
    fft_window_size: Optional[int] = None,
    n_bands: Optional[int] = None,
    station_name: str = "STATION",
) -> dict:
    """构造传给 emtf_rs.estimate_*_spectrum() 的 kwargs 字典。

    RS 后端只需基本参数；pywrap 后端还需 decimation_levels / bands /
    station_name（通过 :func:`emtf_period_grid` 计算）。
    """
    step_size = max(1, int(window_size * step_pct / 100.0))
    kwargs: dict = {
        "sample_rate_hz": sample_rate,
        "window_size": window_size,
        "hz": hz_arr,
        "step_size": step_size,
        "huber_threshold": huber_threshold,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
    }
    if backend == BACKEND_EMTF_PYWRAP:
        decimation_levels, bands = emtf_period_grid(
            sample_rate, window_size, fft_window_size, n_bands,
        )
        kwargs["decimation_levels"] = decimation_levels
        kwargs["bands"] = bands
        kwargs["station_name"] = station_name
    return kwargs
