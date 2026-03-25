from __future__ import annotations

from typing import Optional, Dict

import numpy as np
from PySide6.QtWidgets import QWidget, QPushButton

import scipy as sp
from scipy import fft, signal
from scipy.linalg import lstsq, inv
import warnings

from core.em_data import Channel

default_option = {
    "window": 2000,
    "overlap": 0.8,
    "taper": "hanning",
    "detrend": True,
    "freq_range": [1e-4, 0.1],  # 假设fs=0.1 Hz
    "n_per_decade": 8,
    "robust_max_iter": 30,
    "robust_tol": 1e-4,
    "robust_c": 1.5
}

class RobustCalculateData:
    option: dict = default_option
    channels: Dict[str, Optional[np.ndarray]] = {
        "Ex": None,
        "Ey": None,
        "Hx": None,
        "Hy": None,
    }
    dt: float = 1.0 # 采样频率默认1Hz
    def __init__(self, ex: Channel, ey: Channel, hx: Channel, hy: Channel):
        self.channels["Ex"] = ex.ts.to_numpy()
        self.channels["Ey"] = ey.ts.to_numpy()
        self.channels["Hx"] = hx.ts.to_numpy()
        self.channels["Hy"] = hy.ts.to_numpy()
        self.dt = ex.dt.total_seconds()

class RobustController(QWidget):
    data: RobustCalculateData
    def __init__(self):
        pass