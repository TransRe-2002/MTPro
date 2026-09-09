from __future__ import annotations

"""演示数据和命令行参数加载的共享工具，供各处理器窗口的 standalone demo 使用。"""

import argparse
import sys

import numpy as np
import pandas as pd
import xarray as xr

from core.em_data import EMData
from io_utils.mat_io import MatLoader


class DemoEMData(EMData):
    """轻量 EMData 子类，用于在没有 .mat 文件时快速构建演示数据。"""

    def __init__(self, name: str, path: str, channel_name: str, series: pd.Series):
        super().__init__(path)
        self.name = name
        self.kp_data = None
        self.start_time = series.index[0]
        self.end_time = series.index[-1]
        self.dt = series.index[1] - series.index[0] if len(series.index) > 1 else pd.Timedelta(seconds=1)
        self.datetime_index = series.index
        self.data = xr.Dataset(coords={"time": self.datetime_index})
        self.data[channel_name] = xr.DataArray(
            series.to_numpy(dtype=np.float64, copy=True),
            dims=("time",),
            coords={"time": self.datetime_index},
        )

    def restore_data(self, ch: str):
        return None


def load_channel_from_args(mat_path: str, channel_name: str | None) -> tuple[EMData, str]:
    """从命令行参数加载 MAT 文件并选择通道。"""
    em_data = MatLoader.load(mat_path)
    if em_data.NCh == 0:
        raise ValueError(f"文件中未找到任何通道: {mat_path}")

    if channel_name is None:
        first = em_data.first_channel_name()
        if first is None:
            raise ValueError(f"文件中未找到任何通道: {mat_path}")
        return em_data, first

    if not em_data.has_channel(channel_name):
        available = ", ".join(em_data.channel_names())
        raise ValueError(f"未找到通道 {channel_name}，可用通道: {available}")
    return em_data, channel_name


def run_standalone_demo(
    widget_class,
    demo_name: str,
    mat_path: str | None,
    channel_name: str | None,
    build_demo_fn,
) -> None:
    """统一的 standalone demo 启动流程。"""
    from PySide6.QtWidgets import QApplication

    if mat_path:
        em_data, channel_name = load_channel_from_args(mat_path, channel_name)
    else:
        em_data, channel_name = build_demo_fn()

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    widget = widget_class(em_data, channel_name)
    widget.setWindowTitle(f"{demo_name} - {channel_name}")
    widget.resize(1500, 920)
    widget.show()
    app.exec()
