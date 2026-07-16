import logging
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from core.em_data import EMData

logger = logging.getLogger(__name__)


class BinEMData(EMData):
    """从二进制文件 (.dat) 导入的 EM 数据。

    构造函数直接接收已应用缩放因子后的通道序列 —
    所有仪器参数（缩放因子、电极距等）仅在导入时使用，
    不存入 EMData。
    """

    def __init__(
        self,
        channel_series: dict[str, pd.Series],
        timezone: str,
        dt: pd.Timedelta,
        path: str,
        lat: Optional[float]  = None,
        lon: Optional[float] = None,
        elev: Optional[float] = None,
        name: Optional[str] = None,
    ):
        super().__init__(path)
        self.timezone = timezone
        self.dt = dt

        self.latitude = lat
        self.longitude = lon
        self.elevation = elev

        # 单位固定
        self.e_units = "mV/km"
        self.m_units = "nT"

        # 从第一条 channel 的索引提取公共时间轴
        first_series = next(iter(channel_series.values()))
        self.datetime_index = first_series.index
        self.npts = len(self.datetime_index)
        self.NCh = len(channel_series)
        self.chid = list(channel_series.keys())

        self.name = name if name else (self.chid[0] if self.chid else "")

        # 时间边界由公式计算（忽略记录间隙）
        self.start_time = self.datetime_index[0]
        self.end_time = self.start_time + self.dt * (self.npts - 1)

        # zero_time = 数据年份的 1 月 1 日 00:00:00
        self.zero_time = pd.Timestamp(
            year=self.start_time.year, month=1, day=1,
            tz=self.start_time.tz,
        )

        # 构建 xr.Dataset
        data_vars: dict[str, xr.DataArray] = {}
        for ch_name, series in channel_series.items():
            values = np.asarray(series.values, dtype=np.float64)
            data_vars[ch_name] = xr.DataArray(
                values,
                dims=("time",),
                coords={"time": self.datetime_index},
            )

        self.data = xr.Dataset(data_vars, coords={"time": self.datetime_index})

        self.kp_data = None

    def restore_data(self, ch: str):
        pass
