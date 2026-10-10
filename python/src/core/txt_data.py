from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from core.em_data import EMData


class TxtEMData(EMData):
    """从 MT 采集仪空白分隔 TXT 文件导入的时序数据。"""

    def __init__(
        self,
        channel_series: dict[str, pd.Series],
        timezone: str,
        dt: pd.Timedelta,
        path: str,
        lat: float | None = None,
        lon: float | None = None,
        name: str | None = None,
        diagnostics: dict | None = None,
    ):
        super().__init__(path)
        if not channel_series:
            raise ValueError("没有可导入的通道")

        self.timezone = timezone
        self.dt = dt
        self.latitude = lat
        self.longitude = lon
        self.elevation = None
        # TXT 本身不声明单位；当前格式按 MTPro/MT Portable 的约定解释。
        self.e_units = "mV/km"
        self.m_units = "nT"
        self.name = name if name else "txt"
        self.import_diagnostics = diagnostics or {}

        first_series = next(iter(channel_series.values()))
        self.datetime_index = pd.DatetimeIndex(first_series.index)
        self.npts = len(self.datetime_index)
        self.NCh = len(channel_series)
        self.chid = list(channel_series.keys())

        self.start_time = self.datetime_index[0]
        self.end_time = self.datetime_index[-1]
        self.zero_time = pd.Timestamp(
            year=self.start_time.year,
            month=1,
            day=1,
            tz=self.start_time.tz,
        )

        data_vars: dict[str, xr.DataArray] = {}
        for channel_name, series in channel_series.items():
            values = series.to_numpy(dtype=np.float64, copy=False)
            data_vars[channel_name] = xr.DataArray(
                values,
                dims=("time",),
                coords={"time": self.datetime_index},
            )
        self.data = xr.Dataset(data_vars, coords={"time": self.datetime_index})
        self.kp_data = None

    def restore_data(self, ch: str):
        pass
