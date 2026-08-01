from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from core.em_data import EMData


class XlsxEMData(EMData):
    """从 xlsx 时序工作簿导入的 EM 数据。"""

    def __init__(
        self,
        channel_series: dict[str, pd.Series],
        timezone: str,
        dt: pd.Timedelta,
        path: str,
        name: Optional[str] = None,
        diagnostics: Optional[dict] = None,
    ):
        super().__init__(path)
        if not channel_series:
            raise ValueError("没有可导入的通道")

        self.timezone = timezone
        self.dt = dt
        self.e_units = "mV/km"
        self.m_units = "nT"
        self.name = name if name else "xlsx"
        self.import_diagnostics = diagnostics or {}

        first_series = next(iter(channel_series.values()))
        self.datetime_index = pd.DatetimeIndex(first_series.index)
        self.npts = len(self.datetime_index)
        self.NCh = len(channel_series)
        self.chid = list(channel_series.keys())

        self.start_time = self.datetime_index[0]
        self.end_time = self.start_time + self.dt * (self.npts - 1)
        self.zero_time = pd.Timestamp(
            year=self.start_time.year,
            month=1,
            day=1,
            tz=self.start_time.tz,
        )

        data_vars: dict[str, xr.DataArray] = {}
        for ch_name, series in channel_series.items():
            aligned = series.reindex(self.datetime_index)
            data_vars[ch_name] = xr.DataArray(
                aligned.to_numpy(dtype=np.float64, copy=False),
                dims=("time",),
                coords={"time": self.datetime_index},
            )
        self.data = xr.Dataset(data_vars, coords={"time": self.datetime_index})
        self.kp_data = None

    def restore_data(self, ch: str):
        pass
