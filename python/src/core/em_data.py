from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import xarray as xr


class EMData(ABC):
    name: str
    path: str
    zero_time: Optional[pd.Timestamp]
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    dt: pd.Timedelta
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation: Optional[float] = None
    e_units: Optional[str] = None
    m_units: Optional[str] = None

    timezone: Optional[str] = None  # 采集站原始时区，用于保存时转换
    kp_data: Optional[pd.DataFrame]
    datetime_index: pd.DatetimeIndex
    data: xr.Dataset

    def __init__(self, path: str):
        self.path = path
        self._chid_shadow: list[str] = []
        self._npts_shadow: int = 0
        self._NCh_shadow: int = 0
        self.data = xr.Dataset(coords={"time": pd.DatetimeIndex([])})

    @property
    def chid(self) -> list[str]:
        if self.data.data_vars:
            return list(self.data.data_vars.keys())
        return list(self._chid_shadow)

    @chid.setter
    def chid(self, value: list[str]) -> None:
        self._chid_shadow = list(value)

    @property
    def npts(self) -> int:
        if "time" in self.data.sizes and int(self.data.sizes["time"]) > 0:
            return int(self.data.sizes["time"])
        return int(self._npts_shadow)

    @npts.setter
    def npts(self, value: int) -> None:
        self._npts_shadow = int(value)

    @property
    def NCh(self) -> int:
        if self.data.data_vars:
            return len(self.data.data_vars)
        return int(self._NCh_shadow)

    @NCh.setter
    def NCh(self, value: int) -> None:
        self._NCh_shadow = int(value)

    def channel_names(self) -> list[str]:
        return self.chid

    def has_channel(self, channel_name: str) -> bool:
        return channel_name in self.data.data_vars

    def iter_channels(self) -> Iterator[tuple[str, xr.DataArray]]:
        for channel_name in self.channel_names():
            yield channel_name, self.data[channel_name]

    def get_channel_array(self, channel_name: str) -> xr.DataArray:
        if not self.has_channel(channel_name):
            raise KeyError(channel_name)
        return self.data[channel_name]

    def get_channel_series(self, channel_name: str) -> pd.Series:
        channel = self.get_channel_array(channel_name)
        values = channel.to_numpy()
        if "time" in channel.indexes:
            index = pd.DatetimeIndex(channel.indexes["time"])
        else:
            index = self.datetime_index
        return pd.Series(values, index=index, name=channel_name)

    def set_channel_series(self, channel_name: str, series: pd.Series) -> None:
        normalized = series.copy()
        if not isinstance(normalized.index, pd.DatetimeIndex) and len(normalized) == len(self.datetime_index):
            normalized.index = self.datetime_index

        if self.has_channel(channel_name):
            expected = self.get_channel_series(channel_name)
            if len(normalized) != len(expected):
                raise ValueError(
                    f"channel {channel_name} length mismatch: "
                    f"expected {len(expected)}, got {len(normalized)}"
                )
            if not normalized.index.equals(expected.index):
                normalized = normalized.reindex(expected.index)
        elif len(normalized) != len(self.datetime_index):
            raise ValueError(
                f"channel {channel_name} length mismatch: "
                f"expected {len(self.datetime_index)}, got {len(normalized)}"
            )
        elif not normalized.index.equals(self.datetime_index):
            normalized = normalized.reindex(self.datetime_index)

        self.data[channel_name] = xr.DataArray(
            normalized.to_numpy(dtype=np.float64, copy=False),
            dims=("time",),
            coords={"time": normalized.index},
        )

    def set_channel_data(
        self,
        channel_name: str,
        values: np.ndarray | list[float],
        index: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        if index is None:
            index = self.datetime_index
        series = pd.Series(np.asarray(values, dtype=np.float64), index=index, name=channel_name)
        self.set_channel_series(channel_name, series)

    def channel_time_range(self, channel_name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        series = self.get_channel_series(channel_name)
        if series.empty:
            raise ValueError(f"channel {channel_name} is empty")
        return pd.Timestamp(series.index[0]), pd.Timestamp(series.index[-1])

    def first_channel_name(self) -> Optional[str]:
        names = self.channel_names()
        if not names:
            return None
        return names[0]

    @abstractmethod
    def restore_data(self, ch: str):
        pass
