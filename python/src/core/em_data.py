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

    # ── 就地预处理（数据管理器级操作） ──────────────────────────
    # 这些方法绕过 set_channel_series 的长度校验，一次性原子更新
    # xr.Dataset 与 datetime_index / dt / npts / 起止时间。调用方
    # （UI 层）负责在成功后经 DataManager.notify_changed 通知刷新。

    def set_metadata(
        self,
        *,
        name: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        elevation: Optional[float] = None,
        e_units: Optional[str] = None,
        m_units: Optional[str] = None,
    ) -> None:
        """更新台站标量元数据。参数为 None 表示保持原值。

        非结构性变更：不触碰数据与时间轴。
        """
        if name is not None:
            self.name = name
        if latitude is not None:
            self.latitude = float(latitude)
        if longitude is not None:
            self.longitude = float(longitude)
        if elevation is not None:
            self.elevation = float(elevation)
        if e_units is not None:
            self.e_units = e_units
        if m_units is not None:
            self.m_units = m_units

    def trim_time_range(
        self,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> int:
        """裁剪时间范围到 [start, end]（含端点），对齐到最近采样点。

        start / end 为 None 时保留对应一侧边界。返回新 npts。
        结果为空时抛 ValueError。
        """
        if self.npts == 0:
            raise ValueError("数据集为空，无法裁剪")

        idx = self.datetime_index
        i0 = 0 if start is None else int(idx.searchsorted(start, side="left"))
        i1 = self.npts if end is None else int(idx.searchsorted(end, side="right"))

        if i1 - i0 <= 0:
            raise ValueError(
                f"裁剪范围 [{start}, {end}] 内没有采样点 "
                f"(数据范围: {self.start_time}..{self.end_time})"
            )

        # 原子更新：Dataset 切片 + 时间轴 + 边界
        self.data = self.data.isel(time=slice(i0, i1))
        self.datetime_index = self.datetime_index[i0:i1]
        self.npts = i1 - i0
        self.start_time = self.datetime_index[0]
        self.end_time = self.datetime_index[-1]
        # dt 不变

        self._trim_kp(self.start_time, self.end_time)
        return self.npts

    def _trim_kp(self, new_start: pd.Timestamp, new_end: pd.Timestamp) -> None:
        """把 kp_data 切片到新的时间范围（None 安全）。"""
        kp = getattr(self, "kp_data", None)
        if kp is None or len(kp) == 0 or "Kp_datetime" not in kp.columns:
            return
        col = kp["Kp_datetime"]
        mask = (col >= new_start) & (col <= new_end)
        self.kp_data = kp.loc[mask].reset_index(drop=True)

    def decimate(self, factor: int) -> int:
        """整数倍抗混叠降采样（就地，不可逆）。返回新 npts。

        逐通道做 NaN 感知的零相位滤波抽取（见
        ``utils.signal_decimate``），新时间轴取原索引 [::factor]，
        dt 乘以 factor。Kp 数据不变（3 小时周期与 EM 采样率无关）。
        """
        if not isinstance(factor, int) or isinstance(factor, bool) or factor < 2:
            raise ValueError(f"降采样因子必须是 >= 2 的整数，得到 {factor!r}")
        if self.npts < factor:
            raise ValueError(f"数据点数 ({self.npts}) 小于降采样因子 ({factor})")

        from utils.signal_decimate import decimate_nan_aware, decimated_length

        new_npts = decimated_length(self.npts, factor)
        new_arrays: dict[str, np.ndarray] = {
            ch: decimate_nan_aware(self.data[ch].to_numpy(), factor, new_npts)
            for ch in self.channel_names()
        }
        new_index = self.datetime_index[::factor]

        # 原子更新
        self.data = xr.Dataset(
            {ch: (("time",), arr) for ch, arr in new_arrays.items()},
            coords={"time": new_index},
        )
        self.datetime_index = new_index
        self.npts = new_npts
        self.dt = self.dt * factor
        self.start_time = new_index[0]
        self.end_time = new_index[-1]
        return self.npts

    def rename_channel(self, old_name: str, new_name: str) -> None:
        """重命名通道，保持通道顺序。重名抛 ValueError。"""
        if not self.has_channel(old_name):
            raise KeyError(f"通道 {old_name!r} 不存在")
        if not new_name:
            raise ValueError("新通道名不能为空")
        if new_name == old_name:
            return
        if new_name in self.data.data_vars:
            raise ValueError(f"通道 {new_name!r} 已存在")

        # 重建 Dataset 以保持该通道在序列中的位置
        new_vars = {
            (new_name if ch == old_name else ch): self.data[ch]
            for ch in self.channel_names()
        }
        self.data = xr.Dataset(new_vars, coords={"time": self.datetime_index})

    def remove_channel(self, channel_name: str) -> None:
        """删除通道。仅剩一个通道时拒绝删除。"""
        if not self.has_channel(channel_name):
            raise KeyError(f"通道 {channel_name!r} 不存在")
        if self.NCh <= 1:
            raise ValueError("不能删除最后一个通道")
        self.data = self.data.drop_vars(channel_name)

    def first_channel_name(self) -> Optional[str]:
        names = self.channel_names()
        if not names:
            return None
        return names[0]

    @abstractmethod
    def restore_data(self, ch: str):
        pass
