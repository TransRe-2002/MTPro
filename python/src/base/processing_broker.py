"""数据处理中介（ProcessingBroker）。

跑在后台 QThread 里，统一管理所有数据处理工具对 DataManager 的读写。
工具不再直接持有 DataManager 或 EMData 引用，所有数据操作通过 Broker 完成。

信号槽链路::

    工具._confirm_and_close()
        → broker.apply_result(data_id, channel, result)
            → (后台线程) 写入 DataManager
            → result_applied 信号 → 工具 close
            → data_changed 信号 → UI 刷新
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from PySide6.QtCore import QObject, Signal, Slot, QMutex, QMutexLocker

from base.data_manager import DataManager
from core.region_sync import apply_nan_intervals, compute_time_range_intervals
from utils.series import dti_to_numpy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StationMetadata:
    name: str
    latitude: Optional[float]
    longitude: Optional[float]
    elevation: Optional[float]


class ProcessingBroker(QObject):
    """数据处理中介。"""

    # data_id, channel, data_package (dict)
    data_ready = Signal(int, str, object)
    # data_id, channel
    result_applied = Signal(int, str)
    data_changed = Signal(int, str, object)   # data_id, channel, result_array

    def __init__(self, data_manager: DataManager, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._data_manager = data_manager
        self._mutex = QMutex()

    # ─── 槽：工具调用，跑在后台线程 ──────────────────────────────

    @Slot(int, str)
    def request_data(self, data_id: int, channel: str) -> None:
        """工具请求数据。返回包含 x/y/元信息/全通道数据的 dict::

            {
                'x': np.ndarray,
                'y': np.ndarray,
                'channel_names': list[str],
                'all_y': dict[str, np.ndarray],   # 所有通道原始数据
                'npts': int,
                'start_time': float,
                'dt': float,
                'kp_data': pd.DataFrame | None,
            }
        """
        em_data = self._safe_get(data_id)
        if em_data is None:
            logger.warning("Broker: data_id=%d not found", data_id)
            return

        channel_names = em_data.channel_names()

        if channel == "":
            # 台站级请求：返回所有通道数据 + 全部已加载台站列表
            x = dti_to_numpy(
                em_data.get_channel_series(channel_names[0]).index
            ) if channel_names else np.array([])
            stations = []
            for sid, edata in self._data_manager.items():
                chs = edata.channel_names()
                stations.append({
                    'id': sid,
                    'name': edata.name,
                    'has_magnetic': any(c.startswith(('H', 'B')) for c in chs),
                    'has_electric': any(c.startswith('E') for c in chs),
                    'latitude': edata.latitude,
                    'longitude': edata.longitude,
                    'elevation': edata.elevation,
                })
            pkg = {
                'x': x,
                'y': np.array([]),
                'channel_names': channel_names,
                'all_y': {
                    ch: em_data.get_channel_series(ch).to_numpy(dtype=np.float64, copy=True)
                    for ch in channel_names
                },
                'npts': em_data.npts,
                'start_time': em_data.start_time.timestamp(),
                'dt': em_data.dt.total_seconds(),
                'kp_data': em_data.kp_data,
                'stations': stations,
                'datetime_index': em_data.datetime_index,
                'name': em_data.name,
                'latitude': em_data.latitude,
                'longitude': em_data.longitude,
                'elevation': em_data.elevation,
                'station_metadata': StationMetadata(
                    name=em_data.name,
                    latitude=em_data.latitude,
                    longitude=em_data.longitude,
                    elevation=em_data.elevation,
                ),
            }
        else:
            series = em_data.get_channel_series(channel)
            x = dti_to_numpy(series.index)
            y = series.to_numpy(dtype=np.float64)
            pkg = {
                'x': x,
                'y': y,
                'channel_names': channel_names,
                'all_y': {
                    ch: em_data.get_channel_series(ch).to_numpy(dtype=np.float64, copy=True)
                    for ch in channel_names
                },
                'npts': em_data.npts,
                'start_time': em_data.start_time.timestamp(),
                'dt': em_data.dt.total_seconds(),
                'datetime_index': em_data.datetime_index,
                'kp_data': em_data.kp_data,
            }
        self.data_ready.emit(data_id, channel, pkg)

    @Slot(int, str, object)
    def apply_result(self, data_id: int, channel: str, result: np.ndarray) -> None:
        """应用处理结果到 DataManager。"""
        em_data = self._safe_get(data_id)
        if em_data is None:
            logger.warning("Broker: data_id=%d not found for apply_result", data_id)
            return

        existing = em_data.get_channel_series(channel)
        if len(existing) == len(result):
            em_data.set_channel_series(channel, pd.Series(
                result, index=existing.index, name=existing.name,
            ))
        else:
            em_data.set_channel_data(channel, result)

        logger.info("Broker: applied result to #%d %s (%d points)", data_id, channel, len(result))
        self.result_applied.emit(data_id, channel)
        self.data_changed.emit(data_id, channel, result)

    @Slot(int, str, object, object)
    def apply_result_all_channels(
        self, data_id: int, skip_channel: str, result: np.ndarray,
    ) -> None:
        """跨通道同步：将当前通道的 NaN 位置同步到数据集其余通道。"""
        em_data = self._safe_get(data_id)
        if em_data is None:
            return

        for ch in em_data.channel_names():
            if ch == skip_channel:
                continue
            da = em_data.data[ch]
            vals = da.values.copy()
            vals[np.isnan(result)] = np.nan
            em_data.data[ch] = xr.DataArray(
                vals, dims=da.dims, coords=da.coords,
            )
            self.data_changed.emit(data_id, ch, vals)
        logger.info("Broker: synced NaN to %d other channels in #%d",
                     len(em_data.channel_names()) - 1, data_id)

    @Slot(object)
    def apply_region_to_all(self, params: dict) -> None:
        """RemoveRegion 跨数据集同步。

        接收时间范围列表，对 DataManager 中每个数据集独立计算索引并置 NaN。
        不再依赖 npts/dt/startTime 匹配——每个 EMData 用自身参数。
        """
        time_ranges: list[tuple[float, float]] = params["time_ranges"]
        source_id: int = params.get("source_data_id", -1)

        synced = 0
        locker = QMutexLocker(self._mutex)
        for data_id, em_data in list(self._data_manager.items()):
            if data_id == source_id:
                continue
            npts = em_data.npts
            start_ts = em_data.start_time.timestamp()
            dt_sec = em_data.dt.total_seconds()

            intervals = compute_time_range_intervals(time_ranges, npts, start_ts, dt_sec)
            if not intervals:
                continue

            for ch in em_data.channel_names():
                da = em_data.data[ch]
                if len(da.values) != npts:
                    continue
                vals = apply_nan_intervals(da.values, intervals)
                em_data.data[ch] = xr.DataArray(
                    vals, dims=da.dims, coords=da.coords,
                )
                self.data_changed.emit(data_id, ch, vals)
            synced += 1
        locker.unlock()

        if synced > 0:
            logger.info("Broker: synced region deletion to %d datasets (time ranges: %s)",
                         synced, time_ranges)

    # ─── 内部 ────────────────────────────────────────────────────

    def _safe_get(self, data_id: int):
        locker = QMutexLocker(self._mutex)
        return self._data_manager.get(data_id)
