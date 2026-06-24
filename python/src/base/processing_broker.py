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
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from PySide6.QtCore import QObject, Signal, Slot, QMutex, QMutexLocker

from base.data_manager import DataManager
from utils.series import dti_to_numpy

logger = logging.getLogger(__name__)


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
                    'latitude': getattr(edata, 'latitude', None),
                    'longitude': getattr(edata, 'longitude', None),
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
                'latitude': getattr(em_data, 'latitude', None),
                'longitude': getattr(em_data, 'longitude', None),
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

        不持锁遍历匹配，逐通道修改并实时 emit data_changed 驱动进度条。
        """
        ref_npts: int = params["npts"]
        ref_start: float = params["start_time"]
        ref_dt: float = params["dt"]
        indices: np.ndarray = params["deleted_indices"]
        source_id: int = params.get("source_data_id", -1)

        # ── Phase 1: 收集匹配的数据集 + 通道 ──────────────
        matched: list[tuple[int, str, np.ndarray]] = []  # (data_id, ch, vals_copy)
        locker = QMutexLocker(self._mutex)
        for data_id, em_data in list(self._data_manager.items()):
            if data_id == source_id:
                continue
            if em_data.npts != ref_npts:
                continue
            if abs(em_data.start_time.timestamp() - ref_start) > ref_dt:
                continue
            if abs(em_data.dt.total_seconds() - ref_dt) > 1e-9:
                continue
            for ch in em_data.channel_names():
                da = em_data.data[ch]
                if len(da.values) != ref_npts:
                    continue
                matched.append((data_id, ch, da.values.copy()))
        locker.unlock()

        if not matched:
            return

        # ── Phase 2: 逐通道修改 + 写回 + 发信号 ───────────
        synced = set()
        for data_id, ch, vals in matched:
            vals[indices] = np.nan
            em = self._safe_get(data_id)
            if em is not None:
                em.data[ch] = xr.DataArray(
                    vals, dims=em.data[ch].dims, coords=em.data[ch].coords,
                )
            self.data_changed.emit(data_id, ch, vals)
            synced.add(data_id)

        logger.info("Broker: synced region deletion to %d datasets (%d channels)",
                     len(synced), len(matched))

    # ─── 内部 ────────────────────────────────────────────────────

    def _safe_get(self, data_id: int):
        locker = QMutexLocker(self._mutex)
        return self._data_manager.get(data_id)
