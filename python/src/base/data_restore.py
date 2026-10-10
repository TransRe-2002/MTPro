"""Restore MAT-backed data from disk, then notify every in-memory consumer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from base.data_manager import DataManager
from core.mat_data import MatEMData
from io_utils.file_ops import load_mat


STATION_FIELDS = ("name", "latitude", "longitude", "elevation")


def load_restore_source(data: MatEMData) -> MatEMData:
    if not isinstance(data, MatEMData):
        raise ValueError("目前仅支持从 MAT 源文件还原。")
    path = Path(data.path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"找不到源文件：{path}")
    return load_mat(path, tz=data.timezone or "Asia/Shanghai")


def station_metadata_differs(data: MatEMData, source: MatEMData) -> bool:
    for field in STATION_FIELDS:
        current, original = getattr(data, field), getattr(source, field)
        if pd.isna(current) and pd.isna(original):
            continue
        if current != original:
            return True
    return False


def restore_mat_data(
    manager: DataManager,
    key: int,
    source: MatEMData,
    *,
    channel: str | None = None,
    keep_station_metadata: bool = False,
) -> None:
    data = manager.get(key)
    if not isinstance(data, MatEMData) or not isinstance(source, MatEMData):
        raise ValueError("目前仅支持从 MAT 源文件还原。")
    if source.npts <= 0 or not source.channel_names() or source.dt <= pd.Timedelta(0):
        raise ValueError("源文件缺少有效数据或采样间隔，无法还原。")
    if channel is None:
        data.restore_dataset(source, keep_station_metadata=keep_station_metadata)
        manager.discard_channel_processing_baselines(key)
    else:
        data.restore_data(channel, source=source)
        manager.discard_channel_processing_baseline(key, channel)
    manager.notify_restored(key, channel)
