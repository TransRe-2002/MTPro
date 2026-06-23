"""二进制文件 (.dat / .inf) 的加载与保存。

支持大端序 (big-endian) 固定记录长度的二进制格式：
  - .dat:  7×uint16 (时间) + 8×float32 (通道数据) = 46 字节/记录
  - .inf:  7×uint16 (时间) + 3×float32 (经纬度海拔) = 26 字节/记录
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

from core.bin_data import BinEMData
from core.em_data import EMData
from io_utils.em_io import EMDataLoader, EMDataSaver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 结构化 dtype 定义（大端序）
# ---------------------------------------------------------------------------

# .dat 文件: 每条记录 46 字节
DAT_DTYPE = np.dtype([
    ("year",        ">u2"),
    ("month",       ">u2"),
    ("day",         ">u2"),
    ("hour",        ">u2"),
    ("minute",      ">u2"),
    ("second",      ">u2"),
    ("millisecond", ">u2"),
    ("ch1",         ">f4"),
    ("ch2",         ">f4"),
    ("ch3",         ">f4"),
    ("ch4",         ">f4"),
    ("ch5",         ">f4"),
    ("ch6",         ">f4"),
    ("ch7",         ">f4"),
    ("ch8",         ">f4"),
])

DAT_CH_FIELDS = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8"]

# GPS 坐标除数（原始值 ÷100 = 真实度）
GPS_DIVISOR = 100.0

# .inf 文件: 每条记录 26 字节
INF_DTYPE = np.dtype([
    ("year",        ">u2"),
    ("month",       ">u2"),
    ("day",         ">u2"),
    ("hour",        ">u2"),
    ("minute",      ">u2"),
    ("second",      ">u2"),
    ("millisecond", ">u2"),
    ("lat",         ">f4"),
    ("lon",         ">f4"),
    ("alt",         ">f4"),
])

# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _build_datetime_index(raw: np.ndarray, tz: str) -> pd.DatetimeIndex:
    """从结构化数组的时间字段构建带时区的 DatetimeIndex（内部为 UTC）。"""
    naive = pd.DatetimeIndex(pd.to_datetime({
        "year":   raw["year"].astype(np.int32),
        "month":  raw["month"].astype(np.int32),
        "day":    raw["day"].astype(np.int32),
        "hour":   raw["hour"].astype(np.int32),
        "minute": raw["minute"].astype(np.int32),
        "second": raw["second"].astype(np.int32),
    }))
    # 加上毫秒偏移
    naive = naive + pd.to_timedelta(raw["millisecond"].astype(np.int64), unit="ms")

    tz_obj = pytz.timezone(tz)
    return naive.tz_localize(tz_obj).tz_convert("UTC")


def _infer_dt(raw: np.ndarray) -> pd.Timedelta:
    """从原始时间字段推算采样间隔。

    用 second/minute/hour/day 字段计算总秒数差值的众数，
    避免被不可靠的毫秒字段误导。
    """
    if len(raw) < 2:
        return pd.Timedelta(seconds=1.0)
    total_sec = (
        raw["second"].astype(np.int64)
        + raw["minute"].astype(np.int64) * 60
        + raw["hour"].astype(np.int64) * 3600
        + raw["day"].astype(np.int64) * 86400
    )
    diffs = np.diff(total_sec)
    # 修正跨天回绕
    diffs = np.where(diffs < -3600, diffs + 86400, diffs)
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return pd.Timedelta(seconds=1.0)
    mode_sec = float(np.bincount(positive).argmax())
    if mode_sec == 0:
        mode_sec = float(np.median(positive))
    return pd.Timedelta(seconds=mode_sec)


def _is_empty_channel(name: str) -> bool:
    """判断通道名是否表示空通道。"""
    stripped = name.strip()
    return not stripped or stripped.lower() == "empty"


# ---------------------------------------------------------------------------
# 快速预览（不加载全部数据）
# ---------------------------------------------------------------------------


def _raw_to_naive(raw: np.ndarray) -> pd.DatetimeIndex:
    """结构化数组的时间字段 → naive DatetimeIndex。"""
    idx = pd.DatetimeIndex(pd.to_datetime({
        "year":   raw["year"].astype(np.int32),
        "month":  raw["month"].astype(np.int32),
        "day":    raw["day"].astype(np.int32),
        "hour":   raw["hour"].astype(np.int32),
        "minute": raw["minute"].astype(np.int32),
        "second": raw["second"].astype(np.int32),
    }))
    return idx + pd.to_timedelta(raw["millisecond"].astype(np.int64), unit="ms")


def read_bin_header(path: str) -> dict:
    """读取 .dat 文件摘要，用于向导预览。

    读前 200 条估算 dt，end_time 由公式计算（忽略记录间隙）。

    Returns
    -------
    dict with keys: npts, file_size_mb, start_time, end_time, dt_sec
    """
    file_size = os.path.getsize(path)
    record_size = 46
    npts = file_size // record_size

    if npts == 0:
        return {
            "npts": 0, "file_size_mb": 0.0,
            "start_time": None, "end_time": None, "dt_sec": None,
        }

    raw_head = np.fromfile(path, dtype=DAT_DTYPE, count=min(200, npts))
    naive_head = _raw_to_naive(raw_head)
    dt_est = _infer_dt(raw_head)
    start_est = naive_head[0]
    end_est = start_est + dt_est * (npts - 1)

    return {
        "npts": npts,
        "file_size_mb": file_size / (1024 * 1024),
        "start_time": start_est,
        "end_time": end_est,
        "dt_sec": dt_est.total_seconds(),
    }


def read_gps_header(path: str) -> dict:
    """读取 .inf 文件摘要信息。

    GPS 经纬度已自动 ÷100 转换，取第一条有效记录（lat≠0）。

    Returns
    -------
    dict with keys: npts, lat, lon, alt, start_time, end_time
    """
    file_size = os.path.getsize(path)
    record_size = 26
    npts = file_size // record_size

    if npts == 0:
        return {
            "npts": 0, "lat": None, "lon": None, "alt": None,
            "start_time": None, "end_time": None,
        }

    raw = np.fromfile(path, dtype=INF_DTYPE)

    # 取第一条有效记录（跳过全零头部）
    mask = raw["lat"] != 0
    if mask.any():
        first_valid = raw[mask][0]
        lat = float(first_valid["lat"]) / GPS_DIVISOR
        lon = float(first_valid["lon"]) / GPS_DIVISOR
        alt = float(first_valid["alt"])
    else:
        first_valid = raw[0]
        lat = float(first_valid["lat"]) / GPS_DIVISOR
        lon = float(first_valid["lon"]) / GPS_DIVISOR
        alt = float(first_valid["alt"])

    naive_idx = pd.DatetimeIndex(pd.to_datetime({
        "year":   raw["year"].astype(np.int32),
        "month":  raw["month"].astype(np.int32),
        "day":    raw["day"].astype(np.int32),
        "hour":   raw["hour"].astype(np.int32),
        "minute": raw["minute"].astype(np.int32),
        "second": raw["second"].astype(np.int32),
    }))
    naive_idx = naive_idx + pd.to_timedelta(
        raw["millisecond"].astype(np.int64), unit="ms"
    )

    return {
        "npts": npts,
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "start_time": naive_idx[0],
        "end_time": naive_idx[-1],
    }


# ---------------------------------------------------------------------------
# 加载器 / 保存器
# ---------------------------------------------------------------------------


class BinLoader(EMDataLoader):
    """从二进制 .dat 文件加载 EM 数据，可选关联 .inf GPS 文件。"""

    @staticmethod
    def load(
        path: str,
        tz: str,
        channel_configs: list[dict] | None = None,
        gps_path: str | None = None,
        ex_distance: float | None = None,
        ey_distance: float | None = None,
        manual_lat: float | None = None,
        manual_lon: float | None = None,
        manual_alt: float | None = None,
        name: str | None = None,
        dt_override: float | None = None,
    ) -> BinEMData:
        """加载 .dat 文件为 BinEMData。

        Parameters
        ----------
        path : str
            .dat 文件路径。
        tz : str
            采集站原始时区（如 'Asia/Shanghai'）。
        channel_configs : list[dict] | None
            8 个通道的配置列表。每项 dict 含:
            - name (str): 通道名称，空/"Empty" 则跳过
            - scale (float): 缩放因子
            - electrode_pair (str): "Ex" / "Ey" / ""
            若为 None 则使用 Ch1~Ch8, scale=1.0, 不绑定电极对。
        gps_path : str | None
            同名 .inf GPS 文件路径（可选）。
        ex_distance : float | None
            Ex 电极偶极距 (m)。
        ey_distance : float | None
            Ey 电极偶极距 (m)。
        manual_lat / manual_lon / manual_alt : float | None
            手动指定的经纬度海拔（覆盖 inf 文件读取值）。

        Returns
        -------
        BinEMData
        """
        if channel_configs is None:
            channel_configs = [
                {"name": f"Ch{i}", "scale": 1.0, "electrode_pair": ""}
                for i in range(1, 9)
            ]
        elif len(channel_configs) != 8:
            raise ValueError(
                f"channel_configs 必须有 8 个元素，实际提供了 {len(channel_configs)} 个"
            )

        # 读取全部 .dat 数据
        raw = np.fromfile(path, dtype=DAT_DTYPE)
        if len(raw) == 0:
            raise ValueError(f"文件为空或格式不正确: {path}")

        npts = len(raw)
        dt = pd.Timedelta(seconds=dt_override) if dt_override else _infer_dt(raw)

        # 只取第一条记录的时间作为 start_time，用 pd.date_range 生成干净的时间轴
        # 避免文件中不可靠的毫秒字段污染 DatetimeIndex
        raw_first = raw[0:1]
        start_time = _build_datetime_index(raw_first, tz)[0]
        ts_index = pd.date_range(start=start_time, periods=npts, freq=dt)

        # 构建通道序列（应用缩放因子，跳过空通道）
        # 电场偶极距转换系数: mV / (distance_m / 1000) = mV/km
        ex_factor = 1000.0 / ex_distance if ex_distance and ex_distance > 0 else 1.0
        ey_factor = 1000.0 / ey_distance if ey_distance and ey_distance > 0 else 1.0

        channel_series: dict[str, pd.Series] = {}

        for i, cfg in enumerate(channel_configs):
            ch_name = cfg["name"].strip()
            if _is_empty_channel(ch_name):
                continue

            scale = float(cfg.get("scale", 1.0))
            pair = cfg.get("electrode_pair", "")

            field = DAT_CH_FIELDS[i]
            scaled = raw[field].astype(np.float64) * scale

            # 电场通道除以偶极距 (mV → mV/km)
            if pair == "Ex":
                scaled *= ex_factor
            elif pair == "Ey":
                scaled *= ey_factor

            channel_series[ch_name] = pd.Series(
                scaled, index=ts_index, name=ch_name,
            )

        if not channel_series:
            raise ValueError("所有通道均为空，没有可导入的数据")

        # GPS — 手动输入优先，否则读 inf 文件
        lat: float | None = manual_lat
        lon: float | None = manual_lon
        elev: float | None = manual_alt

        if lat is None and gps_path is not None and os.path.exists(gps_path):
            try:
                gps_raw = np.fromfile(gps_path, dtype=INF_DTYPE)
                if len(gps_raw) > 0:
                    mask = gps_raw["lat"] != 0
                    first = gps_raw[mask][0] if mask.any() else gps_raw[0]
                    lat = float(first["lat"]) / GPS_DIVISOR
                    lon = float(first["lon"]) / GPS_DIVISOR
                    elev = float(first["alt"])
            except Exception:
                logger.warning("读取 GPS 文件失败: %s", gps_path, exc_info=True)

        return BinEMData(
            channel_series=channel_series,
            timezone=tz,
            dt=dt,
            path=path,
            lat=lat,
            lon=lon,
            elev=elev,
            name=name,
        )


class BinSaver(EMDataSaver):
    """保存 EMData 为二进制 .dat 格式（暂未实现）。"""

    @staticmethod
    def save(data: EMData, path: str):
        raise NotImplementedError("二进制格式保存暂未实现，请使用另存为 MAT 格式")
