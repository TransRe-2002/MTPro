from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd
import pytz

from core.txt_data import TxtEMData
from io_utils.em_io import EMDataLoader

try:
    import rust_lib
except ImportError:
    rust_lib = None  # type: ignore


TXT_CHANNELS = ("Hx", "Hy", "Hz", "Ex1", "Ey1", "Ex2", "Ey2")


class TxtLoader(EMDataLoader):
    """加载 MT Portable 同格式的空白分隔 TXT 时序文件。"""

    @staticmethod
    def load(
        path: str,
        tz: str,
        name: str | None = None,
        dt_override: float | None = None,
    ) -> TxtEMData:
        path_obj = Path(path).expanduser().resolve(strict=False)
        if not path_obj.is_file():
            raise FileNotFoundError(str(path_obj))
        if rust_lib is None or not hasattr(rust_lib, "read_mt_txt_rs"):
            raise RuntimeError(
                "当前 rust_lib 未包含 TXT 解析器，请重新构建并安装 MTPro Rust 扩展"
            )

        # 先校验，避免解析完整文件后才报告无效时区。
        pytz.timezone(tz)
        raw = rust_lib.read_mt_txt_rs(str(path_obj))
        row_count = int(raw["row_count"])
        if row_count <= 0:
            raise ValueError("TXT 文件没有有效数据行")

        detected_dt = raw.get("dt_seconds")
        dt_seconds = float(dt_override) if dt_override is not None else detected_dt
        if dt_seconds is None or not np.isfinite(dt_seconds) or dt_seconds <= 0:
            raise ValueError("无法从单行数据推断采样间隔，请指定 dt_override")

        local_start = pd.Timestamp(raw["start_time"]).tz_localize(
            tz,
            ambiguous="raise",
            nonexistent="raise",
        )
        time_index = pd.date_range(
            start=local_start,
            periods=row_count,
            freq=pd.Timedelta(seconds=dt_seconds),
        ).tz_convert("UTC")

        channel_series: dict[str, pd.Series] = {}
        for channel_name in TXT_CHANNELS:
            values = np.asarray(raw[channel_name], dtype=np.float64)
            if values.shape != (row_count,):
                raise ValueError(
                    f"{channel_name} 点数为 {values.size}，预期为 {row_count}"
                )
            channel_series[channel_name] = pd.Series(
                values,
                index=time_index,
                name=channel_name,
                copy=False,
            )

        diagnostics = {
            "detected_dt_seconds": detected_dt,
            "gps_samples": int(raw.get("gps_samples", 0)),
            "source_start_time": str(raw["start_time"]),
            "source_end_time": str(raw["end_time"]),
        }
        return TxtEMData(
            channel_series=channel_series,
            timezone=tz,
            dt=pd.Timedelta(seconds=dt_seconds),
            path=str(path_obj),
            lat=_optional_float(raw.get("latitude")),
            lon=_optional_float(raw.get("longitude")),
            name=name or path_obj.stem,
            diagnostics=diagnostics,
        )

    @staticmethod
    def load_group(
        paths: Iterable[str],
        tz: str,
        name: str | None = None,
    ) -> TxtEMData:
        """把一次选择的多份 TXT 合并为一个时间轴和一个数据集。"""
        path_list = [str(Path(path).expanduser().resolve(strict=False)) for path in paths]
        if not path_list:
            raise ValueError("没有选择 TXT 文件")
        if len(set(path_list)) != len(path_list):
            raise ValueError("TXT 文件列表中包含重复路径")

        loaded = [TxtLoader.load(path, tz=tz) for path in path_list]
        group_dt = loaded[0].dt
        for path, data in zip(path_list[1:], loaded[1:]):
            if data.dt != group_dt:
                raise ValueError(
                    f"采样间隔不一致：{Path(path).name} 为 {data.dt.total_seconds():g} s，"
                    f"首个文件为 {group_dt.total_seconds():g} s"
                )

        start = min(data.datetime_index[0] for data in loaded)
        end = max(data.datetime_index[-1] for data in loaded)
        offset_ns = (loaded[0].datetime_index[0] - start).value
        if offset_ns % group_dt.value:
            raise ValueError(
                f"文件时间没有对齐到 {group_dt.total_seconds():g} s 的公共采样网格"
            )
        target_index = pd.date_range(start=start, end=end, freq=group_dt)
        if target_index.empty:
            raise ValueError("合并后的时间轴为空")

        merged_channels: dict[str, pd.Series] = {}
        duplicate_count = 0
        for channel_name in TXT_CHANNELS:
            combined = pd.concat(
                [data.get_channel_series(channel_name) for data in loaded]
            ).sort_index(kind="stable")
            duplicates = combined[combined.index.duplicated(keep=False)]
            if not duplicates.empty:
                for timestamp, values in duplicates.groupby(level=0, sort=False):
                    if not np.all(values.to_numpy() == values.iloc[0]):
                        raise ValueError(
                            f"所选 TXT 文件在通道 {channel_name}、时间 {timestamp} "
                            "有数值冲突"
                        )
                    duplicate_count += len(values) - 1
                combined = combined[~combined.index.duplicated(keep="first")]

            # DatetimeIndex.asi8 follows the index's internal resolution
            # (pandas may use microseconds); Timedelta.value is always ns.
            # Normalize both sides before comparing the sample grid.
            combined_ns = combined.index.as_unit("ns").asi8
            origin_ns = target_index[0].as_unit("ns").value
            off_grid = (combined_ns - origin_ns) % group_dt.value
            if np.any(off_grid):
                raise ValueError(
                    f"通道 {channel_name} 中有时间点未对齐到 "
                    f"{group_dt.total_seconds():g} s 的公共采样网格"
                )
            merged_channels[channel_name] = combined.reindex(target_index)

        located = [
            (data.latitude, data.longitude, int(data.import_diagnostics.get("gps_samples", 0)))
            for data in loaded
            if data.latitude is not None and data.longitude is not None
        ]
        if located:
            weights = np.asarray([max(1, item[2]) for item in located], dtype=np.float64)
            lat = float(np.average([item[0] for item in located], weights=weights))
            lon = float(np.average([item[1] for item in located], weights=weights))
        else:
            lat = lon = None

        nan_samples = int(sum(series.isna().sum() for series in merged_channels.values()))
        diagnostics = {
            "files": path_list,
            "source_count": len(path_list),
            "duplicate_sample_count": duplicate_count // len(TXT_CHANNELS),
            "gap_sample_count": nan_samples // len(TXT_CHANNELS),
            "nan_channel_values": nan_samples,
            "dt_seconds": group_dt.total_seconds(),
        }
        return TxtEMData(
            channel_series=merged_channels,
            timezone=tz,
            dt=group_dt,
            path=path_list[0],
            lat=lat,
            lon=lon,
            name=name or Path(path_list[0]).stem,
            diagnostics=diagnostics,
        )


def _optional_float(value) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None
