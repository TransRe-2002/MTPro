from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytz
import xarray as xr

from core.em_data import EMData
from utils.fetch_kp import fetch_kp
from utils.timestamp import pts_to_array

logger = logging.getLogger(__name__)


def _field(ts, *names):
    """从 structured array 取字段，支持多个别名（模糊匹配）。"""
    for n in names:
        try:
            return ts[n]
        except (ValueError, KeyError):
            continue
    raise KeyError(f"none of {names} found in tsStruct")


def _opt_field(ts, *names, default=None):
    """取可选字段，不存在时返回 default。"""
    try:
        return _field(ts, *names)
    except KeyError:
        return default


def _opt_float(ts, *names, default=np.nan):
    """取可选浮点字段，返回标量。"""
    try:
        return float(_field(ts, *names)[0, 0][0, 0])
    except (KeyError, ValueError, IndexError):
        return default


class MatEMData(EMData):
    mat_meta: dict
    user_data: list[str]

    def __init__(self, mat_meta: dict, path: str, tz: str = 'Asia/Shanghai'):
        super().__init__(path)
        self.mat_meta = mat_meta

        # ── 自动发现顶层 key（tsStruct 或 ts）──
        self._struct_key = next(
            k for k in mat_meta if not k.startswith('__')
        )
        ts = mat_meta[self._struct_key]

        # ── timeZone：文件内字段优先于用户选择 ──
        file_tz = _opt_field(ts, 'timeZone', default=None)
        if file_tz is not None and file_tz.size > 0:
            tz = str(file_tz[0, 0][0])
        self.timezone = tz

        # ── 基础属性 ──
        self.name: str = _field(ts, 'name')[0, 0][0]
        dt_val = _field(ts, 'dt')[0, 0]
        self.dt = pd.Timedelta(seconds=float(dt_val.flat[0]) if dt_val.size > 0 else 1.0)

        # npts
        npts_raw = _field(ts, 'npts')[0, 0]
        self.npts = int(npts_raw[0, 0]) if npts_raw.size == 1 else int(npts_raw[0])

        # NCh / nch（模糊匹配）
        try:
            self.NCh = int(_field(ts, 'NCh')[0, 0][0, 0])
        except KeyError:
            self.NCh = int(_field(ts, 'nch')[0, 0][0, 0])

        # ── 时间 ──
        # datevec 分量强制取整：MATLAB 文件里 datevec 可能是浮点类型，
        # 而 pd.Timestamp 的分量构造只接受整数（浮点会被当作纳秒 epoch）
        st = _field(ts, 'startTime')[0, 0][0]
        tz_obj = pytz.timezone(tz)
        self.start_time = pd.Timestamp(
            int(st[0]), int(st[1]), int(st[2]),
            int(st[3]), int(st[4]), int(st[5]), tz=tz_obj,
        ).tz_convert('UTC')

        # zeroTime 可选，无则用 startTime
        zt = _opt_field(ts, 'zeroTime', default=None)
        if zt is not None and zt[0, 0].size >= 6:
            zv = zt[0, 0][0]
            self.zero_time = pd.Timestamp(
                int(zv[0]), int(zv[1]), int(zv[2]),
                int(zv[3]), int(zv[4]), int(zv[5]), tz=tz_obj,
            ).tz_convert('UTC')
        else:
            self.zero_time = self.start_time

        # ── 通道数据 ──
        data = _field(ts, 'data')[0, 0]
        chid = _field(ts, 'chid')[0, 0][0]

        # data 方向自适应：(npts, nch) → 转置为 (NCh, npts)
        if data.ndim == 2 and data.shape[0] != self.NCh and data.shape[1] == self.NCh:
            data = data.T

        actual_npts = data.shape[1]
        if actual_npts != self.npts:
            import warnings
            warnings.warn(
                f"npts 字段 ({self.npts}) 与实际数据长度 ({actual_npts}) 不一致，"
                f"以实际数据长度为准"
            )
            self.npts = actual_npts

        self.end_time = self.start_time + self.dt * (self.npts - 1)
        self.datetime_index = pd.date_range(
            start=self.start_time, periods=self.npts, freq=self.dt,
        )

        # timeStamp（十进制年份，npts×8B）从 datetime_index 可完全重建，
        # 且保存时 mat_io 会用 _make_timestamp_ 无条件重写——加载后立即
        # 释放引用：数十个年度 5s 采样文件合计可省 ~2 GB 常驻内存。
        # dict（mcos_reader 路径）用 pop；结构化数组（scipy/v7.3 路径）
        # 的 object 字段置空以释放对 (npts,1) 数组的引用。
        try:
            if isinstance(ts, dict):
                ts.pop("timeStamp", None)
            else:
                ts["timeStamp"] = None
        except Exception:
            pass

        # ── 台站元数据（全部 Optional）──
        self.latitude = _opt_float(ts, 'latitude')
        self.longitude = _opt_float(ts, 'longitude')

        elev = _opt_field(ts, 'elevation', default=None)
        if elev is not None and elev[0, 0].size > 0:
            self.elevation = float(np.asarray(elev[0, 0]).flat[0])
        else:
            self.elevation = None

        # instr → 追加到 UserData
        instr = _opt_field(ts, 'instr', default=None)
        if instr is not None and instr[0, 0].size > 0:
            self.user_data = [str(instr[0, 0][0])]
        else:
            self.user_data = []

        # units（可选）
        units = _opt_field(ts, 'units', default=None)
        self.e_units = None
        self.m_units = None
        if units is not None and units[0, 0].size > 0:
            u = units[0, 0]
            if u.size >= 2:
                # (1, 2) cell: [m_units, e_units]
                self.m_units = u.flat[0][0] if u.flat[0].size > 0 else None
                self.e_units = u.flat[1][0] if u.flat[1].size > 0 else None
            elif u.size == 1 and len(self.chid) > 0:
                ch_name = self.chid[0]
                if ch_name.startswith('E'):
                    self.e_units = u.flat[0][0] if u.flat[0].size > 0 else None
                elif ch_name[0] in ('B', 'H'):
                    self.m_units = u.flat[0][0] if u.flat[0].size > 0 else None

        # ── 构建 xarray Dataset ──
        self.data = xr.Dataset(
            {chid[i][0]: (("time",), data[i, :].astype(np.float64, copy=False))
             for i in range(len(chid))},
            coords={"time": self.datetime_index},
        )

        self.kp_data = fetch_kp(self.start_time, self.end_time)
        self._source_channels = {ch: ch for ch in self.channel_names()}
        self._source_dt = self.dt
        self._source_time_operations = []

    def rename_channel(self, old_name: str, new_name: str) -> None:
        source_name = self._source_channels.get(old_name, old_name)
        super().rename_channel(old_name, new_name)
        self._source_channels.pop(old_name, None)
        self._source_channels[new_name] = source_name

    def remove_channel(self, channel_name: str) -> None:
        super().remove_channel(channel_name)
        self._source_channels.pop(channel_name, None)

    def trim_time_range(self, start=None, end=None) -> int:
        npts = super().trim_time_range(start, end)
        self._source_time_operations.append(("trim", self.start_time, self.end_time))
        return npts

    def decimate(self, factor: int, max_missing_fraction: float = 1.0) -> int:
        npts = super().decimate(factor, max_missing_fraction)
        self._source_time_operations.append(("decimate", factor, max_missing_fraction))
        return npts

    def source_channel_series(self, source: "MatEMData", channel: str) -> pd.Series:
        """Prepare source values on the current grid, without mutating either dataset.

        Replay structural operations, not signal processing. Repeated boxcar
        averages with missing samples are not equivalent to one larger average.
        """
        from utils.signal_decimate import decimate_nan_aware, decimated_length

        if not self.has_channel(channel):
            raise ValueError(f"当前数据中没有通道：{channel}")
        source_name = self._source_channels.get(channel, channel)
        if not source.has_channel(source_name):
            raise ValueError(f"源文件中找不到对应通道：{source_name}（当前名称：{channel}）")
        if source.dt <= pd.Timedelta(0) or self.dt < source.dt:
            raise ValueError("源文件采样率低于当前采样率，无法通过降采样还原。")
        if self.dt.value % source.dt.value:
            raise ValueError("采样间隔不是整数倍，无法使用当前降采样算法还原。")

        series = source.get_channel_series(source_name)
        dt = source.dt
        if dt == self._source_dt:
            for operation, first, second in self._source_time_operations:
                if operation == "trim":
                    if first not in series.index or second not in series.index:
                        raise ValueError("源文件时间范围已改变，无法重放裁剪；请整组还原。")
                    series = series.loc[first:second]
                else:
                    factor = first
                    series = pd.Series(
                        decimate_nan_aware(
                            series.to_numpy(), factor, decimated_length(len(series), factor),
                            max_missing_fraction=second,
                        ),
                        index=series.index[::factor],
                    )
                    dt *= factor
        elif dt != self.dt:
            # The on-disk version may have changed. Only support exact integer
            # downsampling anchored at the current first sample, never interpolation.
            if self.datetime_index[0] not in series.index:
                raise ValueError("源文件与当前时间轴无法对齐，请使用整个数据集还原。")
            series = series.loc[self.datetime_index[0]:]
            factor = int(self.dt.value // dt.value)
            series = pd.Series(
                decimate_nan_aware(series.to_numpy(), factor, decimated_length(len(series), factor)),
                index=series.index[::factor],
            )
            dt *= factor

        if dt != self.dt or not self.datetime_index.isin(series.index).all():
            raise ValueError("源文件与当前时间轴不匹配，请使用整个数据集还原。")
        return series.loc[self.datetime_index].rename(channel).copy()

    def restore_dataset(self, source: "MatEMData", *, keep_station_metadata: bool = False) -> None:
        """Adopt a successfully loaded source while preserving this object's identity."""
        state = dict(vars(source))
        state["data"] = source.data.copy(deep=False)
        state["_source_channels"] = dict(source._source_channels)
        state["_source_time_operations"] = list(source._source_time_operations)
        if keep_station_metadata:
            for field in ("name", "latitude", "longitude", "elevation"):
                state[field] = getattr(self, field)
        self.__dict__.clear()
        self.__dict__.update(state)

    # ─── update_meta ───────────────────────────────────────

    def update_meta(self):
        ts = self.mat_meta[self._struct_key]

        # 裁剪首尾 NaN
        all_first, all_last = [], []
        for ch_name in self.channel_names():
            non_nan = np.flatnonzero(~np.isnan(self.data[ch_name].to_numpy()))
            if len(non_nan) > 0:
                all_first.append(non_nan[0])
                all_last.append(non_nan[-1])

        if all_first and all_last:
            first_valid, last_valid = min(all_first), max(all_last)
            if first_valid > 0 or last_valid < self.npts - 1:
                old_npts = self.npts
                self.trim_time_range(
                    self.datetime_index[first_valid], self.datetime_index[last_valid],
                )

                # timeStamp 加载时被释放、保存时整体重建（见 mat_io
                # 的 _build_mdict），裁剪只同步 npts/start_time 即可，
                # 不再逐段维护 timeStamp

                logger.info(
                    "裁剪首尾 NaN: %d → %d 点 (去除前 %d 后 %d)",
                    old_npts, self.npts, first_valid, old_npts - 1 - last_valid,
                )

        # 反推时间写入
        actual_start = self.datetime_index[0]
        actual_end = self.datetime_index[-1]
        station_tz = pytz.timezone(self.timezone) if self.timezone else None

        def _to_arr(ts_val):
            if station_tz:
                ts_val = ts_val.tz_convert(station_tz)
            return np.array([pts_to_array(ts_val)])

        write_start = _to_arr(actual_start)
        write_end = _to_arr(actual_end)
        write_zero = _to_arr(self.zero_time)

        # 重写 data（写回存在的字段，不存在的跳过）
        new_data = np.empty((self.NCh, self.npts), dtype=np.float64)
        for i, ch_name in enumerate(self.channel_names()):
            new_data[i, :] = np.asarray(
                self.data[ch_name].to_numpy(), dtype=np.float64,
            )
        self._try_set(ts, 'data', new_data)
        self._try_set(ts, 'startTime', write_start)
        self._try_set(ts, 'endTime', write_end)
        self._try_set(ts, 'zeroTime', write_zero)
        self._try_set(ts, 'dt', np.array([self.dt.total_seconds()]))
        # MATLAB 数值默认是 double。若写成 NumPy/Python 整数，转换后的
        # TS.npts/NCh 会成为 int64，TS.findTime 中的日期运算会失败。
        self._try_set(ts, 'npts', np.array([[self.npts]], dtype=np.float64))
        try:
            ts['NCh'][0, 0] = np.array([[self.NCh]], dtype=np.float64)
        except ValueError:
            self._try_set(ts, 'nch', np.array([[self.NCh]], dtype=np.float64))

        # ── 台站元数据与通道信息同步 ─────────────────────────
        # 写回布局与 __init__ 的读取表达式镜像：字段是 (1,1) object
        # cell，ts[field][0,0] 放值数组。
        self._try_set(ts, 'name', np.array([self.name]))
        if self.latitude is not None and not np.isnan(self.latitude):
            self._try_set(ts, 'latitude', np.array([[float(self.latitude)]]))
        if self.longitude is not None and not np.isnan(self.longitude):
            self._try_set(ts, 'longitude', np.array([[float(self.longitude)]]))
        if self.elevation is not None:
            self._try_set(ts, 'elevation', np.array([float(self.elevation)]))

        def _cell_row(values: list[str]) -> np.ndarray:
            # MATLAB 1×N cell 行：每个元素是含字符串的 (1,) char 数组，
            # 与 loadmat 读回的布局一致（chid[i][0] / units.flat[i][0]）
            arr = np.empty((1, len(values)), dtype=object)
            for i, v in enumerate(values):
                arr[0, i] = np.array([str(v)])
            return arr

        self._try_set(ts, 'chid', _cell_row(self.channel_names()))
        units = [u for u in (self.m_units, self.e_units) if u]
        if units:
            self._try_set(ts, 'units', _cell_row(units))

    @staticmethod
    def _try_set(ts, field: str, value):
        try:
            ts[field][0, 0] = value
        except (ValueError, KeyError):
            pass

    def restore_data(self, ch: str, *, source: "MatEMData") -> None:
        self.set_channel_series(ch, self.source_channel_series(source, ch))
