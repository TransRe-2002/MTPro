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
            self.elevation = elev[0, 0][0]
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
                self.data = self.data.isel(time=slice(first_valid, last_valid + 1))
                self.datetime_index = self.datetime_index[first_valid:last_valid + 1]
                self.npts = last_valid - first_valid + 1
                self.start_time = self.datetime_index[0]
                self.end_time = self.datetime_index[-1]

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
        self._try_set(ts, 'npts', np.array([[self.npts]]))
        try:
            ts['NCh'][0, 0] = np.array([[self.NCh]])
        except ValueError:
            self._try_set(ts, 'nch', np.array([[self.NCh]]))

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

    def restore_data(self, ch: str):
        pass
