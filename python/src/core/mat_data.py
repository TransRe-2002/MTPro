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
        self.dt = pd.Timedelta(seconds=float(_field(ts, 'dt')[0, 0][0, 0]))

        # npts
        npts_raw = _field(ts, 'npts')[0, 0]
        self.npts = int(npts_raw[0, 0]) if npts_raw.size == 1 else int(npts_raw[0])

        # NCh / nch（模糊匹配）
        try:
            self.NCh = int(_field(ts, 'NCh')[0, 0][0, 0])
        except KeyError:
            self.NCh = int(_field(ts, 'nch')[0, 0][0, 0])

        # ── 时间 ──
        st = _field(ts, 'startTime')[0, 0][0]
        tz_obj = pytz.timezone(tz)
        self.start_time = pd.Timestamp(
            st[0], st[1], st[2], st[3], st[4], st[5], tz=tz_obj,
        ).tz_convert('UTC')

        # zeroTime 可选，无则用 startTime
        zt = _opt_field(ts, 'zeroTime', default=None)
        if zt is not None and zt[0, 0].size >= 6:
            zv = zt[0, 0][0]
            self.zero_time = pd.Timestamp(
                zv[0], zv[1], zv[2], zv[3], zv[4], zv[5], tz=tz_obj,
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
            elif u.size == 1:
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

                try:
                    stamp = _field(ts, 'timeStamp')[0, 0]
                    ts['timeStamp'][0, 0] = stamp[:, first_valid:last_valid + 1]
                except (KeyError, ValueError):
                    pass

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

        # 重写 data
        new_data = np.empty((self.NCh, self.npts), dtype=np.float64)
        for i, ch_name in enumerate(self.channel_names()):
            new_data[i, :] = np.asarray(
                self.data[ch_name].to_numpy(), dtype=np.float64,
            )
        ts['data'][0, 0] = new_data
        ts['startTime'][0, 0] = write_start
        ts['endTime'][0, 0] = write_end
        ts['zeroTime'][0, 0] = write_zero

        ts['dt'][0, 0] = np.array([self.dt.total_seconds()])
        ts['npts'][0, 0] = np.array([[self.npts]])
        # 保持原有字段名写回
        try:
            ts['NCh'][0, 0] = np.array([[self.NCh]])
        except ValueError:
            ts['nch'][0, 0] = np.array([[self.NCh]])

    def restore_data(self, ch: str):
        pass
