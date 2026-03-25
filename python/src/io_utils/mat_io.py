import copy

import numpy as np
import pandas as pd
from typing import List, Optional, Union
import pytz
from scipy.io import loadmat, savemat

from core.em_data import EMData
from core.mat_data import MatEMData

from io_utils.em_io import EMDataSaver, EMDataLoader
from utils.timestamp import pts_to_array

_TZ_CST_ = pytz.timezone('Asia/Shanghai')

_EMPTY_F64_ = np.zeros((0, 0), dtype=np.float64)
_EMPTY_CELL_ = np.zeros((1, 0), dtype=object)
_DATE_VEC_ = np.zeros((1, 6), dtype=np.float64)

TS_STRUCT_TEMPLATE = {
    # 维度信息
    "NCh": np.float64(0),
    "npts": np.float64(0),

    # 核心数据
    "data": _EMPTY_F64_.copy(),

    # 时间信息
    "startTime": _DATE_VEC_.copy(),
    "zeroTime": _DATE_VEC_.copy(),
    "endTime": _DATE_VEC_.copy(),
    "dt": np.float64(0),
    "timeStamp": _EMPTY_F64_.copy(),
    "timeStampStyle": "years",

    # 台站信息
    "name": "",
    "latitude": np.float64(0),
    "longitude": np.float64(0),
    "elevation": np.float64(0),

    # 通道信息
    "chid": _EMPTY_CELL_.copy(),
    "units": _EMPTY_CELL_.copy(),

    # 其他信息
    "missVals": np.float64(np.nan),
    "segments": _EMPTY_F64_.copy(),
    "badRec": _EMPTY_F64_.copy(),
    "runID": _EMPTY_F64_.copy(),
    "clockRef": _EMPTY_F64_.copy(),
    "UserData": _EMPTY_CELL_.copy(),
    "metadata" : _EMPTY_F64_.copy(),
}

def _make_timestamp_(start: pd.Timestamp, dt_sec: float, npts: int) -> np.ndarray:
    year_start = pd.Timestamp(year=start.year, month=1, day=1, tz=_TZ_CST_)
    year_days  = 366 if (start.year % 4 == 0 and
                         (start.year % 100 != 0 or start.year % 400 == 0)) else 365
    secs_in_year = year_days * 86400.0
    t0_frac    = (start - year_start).total_seconds() / secs_in_year
    arr = np.array(
        [start.year + t0_frac + i * dt_sec / secs_in_year for i in range(npts)],
        dtype=np.float64
    )
    return arr.reshape(1, -1)   # 存为 1×npts 行向量，与 MATLAB 一致


def _make_cell_row_(str_list: List[str]) -> np.ndarray:
    """
    字符串列表 → MATLAB 1×N cell 行向量。
    shape 必须是 (1, N)，否则 MATLAB 变量编辑器会报索引错误。
    """
    arr = np.empty((1, len(str_list)), dtype=object)
    for i, s in enumerate(str_list):
        arr[0, i] = s
    return arr

def new_ts_struct() -> dict:
    return copy.deepcopy(TS_STRUCT_TEMPLATE)

class MatLoader(EMDataLoader):
    @staticmethod
    def load(path: str) -> MatEMData:
        mat_data = loadmat(path)
        return MatEMData(mat_data, path)

class MatSaver(EMDataSaver):
    @staticmethod
    def save(em_data: MatEMData, path: str):
        if isinstance(em_data, MatEMData):
            em_data.update_meta()
            savemat(path, em_data.mat_meta)
        else:
            ts = new_ts_struct()
            ts["NCh"] = em_data.NCh
            ts["npts"] = em_data.npts
            ts["startTime"] = np.array([pts_to_array(em_data.start_time)])
            ts["endTime"] = np.array([pts_to_array(em_data.end_time)])
            ts["zeroTime"] = np.array([pts_to_array(em_data.zero_time)])
            ts["dt"] = np.array([em_data.dt.total_seconds()])

            ts["name"] = em_data.name
            ts["latitude"] = np.float64(em_data.latitude)
            ts["longitude"] = np.float64(em_data.longitude)
            ts["elevation"] = np.float64(em_data.elevation)

            ts["chid"] = _make_cell_row_(em_data.chid)
            if em_data.m_units is not None and em_data.e_units is not None:
                units = [em_data.m_units, em_data.e_units]
            elif em_data.m_units is not None:
                units = [em_data.m_units]
            elif em_data.e_units is not None:
                units = [em_data.e_units]
            else:
                units = []
            ts["units"] = _make_cell_row_(units)

            data = np.empty((em_data.NCh, em_data.npts), dtype=np.float64)
            for i, ch in enumerate(em_data.chid):
                data[i, :] = em_data.data[ch].cts.to_numpy()
            ts["data"] = data

            ts["timeStamp"] = _make_timestamp_(em_data.start_time, em_data.dt.total_seconds(), em_data.npts)
            if hasattr(em_data, "UserData"):
                ts["UserData"] = _make_cell_row_(em_data.user_data)
            savemat(path, {"tsStruct": ts})
