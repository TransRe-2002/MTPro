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

_TZ_UTC_ = pytz.timezone('UTC')

_EMPTY_F64_ = np.zeros((0, 0), dtype=np.float64)
_EMPTY_CELL_ = np.zeros((1, 0), dtype=object)
_DATE_VEC_ = np.zeros((1, 6), dtype=np.float64)

# ---------------------------------------------------------------------------
# MATLAB v7.3 (HDF5) 文件读取支持
# ---------------------------------------------------------------------------

def _load_mat_v73(path: str) -> dict:
    """读取 MATLAB v7.3 (HDF5) 文件并转换为 scipy.io.loadmat 兼容的 dict。

    v7.3 文件用 HDF5 存储，结构与 scipy 的 (1,1) cell 嵌套格式完全不同。
    此函数负责将 HDF5 结构递归转换为与 scipy 输出一致的形式。
    """
    import h5py

    def _decode_char(raw: np.ndarray) -> str:
        """MATLAB char 数组 (uint16 UTF-16) → Python str。"""
        return raw.tobytes().decode('utf-16').rstrip('\x00')

    def _is_empty_ref(ds: h5py.Dataset) -> bool:
        """HDF5 中对空 MATLAB 数组的引用桩。"""
        return ds.dtype == np.uint64 and ds.ndim == 1 and ds.shape[0] <= 3

    def _wrap_cell(value) -> np.ndarray:
        """将值包装到 (1, 1) object 数组（模拟 MATLAB cell 语义）。"""
        cell = np.empty((1, 1), dtype=object)
        cell[0, 0] = value
        return cell

    def _convert_value(item, h5f: h5py.File):
        """将 HDF5 项（Group/Dataset）转换为 scipy 格式的值（未包装 cell）。"""
        # ---- Group → MATLAB struct → dict ----
        if isinstance(item, h5py.Group):
            fields = {}
            for key in item.keys():
                if key.startswith('#'):
                    continue
                fields[key] = _wrap_cell(_convert_value(item[key], h5f))
            # 返回 dict，外层会包装到 structured array
            return fields

        # ---- Dataset ----
        ds = item

        # 空数组引用桩
        if _is_empty_ref(ds):
            return np.zeros((0, 0), dtype=np.float64)

        # char 数组 (uint16) → 包装为 (1,) 数组，与 scipy 格式一致
        if ds.dtype == np.uint16:
            return np.array([_decode_char(ds[:])])

        # cell 数组 (object dtype, 元素是 HDF5 references)
        if ds.dtype == object:
            raw = ds[:]
            rows, cols = raw.shape
            # 构建 (rows, cols) object 数组，每个单元格递归转换
            cells = np.empty((rows, cols), dtype=object)
            for r in range(rows):
                for c in range(cols):
                    ref = raw[r, c]
                    if isinstance(ref, h5py.h5r.Reference):
                        obj = h5f[ref]
                        if isinstance(obj, h5py.Dataset) and obj.dtype == np.uint16:
                            # char 数组引用 → 字符串 → 包装为 (1,) 数组
                            cells[r, c] = np.array([_decode_char(obj[:])])
                        elif isinstance(obj, h5py.Dataset):
                            cells[r, c] = _convert_value(obj, h5f)
                        else:
                            cells[r, c] = _convert_value(obj, h5f)
                    else:
                        cells[r, c] = ref
            # scipy 格式中 cell 数组通常是行向量 (1, N)，与数值数组一样需转置
            if cells.ndim == 2:
                return cells.T
            return cells

        # 普通数值数组
        data = ds[:]
        # MATLAB 列优先 vs HDF5/C 行优先：所有 2D 数组需转置还原
        if data.ndim == 2:
            data = data.T
        # float 数组但值全为整数（如时间向量 [2024, 8, 11, ...]）
        # → 转为 int，避免 pandas Timestamp 误将 float 解释为 epoch
        if data.dtype == np.float64 and np.all(np.abs(data - np.round(data)) < 1e-10):
            data = data.astype(np.int64)
        return data

    with h5py.File(path, 'r') as f:
        result = {}
        for var_name in f.keys():
            if var_name.startswith('#'):
                continue
            item = f[var_name]
            converted = _convert_value(item, f)
            if isinstance(item, h5py.Group):
                # struct: 构建 (1, 1) structured array 模拟 scipy 格式
                field_names = list(converted.keys())
                dtype_spec = np.dtype([(fn, 'O') for fn in field_names])
                arr = np.empty((1, 1), dtype=dtype_spec)
                for fn in field_names:
                    arr[fn] = converted[fn]
                result[var_name] = arr
            else:
                result[var_name] = _wrap_cell(converted)
    return result

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
    year_start = pd.Timestamp(year=start.year, month=1, day=1, tz=_TZ_UTC_)
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
    def load(path: str, tz: str = 'Asia/Shanghai') -> MatEMData:
        # 检测文件格式：v7.3 / HDF5 需要 h5py 读取，v5/v4 用 scipy
        with open(path, 'rb') as fh:
            header = fh.read(128)
        if b'MATLAB 7.3 MAT-file' in header or header[:4] == b'\x89HDF':
            mat_data = _load_mat_v73(path)
        else:
            mat_data = loadmat(path)
        return MatEMData(mat_data, path, tz=tz)

def _save_mat_v73(path: str, mdict: dict) -> None:
    """将 scipy 格式 dict 写入 HDF5 文件（MTPro 内部格式，非 MATLAB 兼容）。"""
    import h5py

    def _write_item(h5f, name, value):
        if isinstance(value, np.ndarray) and value.dtype.names:
            grp = h5f.create_group(name)
            val_flat = value[0, 0]
            for fname in value.dtype.names:
                _write_item(grp, fname, val_flat[fname])
        elif isinstance(value, np.ndarray) and value.dtype == object and value.ndim == 2:
            rows, cols = value.shape[1], value.shape[0]
            dt = h5py.special_dtype(ref=h5py.Reference)
            ds = h5f.create_dataset(name, shape=(rows, cols), dtype=dt)
            for r in range(rows):
                for c in range(cols):
                    cell_val = value[c, r]
                    ref_name = f"{name}_r{r}_c{c}"
                    _write_item(h5f, ref_name, cell_val)
                    ds[r, c] = h5f[ref_name].ref
        elif isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == 1:
            if value.dtype == object and value.size == 1:
                inner = value[0]
                if isinstance(inner, str):
                    h5f.create_dataset(name, data=np.frombuffer(inner.encode('utf-16-le'), dtype=np.uint16))
                    return
                if isinstance(inner, np.ndarray) and inner.dtype.kind in ('U', 'S'):
                    s = str(inner[0]) if inner.size > 0 else ''
                    h5f.create_dataset(name, data=np.frombuffer(s.encode('utf-16-le'), dtype=np.uint16))
                    return
            _write_array(h5f, name, value)
        elif isinstance(value, np.ndarray):
            _write_array(h5f, name, value)
        elif isinstance(value, str):
            h5f.create_dataset(name, data=np.frombuffer(value.encode('utf-16-le'), dtype=np.uint16))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            h5f.create_dataset(name, data=np.array([value]))

    def _write_array(h5f, name, arr):
        if arr.size == 0:
            h5f.create_dataset(name, data=np.zeros((0, 0), dtype=np.float64))
            return
        if arr.dtype.kind == 'U':
            s = str(arr.flat[0]) if arr.size > 0 else ''
            h5f.create_dataset(name, data=np.frombuffer(s.encode('utf-16-le'), dtype=np.uint16))
            return
        if arr.ndim == 2:
            arr = arr.T
        if arr.dtype == np.int64:
            arr = arr.astype(np.float64)
        h5f.create_dataset(name, data=arr)

    with h5py.File(path, 'w') as h5f:
        for var_name, value in mdict.items():
            _write_item(h5f, var_name, value)
class MatSaver(EMDataSaver):
    @staticmethod
    def save(em_data: MatEMData, path: str):
        if isinstance(em_data, MatEMData):
            em_data.update_meta()
            if hasattr(em_data, "mat_meta"):
                key = em_data._struct_key
                savemat(path, {key: em_data.mat_meta[key]})
            else:
                raise TypeError(f"{type(em_data).__name__} is missing 'mat_meta' attribute")
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
            for i, channel_name in enumerate(em_data.channel_names()):
                data[i, :] = np.asarray(em_data.data[channel_name].to_numpy(), dtype=np.float64)
            ts["data"] = data

            ts["timeStamp"] = _make_timestamp_(em_data.start_time, em_data.dt.total_seconds(), em_data.npts)
            if hasattr(em_data, "UserData"):
                ts["UserData"] = _make_cell_row_(em_data.user_data)
            savemat(path, {"tsStruct": ts})
