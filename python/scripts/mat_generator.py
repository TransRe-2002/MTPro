"""
tsStruct MAT文件构建器
======================
设计模式：
  1. TS_STRUCT_TEMPLATE  —— 原始空模板（只读，标注类型，不要直接修改）
  2. new_ts_struct()     —— 每次深拷贝模板，返回一个可填充的副本
  3. fill_ts_struct()    —— 把你的实际数据填入副本
  4. save_ts_mat()       —— 存储为 .mat 文件

时区约定：
  - 所有时间输入默认视为东八区（UTC+8）
  - 若传入带 tzinfo 的 pd.Timestamp，会先转换到 UTC+8 再使用
  - MATLAB datevec 存储的是本地时间（东八区），不含时区信息

依赖:
    pip install scipy numpy pandas
"""

import copy
import numpy as np
import pandas as pd
import scipy.io as sio
from typing import List, Optional, Union

# 统一的时间类型别名
TimeInput = Union[pd.Timestamp, str]   # 支持 pd.Timestamp 或可被 pd.Timestamp() 解析的字符串

# 东八区
_TZ_CST_ = "Asia/Shanghai"


# ══════════════════════════════════════════════════════════════
#  模板私有常量（尾部加下划线表示模块内部使用）
# ══════════════════════════════════════════════════════════════

_EMPTY_F64_    = np.zeros((0, 0), dtype=np.float64)   # MATLAB []      (0×0 double)
_EMPTY_CELL_   = np.empty((1, 0), dtype=object)        # MATLAB {}      (1×0 cell)  ← 必须是行向量形状
_DATEVEC_ZERO_ = np.zeros((1, 6), dtype=np.float64)   # MATLAB [0×6]   (1×6 double)


# ══════════════════════════════════════════════════════════════
#  原始空模板（只读，字段顺序与 MATLAB tsStruct 保持一致）
# ══════════════════════════════════════════════════════════════

TS_STRUCT_TEMPLATE: dict = {
    # ── 维度信息 ──────────────────────────────────────────────
    "NCh":           np.float64(0),           # 通道总数          | double 1×1
    "npts":          np.float64(0),           # 单通道采样点数    | double 1×1

    # ── 核心数据 ──────────────────────────────────────────────
    "data":          _EMPTY_F64_.copy(),      # 波形数据          | double NCh×npts

    # ── 时间信息 ──────────────────────────────────────────────
    "startTime":     _DATEVEC_ZERO_.copy(),   # 开始时刻          | double 1×6  [y,m,d,H,M,S]
    "zeroTime":      _DATEVEC_ZERO_.copy(),   # 零时刻            | double 1×6
    "endTime":       _DATEVEC_ZERO_.copy(),   # 结束时刻          | double 1×6
    "dt":            np.float64(0),           # 采样间隔（秒）    | double 1×1
    "timeStamp":     _EMPTY_F64_.copy(),      # 时间戳序列        | double 1×npts
    "timeStampType": "years",                 # 时间戳单位        | char

    # ── 台站地理信息 ──────────────────────────────────────────
    "name":          "",                      # 台站名            | char
    "latitude":      np.float64(0),           # 纬度（度）        | double 1×1
    "longitude":     np.float64(0),           # 经度（度）        | double 1×1
    "elevation":     _EMPTY_F64_.copy(),      # 海拔（米，可选）  | double 1×1 or []

    # ── 通道 / 记录描述 ───────────────────────────────────────
    "chid":          _EMPTY_CELL_.copy(),     # 通道ID列表        | cell  1×NCh  ← 行向量
    "units":         _EMPTY_CELL_.copy(),     # 物理量单位        | cell  1×2   ← 固定两格：[磁感应强度单位, 电场强度单位]

    # ── 质量控制 ──────────────────────────────────────────────
    "missVals":      np.float64(np.nan),      # 缺失值标记        | double 1×1
    "segments":      _EMPTY_F64_.copy(),      # 分段信息          | double (可选)
    "badRec":        _EMPTY_F64_.copy(),      # 坏记录标记        | double (可选)

    # ── 其他元信息 ────────────────────────────────────────────
    "runID":         _EMPTY_F64_.copy(),      # 运行ID            | double or char (可选)
    "clockRef":      _EMPTY_F64_.copy(),      # 时钟参考          | double (可选)
    "UserData":      _EMPTY_CELL_.copy(),     # 用户自定义        | cell   1×N   ← 行向量
    "metadata":      _EMPTY_F64_.copy(),      # 元数据            | double (可选)
}


# ══════════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════════

def _to_cst_(t: TimeInput) -> pd.Timestamp:
    """
    将任意 TimeInput 统一转换为东八区（CST/UTC+8）的 pd.Timestamp。

    规则：
      - 无时区信息（naive）→ 直接视为东八区，不做数值偏移
      - 有时区信息（aware）→ 转换到东八区（数值会偏移）
    """
    ts = pd.Timestamp(t)
    if ts.tzinfo is None:
        return ts.tz_localize(_TZ_CST_)
    return ts.tz_convert(_TZ_CST_)


def _ts_to_datevec_(t: TimeInput) -> np.ndarray:
    """
    pd.Timestamp / 字符串 → MATLAB datevec，shape (1, 6)，dtype float64。
    秒保留小数（含亚秒精度）。存储的是东八区本地时间。
    """
    cst = _to_cst_(t)
    sec = cst.second + cst.microsecond / 1e6
    return np.array(
        [[cst.year, cst.month, cst.day, cst.hour, cst.minute, sec]],
        dtype=np.float64
    )


def _make_timestamp_(start: TimeInput, dt_sec: float, npts: int) -> np.ndarray:
    """
    生成以"年的小数"表示的时间戳序列，shape (1, npts)。
    对应 MATLAB timeStampType = 'years'，基准为东八区本地时间。
    """
    cst        = _to_cst_(start)
    year_start = pd.Timestamp(year=cst.year, month=1, day=1, tz=_TZ_CST_)
    year_days  = 366 if (cst.year % 4 == 0 and
                         (cst.year % 100 != 0 or cst.year % 400 == 0)) else 365
    secs_in_year = year_days * 86400.0
    t0_frac    = (cst - year_start).total_seconds() / secs_in_year
    arr = np.array(
        [cst.year + t0_frac + i * dt_sec / secs_in_year for i in range(npts)],
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


# ══════════════════════════════════════════════════════════════
#  Step 1 — 从模板深拷贝，得到可写副本
# ══════════════════════════════════════════════════════════════

def new_ts_struct() -> dict:
    """
    返回模板的深拷贝（空白副本），之后用 fill_ts_struct() 填充。
    永远不要直接修改 TS_STRUCT_TEMPLATE。
    """
    return copy.deepcopy(TS_STRUCT_TEMPLATE)


# ══════════════════════════════════════════════════════════════
#  Step 2 — 填充核心字段
# ══════════════════════════════════════════════════════════════

def fill_ts_struct(
    ts: dict,
    *,
    # ── 必填 ──────────────────────────────────────────────────
    data:       np.ndarray,          # shape (NCh, npts)，dtype float64
    ch_names:   List[str],           # 长度 == NCh，每通道独立名称
    start_dt:   TimeInput,           # 支持 pd.Timestamp / 字符串，默认视为东八区
    end_dt:     TimeInput,
    dt_sec:     float,               # 采样间隔（秒）
    name:       str,                 # 台站名
    latitude:   float,
    longitude:  float,
    # ── 选填 ──────────────────────────────────────────────────
    zero_dt:    Optional[TimeInput]  = None,   # 默认：start_dt 所在年 1月1日 00:00 CST
    units:      Optional[List[str]]  = None,   # 固定 1×2：[磁感应强度单位, 电场强度单位]，如 ["nT", "mv/km"]；None → ["", ""]
    elevation:  Optional[float]      = None,   # None → 保持空矩阵 []
    miss_vals:  float                = np.nan,
    auto_timestamp: bool             = True,   # True → 自动生成 timeStamp
) -> dict:
    """
    将实际数据填入 new_ts_struct() 返回的副本（in-place 修改并返回）。

    时区约定
    --------
    - naive Timestamp（无时区）→ 直接视为东八区，数值不变
    - aware Timestamp（带时区）→ 转换到东八区后写入 datevec

    cell 字段维度约定
    -----------------
    - chid / UserData 存为 1×N object array（MATLAB cell 行向量）
    - units 固定存为 1×2：[磁感应强度单位, 电场强度单位]，与通道数无关
    避免变量编辑器因列向量触发索引错误。
    """
    data = np.atleast_2d(np.array(data, dtype=np.float64))
    NCh, npts = data.shape

    # ── 入参校验 ──────────────────────────────────────────────
    if len(ch_names) != NCh:
        raise ValueError(
            f"ch_names 长度 ({len(ch_names)}) 与 data 行数 ({NCh}) 不一致"
        )
    if units is not None and len(units) != 2:
        raise ValueError(
            f"units 必须恰好包含 2 个元素 [磁感应强度单位, 电场强度单位]，当前长度 {len(units)}"
        )

    # zero_dt 默认值
    if zero_dt is None:
        cst_start = _to_cst_(start_dt)
        zero_dt   = pd.Timestamp(year=cst_start.year, month=1, day=1, tz=_TZ_CST_)

    # ── 填充字段 ──────────────────────────────────────────────
    ts["NCh"]       = np.float64(NCh)
    ts["npts"]      = np.float64(npts)
    ts["data"]      = data

    ts["startTime"] = _ts_to_datevec_(start_dt)
    ts["zeroTime"]  = _ts_to_datevec_(zero_dt)
    ts["endTime"]   = _ts_to_datevec_(end_dt)
    ts["dt"]        = np.float64(dt_sec)

    ts["name"]      = name
    ts["latitude"]  = np.float64(latitude)
    ts["longitude"] = np.float64(longitude)
    ts["missVals"]  = np.float64(miss_vals)

    # cell 字段：必须是 1×N object array
    ts["chid"]  = _make_cell_row_(ch_names)
    ts["units"] = _make_cell_row_(units if units is not None else ["", ""])  # 固定 1×2

    # 海拔（选填）
    if elevation is not None:
        ts["elevation"] = np.float64(elevation)

    # 时间戳（选填，默认自动生成）
    if auto_timestamp:
        ts["timeStamp"] = _make_timestamp_(start_dt, dt_sec, npts)

    return ts


# ══════════════════════════════════════════════════════════════
#  Step 3 — 存储为 .mat 文件
# ══════════════════════════════════════════════════════════════

def save_ts_mat(filepath: str, ts: dict, var_name: str = "tsStruct") -> None:
    """
    将填充好的 tsStruct 字典保存为 MATLAB .mat 文件。

    Parameters
    ----------
    filepath : 输出路径，建议以 .mat 结尾
    ts       : fill_ts_struct() 填充完毕的字典
    var_name : MATLAB 工作区中的变量名，默认 'tsStruct'
    """
    sio.savemat(filepath, {var_name: ts}, do_compression=True)
    print(f"[保存成功] {filepath}  (MATLAB 变量名: '{var_name}')")


# ══════════════════════════════════════════════════════════════
#  使用示例
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── 例1：pd.Timestamp naive（无时区）→ 直接视为东八区 ───
    NCh_A, npts_A = 7, 259201
    data_A = np.random.randn(NCh_A, npts_A) * 1e-6

    ts_A = new_ts_struct()
    fill_ts_struct(
        ts_A,
        data      = data_A,
        ch_names  = ["BHZ", "BHN", "BHE", "HHZ", "HHN", "HHE", "LHZ"],
        start_dt  = pd.Timestamp("2024-04-30 16:00:00"),   # naive → 视为 CST
        end_dt    = pd.Timestamp("2024-05-15 16:00:00"),
        dt_sec    = 5.0,
        name      = "039BE",
        latitude  = 40.7348,
        longitude = 106.6959,
        units     = ["nT", "mv/km"],   # 固定 1×2：[磁感应强度, 电场强度]
        elevation = 1200.0,
    )
    save_ts_mat("039BE_example.mat", ts_A)

    # ── 例2：pd.Timestamp aware（带时区）→ 自动转为东八区 ───
    NCh_B, npts_B = 3, 86400
    data_B = np.random.randn(NCh_B, npts_B) * 5e-7

    ts_B = new_ts_struct()
    fill_ts_struct(
        ts_B,
        data      = data_B,
        ch_names  = ["Z", "N", "E"],
        start_dt  = pd.Timestamp("2024-05-01 00:00:00", tz="UTC"),  # aware → 转 CST
        end_dt    = pd.Timestamp("2024-05-02 00:00:00", tz="UTC"),
        dt_sec    = 1.0,
        name      = "005E",
        latitude  = 39.9042,
        longitude = 116.4074,
        units     = ["nT", "mv/km"],   # 固定 1×2
    )
    save_ts_mat("005E_example.mat", ts_B)

    # ── 验证读取 ─────────────────────────────────────────────
    print("\n=== 验证 039BE ===")
    s = sio.loadmat("039BE_example.mat", squeeze_me=True, struct_as_record=False)["tsStruct"]
    print(f"  NCh={int(s.NCh)}, npts={int(s.npts)}, data.shape={s.data.shape}")
    print(f"  chid       = {list(s.chid)}")
    print(f"  units      = {list(s.units)}  (应为 ['nT', 'mv/km'])")
    print(f"  startTime  = {s.startTime}")

    print("\n=== 验证 005E（UTC 输入转 CST 后存储）===")
    s2 = sio.loadmat("005E_example.mat", squeeze_me=True, struct_as_record=False)["tsStruct"]
    print(f"  NCh={int(s2.NCh)}, npts={int(s2.npts)}, data.shape={s2.data.shape}")
    print(f"  startTime = {s2.startTime}  (应为 CST 08:00:00)")