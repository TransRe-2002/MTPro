"""Pure-Python reader for MATLAB ``TS`` objects serialized in v5/v7 .mat files.

scipy.io cannot interpret MATLAB class objects (MCOS serialization): it only
returns ``MatlabOpaque`` placeholders and even loses the variable name.  The
object content is, however, still fully present in the file's hidden
workspace stream.  This module extracts that stream and reconstructs a plain
``tsStruct`` dict that :class:`core.mat_data.MatEMData` can consume — no
MATLAB or MATLAB Runtime involved.

Layout summary (verified against R2025b-written files)::

    loadmat(path)
      └─ '__function_workspace__'   raw uint8 blob
           [8-byte mini header: version(2B) + endian(2B) + zeros(4B)]
           └─ miMATRIX struct { MCOS: opaque }
                opaque['arr']  (N,1) object cell
                  [0]   classdef blob: u32 header + null-terminated string
                        table (property names + class name)
                  [1]   internal
                  [2:]  property values, arr[2+i] <-> i-th property name
                        (string table order, class name excluded)

Files this cannot handle (e.g. v7.3/HDF5 object saves) use the complete
mat-io reader in ts_object_io, also without MATLAB Runtime.
"""

from __future__ import annotations

import io
import logging

import numpy as np
from scipy.io.matlab import MatlabObject, MatlabOpaque
from scipy.io.matlab._mio5 import MatFile5Reader

logger = logging.getLogger(__name__)

# scipy 合成 v5 文件头：116 描述文本 + 8 子系统偏移 + 2 版本 + 2 字节序
_SYNTH_HEADER_PREFIX = b'MCOS-extract'.ljust(116, b' ') + b'\x00' * 8 + b'\x00\x01'

_MCOS_WORKSPACE_MINI_HEADER = 8  # version(2B) + endian(2B) + zeros(4B)

# 与 matlab/TS.m 一致的属性集合。新版 MATLAB 的 TS 对象文件可能携带额外的
# table 内部属性（ndims/nrows/rownames/nvars/varnames/props/table 等），
# 其中可能含有 scipy 无法写回的值（如 None），TS 写入适配器
# 只会拷贝 TS 类已定义的属性——这些额外属性在所有路径上都是
# 死数据，重建时直接丢弃。
_TS_FIELDS = frozenset({
    'NCh', 'npts', 'data', 'startTime', 'zeroTime', 'endTime',
    'clockRef', 'UserData', 'dt', 'name', 'runID', 'chid',
    'latitude', 'longitude', 'elevation', 'missVals', 'segments',
    'badRec', 'timeStamp', 'timeStampType', 'metadata', 'units',
})


def _opaque_parts(opaque: np.ndarray):
    """把 MatlabOpaque 占位桩归一化为 (类型系统, 类名, 载荷)。

    scipy 两代布局：
      <= 1.17: dtype ('s0','s1','s2','arr')，值为 bytes，变量名丢失为 'None'
      >= 1.18: dtype ('_TypeSystem','_Class','_ObjectMetadata')，值为 str，变量名保留
    载荷在外层桩中是 uint32 元数据，在 FileWrapper__ 桩中是属性值 cell 表。
    未知布局返回 None。
    """
    names = opaque.dtype.names or ()
    row = opaque.ravel()[0]
    if 's1' in names and 's2' in names:
        marker, classname = row['s1'], row['s2']
        payload = row['arr'] if 'arr' in names else None
    elif '_TypeSystem' in names and '_Class' in names:
        marker, classname = row['_TypeSystem'], row['_Class']
        payload = row['_ObjectMetadata'] if '_ObjectMetadata' in names else None
    else:
        return None
    if isinstance(marker, bytes):
        marker = marker.decode('latin1')
    if isinstance(classname, bytes):
        classname = classname.decode('latin1')
    return marker, str(classname), payload


def find_ts_object_stub(mat_data: dict):
    """在 loadmat 结果中查找 TS 对象的 MCOS 占位桩。

    返回 (变量 key, 类名)；未找到返回 (None, None)。
    """
    for key, value in mat_data.items():
        if not isinstance(value, MatlabOpaque) or value.size == 0:
            continue
        parts = _opaque_parts(value)
        if parts is None:
            continue
        marker, classname, _ = parts
        if marker == 'MCOS' and classname == 'TS':
            return key, classname
    return None, None


def extract_ts_struct(mat_data: dict) -> dict | None:
    """从 loadmat 结果重建 ``{'tsStruct': structured_array}``。

    仅当 *mat_data* 来自含序列化 TS 对象的 v5/v7 文件时成功；其余情况
    （无对象、v7.3 对象、未知布局）返回 None，由调用方决定兜底策略。
    """
    # 老式 @TS 类对象：scipy 能还原出带属性字段的 MatlabObject，直接可用
    for value in mat_data.values():
        if (isinstance(value, MatlabObject) and getattr(value, 'classname', None) == 'TS'
                and value.dtype.names):
            names = [n for n in value.dtype.names if n in _TS_FIELDS]
            return _build_ts_struct(names, lambda n, v=value: v[n][0, 0])

    stub_key, class_name = find_ts_object_stub(mat_data)
    if stub_key is None:
        return None

    workspace = mat_data.get('__function_workspace__')
    if workspace is None:
        logger.debug("TS 对象文件缺少隐藏工作区流，无法纯 Python 解析")
        return None
    raw = np.asarray(workspace, dtype=np.uint8).tobytes()
    if len(raw) < _MCOS_WORKSPACE_MINI_HEADER + 16:
        return None

    endian = raw[2:4]
    if endian not in (b'IM', b'MI'):
        logger.debug("未知工作区字节序标记: %r", endian)
        return None
    byte_order = '<' if endian == b'IM' else '>'

    try:
        values = _read_mcos_values(raw, endian)
        prop_names = _parse_classdef_strings(values, byte_order, class_name)
    except Exception as exc:  # noqa: BLE001 - 解析失败应走兜底而不是抛出
        logger.debug("MCOS 工作区解析失败: %s", exc)
        return None

    if prop_names is None:
        return None

    n_props = len(prop_names)
    if values.shape[0] < 2 + n_props:
        logger.debug("MCOS 值表长度不足: %d < %d", values.shape[0], 2 + n_props)
        return None

    dropped = [n for n in prop_names if n not in _TS_FIELDS]
    if dropped:
        logger.debug("丢弃 TS 对象的未知属性: %s", dropped)
    kept = [n for n in prop_names if n in _TS_FIELDS]
    prop_values = {name: values[2 + i, 0] for i, name in enumerate(prop_names)}
    return _build_ts_struct(kept, prop_values.__getitem__)


def _read_mcos_values(raw: bytes, endian: bytes) -> np.ndarray:
    """解开隐藏工作区流，返回 FileWrapper__ 对象的值 cell 数组。"""
    synth_header = _SYNTH_HEADER_PREFIX + endian
    reader = MatFile5Reader(io.BytesIO(synth_header + raw[_MCOS_WORKSPACE_MINI_HEADER:]))
    reader.initialize_read()
    reader.read_file_header()

    while not reader.end_of_stream():
        hdr, next_position = reader.read_var_header()
        try:
            arr = reader.read_var_array(hdr, process=True)
        finally:
            reader.mat_stream.seek(next_position)
        values = _find_mcos_values(arr)
        if values is not None:
            return values
    raise ValueError("工作区流中未找到 MCOS 容器")


def _find_mcos_values(arr):
    """若 arr 是含 MCOS 字段的 struct（或 MCOS opaque 本体），返回属性值表。"""
    opaque = None
    if isinstance(arr, MatlabOpaque):
        opaque = arr
    elif isinstance(arr, np.ndarray) and arr.dtype.names and 'MCOS' in arr.dtype.names:
        value = arr['MCOS'][0, 0]
        if isinstance(value, MatlabOpaque):
            opaque = value
    if opaque is None:
        return None
    parts = _opaque_parts(opaque)
    if parts is None:
        return None
    _, _, payload = parts
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        return payload
    return None


def _parse_classdef_strings(
    values: np.ndarray, byte_order: str, class_name: str,
) -> list[str] | None:
    """解析 classdef blob 的字符串表，返回去掉类名后的属性名列表（保持表内顺序）。"""
    classdef = np.asarray(values[0, 0]).ravel().tobytes()
    u32 = np.frombuffer(classdef, dtype=f'{byte_order}u4')
    if u32.size < 4:
        return None
    n_offset_pairs = int(u32[0])
    count = int(u32[1])
    pos = 4 * (2 + 2 * n_offset_pairs)

    names: list[str] = []
    for _ in range(count):
        end = classdef.find(b'\x00', pos)
        if end < 0:
            return None
        names.append(classdef[pos:end].decode('latin1'))
        pos = end + 1

    # 字符串表含一个类名条目（如 'TS'），属性值表不含它
    if class_name in names:
        names.remove(class_name)
    return names


def _build_ts_struct(prop_names: list[str], get_value) -> dict:
    """把属性值包装成 scipy loadmat 风格的 (1,1) structured array。"""
    dtype = np.dtype([(name, 'O') for name in prop_names])
    ts = np.empty((1, 1), dtype=dtype)
    for name in prop_names:
        value = get_value(name)
        if value is None:  # 防御：savemat 无法写 None
            value = np.zeros((0, 0))
        cell = np.empty((1, 1), dtype=object)
        cell[0, 0] = value
        ts[name] = cell
    return {'tsStruct': ts}


__all__ = ['extract_ts_struct', 'find_ts_object_stub']
