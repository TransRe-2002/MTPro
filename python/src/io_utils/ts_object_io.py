"""Read and write genuine MCOS TS objects without MATLAB or its Runtime."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from matio import load_from_mat, save_to_mat
from matio.utils import MatlabOpaque
from scipy.io import loadmat, savemat

from io_utils.mcos_reader import _TS_FIELDS, _build_ts_struct


def read_ts_object(path: str | Path) -> dict:
    """Return the scalar TS object's properties in MTPro's struct format."""
    contents = load_from_mat(str(path), raw_data=True)
    for value in contents.values():
        if isinstance(value, MatlabOpaque) and value.classname == 'TS':
            if not isinstance(value.properties, dict):
                raise ValueError('Only scalar TS objects are supported')
            properties = {key: val for key, val in value.properties.items() if key in _TS_FIELDS}
            return _build_ts_struct(list(properties), properties.__getitem__)
    raise ValueError(f'Input file contains no TS object: {path}')


def _struct_properties(mdict: dict) -> dict:
    candidates = [mdict['tsStruct']] if 'tsStruct' in mdict else list(mdict.values())
    for value in candidates:
        if isinstance(value, dict) and 'data' in value and 'dt' in value:
            return dict(value)
        if isinstance(value, np.ndarray) and value.dtype.names:
            if {'data', 'dt'} <= set(value.dtype.names):
                if value.size != 1:
                    raise ValueError('Only scalar tsStruct values are supported')
                record = value.reshape(-1)[0]
                return {name: record[name] for name in value.dtype.names}
    raise ValueError('Input contains no tsStruct with data and dt fields')


def read_ts_struct(path: str | Path) -> dict:
    # SciPy handles UTF-8 char arrays from its own v7 writer correctly.
    # Use mat-io only for v7.3, which SciPy does not support.
    try:
        contents = loadmat(str(path))
    except NotImplementedError:
        contents = load_from_mat(str(path), raw_data=True)
    _struct_properties(contents)
    return contents


def _date_time(value) -> datetime:
    array = np.asarray(value)
    if array.dtype.kind in 'US':
        return pd.Timestamp(str(array.reshape(-1)[0])).to_pydatetime().replace(tzinfo=None)
    parts = np.asarray(value, dtype=np.float64).reshape(-1)
    if parts.size != 6:
        raise ValueError('TS date fields must contain six datevec components')
    return datetime(*(int(part) for part in parts[:5])) + timedelta(seconds=float(parts[5]))


def _datevec(value: datetime) -> np.ndarray:
    return np.array([[value.year, value.month, value.day, value.hour, value.minute,
                      value.second + value.microsecond / 1e6]], dtype=np.float64)


def _decode_matlab_bytes(value: bytes | np.bytes_) -> str:
    """Convert byte strings from older MAT readers into ordinary Python text."""
    raw = bytes(value)
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        # Older MATLAB/scipy paths may expose one-byte character data rather
        # than UTF-8. Latin-1 is lossless for those raw byte values.
        return raw.decode('latin1')


def _normalize_text_values(value):
    """Recursively remove NumPy/Python byte scalars before mat-io encoding.

    ``matio.save_to_mat`` expects character values to provide ``str.encode``.
    NumPy's ``bytes_`` inherits from ``bytes`` and therefore has no ``encode``
    method, which used to make Save As fail for otherwise valid legacy files.
    """
    if isinstance(value, (bytes, np.bytes_)):
        return _decode_matlab_bytes(value)
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, np.void) and value.dtype.names:
        return {
            name: _normalize_text_values(value[name])
            for name in value.dtype.names
        }
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            # Structured metadata may contain fixed-width byte fields several
            # levels below the TS property. Rebuild it with object fields so
            # decoded Python strings are not coerced back to ``np.bytes_``.
            normalized = np.empty(
                value.shape,
                dtype=[(name, object) for name in value.dtype.names],
            )
            for name in value.dtype.names:
                for index in np.ndindex(value.shape):
                    normalized[name][index] = _normalize_text_values(
                        value[name][index]
                    )
            return normalized
        if value.dtype.kind == 'S':
            normalized = np.empty(value.shape, dtype=object)
            for index in np.ndindex(value.shape):
                normalized[index] = _decode_matlab_bytes(value[index])
            return normalized
        if value.dtype == object:
            normalized = np.empty(value.shape, dtype=object)
            for index in np.ndindex(value.shape):
                normalized[index] = _normalize_text_values(value[index])
            return normalized
        return value
    if isinstance(value, dict):
        return {
            key: _normalize_text_values(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_text_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_text_values(item) for item in value)
    return value


def ts_properties(mdict: dict) -> dict:
    """Match struct_to_ts's defaults and MATLAB numeric date semantics."""
    source = _struct_properties(mdict)
    for field in ('data', 'startTime', 'dt'):
        if field not in source:
            raise ValueError(f'tsStruct is missing required field {field!r}')
    properties = {key: _normalize_text_values(
                      val if val is not None else np.empty((0, 0)))
                  for key, val in source.items() if key in _TS_FIELDS}
    data = np.asarray(properties['data'])
    if data.ndim != 2:
        raise ValueError('TS data must have shape (n_channels, n_samples)')
    dt = np.asarray(properties['dt'], dtype=np.float64)
    if dt.size != 1 or not np.isfinite(dt.item()) or dt.item() <= 0:
        raise ValueError('TS dt must be a finite positive scalar')
    properties['dt'] = dt
    properties['NCh'] = np.float64(data.shape[0])
    properties['npts'] = np.float64(data.shape[1])
    properties.setdefault('name', '')
    properties.setdefault('zeroTime', properties['startTime'])
    properties.setdefault('missVals', np.float64(np.nan))
    properties.setdefault('timeStampType', source.get('timeStampStyle', 'years'))
    for field in ('startTime', 'zeroTime', 'endTime'):
        if field in properties and np.asarray(properties[field]).dtype.kind not in 'US':
            properties[field] = np.asarray(properties[field], dtype=np.float64)
    start = _date_time(properties['startTime'])
    if np.asarray(properties.get('endTime', [])).size == 0:
        properties['endTime'] = _datevec(start + timedelta(seconds=(data.shape[1] - 1) * dt.item()))
    if np.asarray(properties.get('timeStamp', [])).size == 0:
        zero = _date_time(properties['zeroTime'])
        offset = (start - zero).total_seconds() / (365.25 * 86400) + zero.year
        properties['timeStamp'] = (
            offset + np.arange(data.shape[1]) * dt.item() / (365.25 * 86400)
        ).reshape(1, -1)
        properties['timeStampType'] = 'years'
    return properties


def _atomic_write(path: str | Path, writer) -> Path:
    """A failed write must leave any existing destination untouched."""
    destination = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.mtpro_ts_', suffix='.mat', dir=destination.parent)
    os.close(fd)
    try:
        writer(temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def write_ts_object(mdict: dict, path: str | Path, *, version: str = 'v7') -> Path:
    properties = ts_properties(mdict)
    obj = MatlabOpaque(properties=properties, classname='TS')
    return _atomic_write(path, lambda temporary: save_to_mat(
        temporary, {'tsObj': obj}, version=version, oned_as='row',
    ))


def write_ts_struct(mdict: dict, path: str | Path) -> Path:
    return _atomic_write(path, lambda temporary: savemat(temporary, mdict))
