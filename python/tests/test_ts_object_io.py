"""Deterministic MCOS I/O tests; no MATLAB, Runtime or measurement files."""
from __future__ import annotations

import numpy as np
import pytest
from matio import load_from_mat
from scipy.io import loadmat, savemat

from io_utils import ts_convert, ts_object_io
from io_utils.mat_io import MatLoader, MatSaver
from io_utils.mcos_reader import extract_ts_struct, find_ts_object_stub


def _source():
    data = np.arange(24, dtype=np.float64).reshape(2, 12)
    data[0, 3] = np.nan
    return {'tsStruct': {
        'data': data, 'dt': 5, 'name': '测试台站',
        'startTime': np.array([[2024, 4, 30, 16, 0, 0]], dtype=np.int64),
        'zeroTime': np.array([[2024, 4, 30, 0, 0, 0]], dtype=np.int64),
        'endTime': np.empty((0, 0)), 'timeStamp': np.empty((0, 0)),
        'chid': np.array([['Hx', 'Hy']], dtype=object),
        'metadata': {'source': 'test'}, 'missVals': 9999,
    }}


@pytest.mark.parametrize('version', ['v7', 'v7.3'])
def test_real_ts_object_preserves_fields_without_converter(tmp_path, monkeypatch, version):
    monkeypatch.setattr(ts_convert, 'find_converter', lambda: pytest.fail('external converter used'))
    source = _source()
    target = tmp_path / f'object_{version}.mat'
    ts_object_io.write_ts_object(source, target, version=version)
    obj = load_from_mat(str(target), raw_data=True)['tsObj']
    assert obj.classname == 'TS'
    np.testing.assert_equal(obj.properties['data'], source['tsStruct']['data'])
    assert np.asarray(obj.properties['npts']).item() == 12
    assert np.asarray(obj.properties['NCh']).item() == 2
    assert np.asarray(obj.properties['startTime']).dtype == np.float64
    assert np.asarray(obj.properties['missVals']).item() == 9999
    if version == 'v7':
        assert find_ts_object_stub(loadmat(target))[1] == 'TS'
        assert extract_ts_struct(loadmat(target)) is not None
    # MatLoader exercises both its light v7 path and the full v7.3 MCOS path.
    monkeypatch.setattr('core.mat_data.fetch_kp', lambda *_args: None)
    em = MatLoader.load(str(target))
    assert em.name == '测试台站'
    assert em.npts == 12 and em.NCh == 2
    np.testing.assert_equal(em.get_channel_series('Hx').to_numpy(), source['tsStruct']['data'][0])


def test_conversion_and_matsaver_never_call_external_converter(tmp_path, monkeypatch):
    monkeypatch.setattr(ts_convert, 'find_converter', lambda: pytest.fail('external converter used'))
    source = tmp_path / 'struct.mat'
    savemat(source, _source())
    obj_path = ts_convert.convert_to_ts_object(source, tmp_path / 'object.mat')
    struct_path = ts_convert.convert_to_struct(obj_path, tmp_path / 'back.mat')
    np.testing.assert_equal(loadmat(struct_path)['tsStruct']['data'][0, 0], _source()['tsStruct']['data'])
    saved = tmp_path / 'saved.mat'
    MatSaver._write_mat(_source(), str(saved))
    assert find_ts_object_stub(loadmat(saved))[1] == 'TS'


def test_failed_write_preserves_existing_destination(tmp_path, monkeypatch):
    target = tmp_path / 'existing.mat'
    target.write_bytes(b'previous contents')

    def fail(path, *_args, **_kwargs):
        from pathlib import Path
        Path(path).write_bytes(b'partial write')
        raise OSError('simulated write failure')

    monkeypatch.setattr(ts_object_io, 'save_to_mat', fail)
    with pytest.raises(OSError, match='simulated'):
        ts_object_io.write_ts_object(_source(), target)
    assert target.read_bytes() == b'previous contents'
    assert list(tmp_path.iterdir()) == [target]


def test_missing_fields_fail_without_downgrading_to_struct(tmp_path):
    with pytest.raises(ValueError, match='startTime'):
        MatSaver._write_mat({'tsStruct': {'data': np.ones((2, 4)), 'dt': 1}}, str(tmp_path / 'bad.mat'))
    assert not (tmp_path / 'bad.mat').exists()


def test_missing_input_keeps_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        MatLoader.load(str(tmp_path / 'missing.mat'))
