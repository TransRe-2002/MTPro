"""TS 对象 .mat 文件读取测试：纯 Python MCOS 解析 + mtpro_ts_convert 兜底。

scipy.io 无法解析序列化的 MATLAB TS 对象（只得到 MatlabOpaque 占位），
主路径由 io_utils/mcos_reader.py 纯 Python 重建 tsStruct（免 MATLAB Runtime）；
scripts/build_matlab.sh 编译的 mtpro_ts_convert 作为可选兜底（如 v7.3 对象文件）。
缺失 fixture 或可执行文件时相应测试自动跳过。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat
from scipy.io.matlab import MatlabOpaque

from core.mat_data import MatEMData
from io_utils import mcos_reader, ts_convert
from io_utils.mat_io import MatLoader, MatSaver

TESTS_DIR = Path(__file__).resolve().parent
# 两个 fixture 由 matlab/make_test_fixture.m / MATLAB 手工生成，
# 类定义字符串表布局不同（属性默认值设置情况不同），覆盖动态名称映射
TS_OBJECT_MAT = TESTS_DIR / "fixtures" / "ts_object.mat"
TS_OBJECT_NONDEFAULT_MAT = TESTS_DIR / "fixtures" / "ts_object_nondefault.mat"
STRUCT_MAT = TESTS_DIR / "test1.mat"                       # fixture 的数据来源
N_KEEP = 2000                                              # fixture 保留的采样数


def _converter_available() -> bool:
    try:
        ts_convert.find_converter()
        return True
    except FileNotFoundError:
        return False


needs_fixture = pytest.mark.skipif(
    not TS_OBJECT_MAT.exists(),
    reason=f"缺少 fixture: {TS_OBJECT_MAT}（matlab/make_test_fixture.m 生成）",
)
needs_converter = pytest.mark.skipif(
    not _converter_available(),
    reason="未找到 mtpro_ts_convert 可执行文件（scripts/build_matlab.sh 构建）",
)


def _expected_data() -> np.ndarray:
    return loadmat(str(STRUCT_MAT))['tsStruct']['data'][0, 0][:, :N_KEEP]


@needs_fixture
def test_scipy_cannot_parse_ts_object_file():
    """确认 TS 对象文件在 scipy 下只能得到不可解析的占位结果（解析的前提）。"""
    mat = loadmat(str(TS_OBJECT_MAT))
    non_dunder = {
        k: v for k, v in mat.items()
        if not (isinstance(k, str) and k.startswith('__'))
    }
    assert non_dunder
    assert any(
        not isinstance(k, str) or isinstance(v, MatlabOpaque)
        for k, v in non_dunder.items()
    )


@needs_fixture
@pytest.mark.parametrize("fixture", [TS_OBJECT_MAT, TS_OBJECT_NONDEFAULT_MAT])
def test_mcos_extract_ts_struct(fixture):
    """纯 Python 解析两种字符串表布局，字段与源数据一致。"""
    mat = loadmat(str(fixture))
    result = mcos_reader.extract_ts_struct(mat)
    assert result is not None and 'tsStruct' in result

    em = MatEMData(result, str(fixture))
    assert em.name == '039BE'
    assert em.npts == N_KEEP
    assert em.NCh == 7
    assert em.dt == pd.Timedelta(seconds=5)
    assert em.start_time == pd.Timestamp('2024-04-30 16:00:00', tz='Asia/Shanghai')
    assert list(em.chid) == ['Bx', 'By', 'Bz', 'Ex1', 'Ey1', 'Ex2', 'Ey2']
    actual = np.stack([em.data[ch].to_numpy() for ch in em.channel_names()])
    assert np.allclose(actual, _expected_data(), equal_nan=True)


@needs_fixture
@pytest.mark.parametrize("fixture", [TS_OBJECT_MAT, TS_OBJECT_NONDEFAULT_MAT])
def test_load_ts_object_mat_pure_python(fixture):
    """MatLoader.load 对 TS 对象文件走纯 Python 路径，无需任何外部可执行文件。"""
    em = MatLoader.load(str(fixture))
    assert em.name == '039BE'
    assert em.npts == N_KEEP
    assert em.start_time == pd.Timestamp('2024-04-30 16:00:00', tz='Asia/Shanghai')
    actual = np.stack([em.data[ch].to_numpy() for ch in em.channel_names()])
    assert np.allclose(actual, _expected_data(), equal_nan=True)


def test_extract_returns_none_for_plain_struct_file():
    """普通 tsStruct 文件不含 MCOS 对象，应返回 None（不误触发）。"""
    mat = loadmat(str(STRUCT_MAT))
    assert mcos_reader.extract_ts_struct(mat) is None


@needs_fixture
def test_extract_keeps_only_known_ts_fields():
    """重建结果只保留 TS 类定义的属性（新版类的 table 内部属性被丢弃）。"""
    mat = loadmat(str(TS_OBJECT_MAT))
    result = mcos_reader.extract_ts_struct(mat)
    assert set(result['tsStruct'].dtype.names) <= mcos_reader._TS_FIELDS


def test_build_ts_struct_replaces_none_with_empty():
    """None 值（savemat 无法写入）应被替换为空数组。"""
    result = mcos_reader._build_ts_struct(['name'], lambda n: None)
    value = result['tsStruct']['name'][0, 0]
    assert isinstance(value, np.ndarray) and value.size == 0


# 新版 MATLAB（Windows）写出的 TS 对象文件带额外 table 属性，其中可能含
# scipy 无法写回的 None——曾导致保存失败。本地数据条件覆盖。
NE_NEW_LAYOUT_MAT = Path('/home/transen5/Works/数据处理/东北数据/22-20260601-20260720-dt2.mat')


@pytest.mark.skipif(not NE_NEW_LAYOUT_MAT.exists(), reason="缺少东北数据测试文件")
@needs_converter
def test_load_and_save_new_layout_ts_object(tmp_path):
    em = MatLoader.load(str(NE_NEW_LAYOUT_MAT))
    assert set(em.mat_meta['tsStruct'].dtype.names) <= mcos_reader._TS_FIELDS

    out = tmp_path / 'saved.mat'
    MatSaver.save(em, str(out))
    _, class_name = mcos_reader.find_ts_object_stub(loadmat(str(out)))
    assert class_name == 'TS'

    em2 = MatLoader.load(str(out))
    assert em2.npts == em.npts and em2.NCh == em.NCh
    actual = np.stack([em2.data[ch].to_numpy() for ch in em2.channel_names()])
    expected = np.stack([em.data[ch].to_numpy() for ch in em.channel_names()])
    assert np.allclose(actual, expected, equal_nan=True)


@needs_fixture
@needs_converter
def test_load_ts_object_mat_via_converter_fallback(monkeypatch):
    """纯 Python 解析不可用时，回退到 mtpro_ts_convert 可执行文件。"""
    monkeypatch.setattr(mcos_reader, 'extract_ts_struct', lambda mat_data: None)
    em = MatLoader.load(str(TS_OBJECT_MAT))
    assert em.name == '039BE'
    assert em.npts == N_KEEP
    actual = np.stack([em.data[ch].to_numpy() for ch in em.channel_names()])
    assert np.allclose(actual, _expected_data(), equal_nan=True)


@needs_converter
def test_save_mat_writes_ts_object(tmp_path):
    """MatSaver.save 默认写 TS 对象格式，且可往返读取。"""
    em = MatLoader.load(str(STRUCT_MAT))
    out = tmp_path / "saved.mat"
    MatSaver.save(em, str(out))

    # 输出应为 TS 对象文件（用版本无关的检测助手，兼容 scipy 新旧布局）
    mat = loadmat(str(out))
    _, class_name = mcos_reader.find_ts_object_stub(mat)
    assert class_name == 'TS'

    # 往返读取：保存的正是 update_meta 后的数据
    expected = np.stack([em.data[ch].to_numpy() for ch in em.channel_names()])
    em2 = MatLoader.load(str(out))
    assert em2.name == em.name
    assert em2.npts == em.npts
    assert em2.NCh == em.NCh
    actual = np.stack([em2.data[ch].to_numpy() for ch in em2.channel_names()])
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, equal_nan=True)


@needs_fixture
@needs_converter
def test_convert_to_struct(tmp_path):
    """直接调用转换器：TS 对象 → tsStruct，字段可由 scipy 读取。"""
    out = ts_convert.convert_to_struct(str(TS_OBJECT_MAT), tmp_path / "out.mat")
    ts = loadmat(str(out))['tsStruct']
    assert ts['name'][0, 0][0] == '039BE'
    assert int(ts['npts'][0, 0][0, 0]) == N_KEEP
    assert int(ts['NCh'][0, 0][0, 0]) == 7


@needs_converter
def test_convert_rejects_plain_struct_file(tmp_path):
    """显式 ts2struct 作用于普通 tsStruct 文件时应报错（而非静默通过）。"""
    with pytest.raises(RuntimeError, match="mtpro_ts_convert failed"):
        ts_convert.convert_to_struct(str(STRUCT_MAT), tmp_path / "out.mat")


def test_find_converter_missing_reports_searched_paths(monkeypatch):
    """转换器缺失时报错信息应包含搜索路径与构建指引。"""
    monkeypatch.setenv("MTPRO_TS_CONVERT", "/nonexistent/mtpro_ts_convert")
    real_is_file = Path.is_file
    monkeypatch.setattr(
        Path, "is_file",
        lambda self: False if self.name.startswith("mtpro_ts_convert") else real_is_file(self),
    )
    with pytest.raises(FileNotFoundError, match="build_matlab.sh"):
        ts_convert.find_converter()
