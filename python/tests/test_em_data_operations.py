"""EMData 就地预处理操作（元数据/裁剪/降采样/通道管理）的单元测试。

MatEMData 的保存-重载回读测试在本文件后半部分，需要
fixtures/ts_object.mat（缺失时跳过）。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.em_data import EMData

FIXTURE_DIR = Path(__file__).parent / "fixtures"
TS_OBJECT_MAT = FIXTURE_DIR / "ts_object.mat"
needs_ts_fixture = pytest.mark.skipif(
    not TS_OBJECT_MAT.exists(), reason="需要 fixtures/ts_object.mat"
)


class DemoData(EMData):
    """最小可用的 EMData 存根：三通道、1s 采样。"""

    def __init__(self, npts: int = 100, channels=("Ex", "Hy", "Bz")):
        super().__init__("/tmp/demo.mat")
        self.name = "demo"
        self.timezone = "Asia/Shanghai"
        self.dt = pd.Timedelta(seconds=1)
        self.start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        self.end_time = self.start_time + self.dt * (npts - 1)
        self.datetime_index = pd.date_range(
            start=self.start_time, periods=npts, freq=self.dt,
        )
        self.npts = npts
        self.kp_data = None
        self.e_units = "mV/km"
        self.m_units = "nT"
        self.latitude = 30.0
        self.longitude = 120.0
        self.elevation = 50.0

        rng = np.random.default_rng(7)
        self.data = xr.Dataset(
            {
                ch: (("time",), rng.standard_normal(npts))
                for ch in channels
            },
            coords={"time": self.datetime_index},
        )

    def restore_data(self, ch: str):
        return None


# ── set_metadata ──────────────────────────────────────────────


class TestSetMetadata:
    def test_updates_given_fields(self):
        d = DemoData()
        d.set_metadata(name="NEW", latitude=31.5, m_units="pT")
        assert d.name == "NEW"
        assert d.latitude == 31.5
        assert d.m_units == "pT"
        # 未给定的保持原值
        assert d.longitude == 120.0
        assert d.e_units == "mV/km"

    def test_none_keeps_original(self):
        d = DemoData()
        d.set_metadata()
        assert d.name == "demo"
        assert d.latitude == 30.0

    def test_non_structural(self):
        d = DemoData()
        npts_before = d.npts
        idx_before = d.datetime_index
        d.set_metadata(name="X")
        assert d.npts == npts_before
        assert d.datetime_index is idx_before


# ── trim_time_range ───────────────────────────────────────────


class TestTrimTimeRange:
    def test_trim_middle(self):
        d = DemoData(100)
        start = d.start_time + pd.Timedelta(seconds=10)
        end = d.start_time + pd.Timedelta(seconds=19)
        new_npts = d.trim_time_range(start, end)

        assert new_npts == 10
        assert d.npts == 10
        assert d.start_time == start
        assert d.end_time == end
        assert len(d.datetime_index) == 10
        for ch in d.channel_names():
            assert len(d.data[ch]) == 10
        # dt 不变
        assert d.dt == pd.Timedelta(seconds=1)

    def test_trim_head_only(self):
        d = DemoData(100)
        start = d.start_time + pd.Timedelta(seconds=90)
        d.trim_time_range(start=start)
        assert d.npts == 10
        assert d.start_time == start
        assert d.end_time == d.start_time + d.dt * 9

    def test_trim_tail_only(self):
        d = DemoData(100)
        end = d.start_time + pd.Timedelta(seconds=9)
        d.trim_time_range(end=end)
        assert d.npts == 10
        assert d.end_time == end

    def test_empty_range_raises(self):
        d = DemoData(100)
        with pytest.raises(ValueError):
            d.trim_time_range(
                d.start_time - pd.Timedelta(hours=2),
                d.start_time - pd.Timedelta(hours=1),
            )

    def test_snaps_to_samples(self):
        # 起止不对齐采样点：向数据范围对齐（searchsorted 语义）
        d = DemoData(100)
        start = d.start_time + pd.Timedelta(milliseconds=1500)  # 1.5s → 采样点 2
        end = d.start_time + pd.Timedelta(milliseconds=4500)  # 4.5s → 含采样点 4
        d.trim_time_range(start, end)
        assert d.npts == 3
        assert d.start_time == d.datetime_index[0]

    def test_preserves_nan_gaps(self):
        d = DemoData(100)
        d.data["Ex"][30:40] = np.nan
        d.trim_time_range(d.start_time, d.start_time + pd.Timedelta(seconds=50))
        ex = d.data["Ex"].to_numpy()
        assert np.isnan(ex[30:40]).all()
        assert np.isfinite(ex[:30]).all()

    def test_kp_sliced(self):
        d = DemoData(100)
        d.kp_data = pd.DataFrame(
            {
                "Kp_datetime": pd.date_range(
                    d.start_time, periods=10, freq="10s",
                ),
                "Kp": np.arange(10, dtype=np.float64),
            }
        )
        d.trim_time_range(
            d.start_time + pd.Timedelta(seconds=20),
            d.start_time + pd.Timedelta(seconds=50),
        )
        assert len(d.kp_data) == 4  # 20s, 30s, 40s, 50s
        assert d.kp_data["Kp"].iloc[0] == 2.0

    def test_kp_none_is_safe(self):
        d = DemoData(100)
        d.kp_data = None
        d.trim_time_range(d.start_time + pd.Timedelta(seconds=10))
        assert d.kp_data is None

    def test_empty_dataset_raises(self):
        d = DemoData(100)
        d.data = xr.Dataset(coords={"time": pd.DatetimeIndex([])})
        d.datetime_index = pd.DatetimeIndex([])
        d.npts = 0
        with pytest.raises(ValueError):
            d.trim_time_range(None, None)


# ── decimate ──────────────────────────────────────────────────


class TestDecimate:
    def test_basic(self):
        d = DemoData(100)
        new_npts = d.decimate(5)

        assert new_npts == 20
        assert d.npts == 20
        assert d.dt == pd.Timedelta(seconds=5)
        assert len(d.datetime_index) == 20
        assert d.datetime_index[0] == d.start_time
        # 时间轴等于原索引 [::5]
        original = pd.date_range(
            start=d.start_time, periods=100, freq="1s",
        )
        assert d.datetime_index.equals(original[::5])
        for ch in d.channel_names():
            assert len(d.data[ch]) == 20

    def test_nan_gap_preserved(self):
        d = DemoData(100)
        d.data["Ex"][10:15] = np.nan  # 完整落在第 2 个窗口 [10,15)
        d.decimate(5)
        ex = d.data["Ex"].to_numpy()
        assert np.isnan(ex[2])
        assert np.isfinite(ex[0]) and np.isfinite(ex[1]) and np.isfinite(ex[3])

    def test_all_nan_channel(self):
        d = DemoData(100)
        d.data["Hy"][:] = np.nan
        d.decimate(5)
        assert np.isnan(d.data["Hy"].to_numpy()).all()
        assert d.npts == 20

    def test_factor_validation(self):
        d = DemoData(100)
        for bad in (1, 0, -2, 2.5, True):
            with pytest.raises(ValueError):
                d.decimate(bad)

    def test_factor_larger_than_npts(self):
        d = DemoData(3)
        with pytest.raises(ValueError):
            d.decimate(5)

    def test_non_divisible(self):
        d = DemoData(103)
        d.decimate(5)
        assert d.npts == 21

    def test_kp_unchanged(self):
        d = DemoData(100)
        kp = pd.DataFrame(
            {"Kp_datetime": [d.start_time], "Kp": [3.0]},
        )
        d.kp_data = kp
        d.decimate(5)
        assert d.kp_data is kp


# ── rename_channel / remove_channel ───────────────────────────


class TestChannelManagement:
    def test_rename_basic(self):
        d = DemoData()
        d.rename_channel("Ex", "Ex1")
        assert "Ex1" in d.chid
        assert "Ex" not in d.chid
        assert len(d.get_channel_series("Ex1")) == d.npts

    def test_rename_preserves_order(self):
        d = DemoData()
        d.rename_channel("Hy", "By")
        assert d.chid == ["Ex", "By", "Bz"]

    def test_rename_collision_raises(self):
        d = DemoData()
        with pytest.raises(ValueError):
            d.rename_channel("Ex", "Hy")

    def test_rename_missing_raises(self):
        d = DemoData()
        with pytest.raises(KeyError):
            d.rename_channel("Nope", "X")

    def test_rename_empty_name_raises(self):
        d = DemoData()
        with pytest.raises(ValueError):
            d.rename_channel("Ex", "")

    def test_rename_noop_same_name(self):
        d = DemoData()
        d.rename_channel("Ex", "Ex")
        assert d.chid == ["Ex", "Hy", "Bz"]

    def test_remove_basic(self):
        d = DemoData()
        d.remove_channel("Hy")
        assert d.chid == ["Ex", "Bz"]
        assert d.NCh == 2

    def test_remove_missing_raises(self):
        d = DemoData()
        with pytest.raises(KeyError):
            d.remove_channel("Nope")

    def test_remove_last_channel_raises(self):
        d = DemoData(channels=("Ex",))
        with pytest.raises(ValueError):
            d.remove_channel("Ex")
        assert d.chid == ["Ex"]


# ── MatEMData：update_meta 回写与保存-重载往返 ────────────────
# 需要 fixtures/ts_object.mat（由 matlab/make_test_fixture.m 生成）。
# fetch_kp 被 monkeypatch 为 None，避免测试触发网络请求。


def _load_fixture(monkeypatch):
    from core import mat_data as mat_data_mod
    from io_utils.mat_io import MatLoader

    monkeypatch.setattr(mat_data_mod, "fetch_kp", lambda *args, **kwargs: None)
    return MatLoader.load(str(TS_OBJECT_MAT))


@needs_ts_fixture
def test_update_meta_syncs_metadata_fields(monkeypatch):
    em = _load_fixture(monkeypatch)

    em.set_metadata(
        name="NEWSTA", latitude=31.25, longitude=121.5,
        elevation=12.0, e_units="uV/km", m_units="pT",
    )
    em.update_meta()

    ts = em.mat_meta[em._struct_key]
    assert ts["name"][0, 0][0] == "NEWSTA"
    assert float(ts["latitude"][0, 0][0, 0]) == pytest.approx(31.25)
    assert float(ts["longitude"][0, 0][0, 0]) == pytest.approx(121.5)
    assert float(ts["elevation"][0, 0][0]) == pytest.approx(12.0)
    units = ts["units"][0, 0]
    assert units.flat[0][0] == "pT"
    assert units.flat[1][0] == "uV/km"
    # 未改动的字段不受影响
    assert int(ts["npts"][0, 0][0, 0]) == em.npts


@needs_ts_fixture
def test_update_meta_syncs_chid_after_rename(monkeypatch):
    em = _load_fixture(monkeypatch)
    em.rename_channel("Bx", "Hx")
    em.update_meta()

    ts = em.mat_meta[em._struct_key]
    chid_row = ts["chid"][0, 0][0]
    assert [c[0] for c in chid_row] == em.chid
    assert "Hx" in em.chid and "Bx" not in em.chid


@needs_ts_fixture
def test_save_reload_metadata_round_trip(tmp_path, monkeypatch):
    from io_utils.mat_io import MatLoader, MatSaver

    em = _load_fixture(monkeypatch)
    em.set_metadata(name="RTSTA", latitude=40.0, longitude=116.0, m_units="pT")
    em.rename_channel("Ex1", "ExA")

    out = tmp_path / "meta_rt.mat"
    MatSaver.save(em, str(out))

    em2 = MatLoader.load(str(out))
    assert em2.name == "RTSTA"
    assert em2.latitude == pytest.approx(40.0)
    assert em2.longitude == pytest.approx(116.0)
    assert em2.m_units == "pT"
    assert em2.e_units == em.e_units
    assert list(em2.chid) == list(em.chid)


@needs_ts_fixture
def test_save_reload_decimate_round_trip(tmp_path, monkeypatch):
    from io_utils.mat_io import MatLoader, MatSaver

    em = _load_fixture(monkeypatch)
    assert em.dt == pd.Timedelta(seconds=5)
    em.decimate(5)

    out = tmp_path / "dec_rt.mat"
    MatSaver.save(em, str(out))

    em2 = MatLoader.load(str(out))
    # update_meta 可能在保存时裁剪首尾 NaN，以保存后的 em 状态为准
    assert em2.npts == em.npts
    assert em2.dt == em.dt
    assert em2.dt == pd.Timedelta(seconds=25)
    assert list(em2.chid) == list(em.chid)
    for ch in em.channel_names():
        np.testing.assert_allclose(
            em2.get_channel_series(ch).to_numpy(),
            em.get_channel_series(ch).to_numpy(),
            equal_nan=True,
        )


@needs_ts_fixture
def test_save_reload_remove_channel_round_trip(tmp_path, monkeypatch):
    from io_utils.mat_io import MatLoader, MatSaver

    em = _load_fixture(monkeypatch)
    em.remove_channel("Ex2")
    em.remove_channel("Ey2")

    out = tmp_path / "rmch_rt.mat"
    MatSaver.save(em, str(out))

    em2 = MatLoader.load(str(out))
    assert list(em2.chid) == ["Bx", "By", "Bz", "Ex1", "Ey1"]
    assert em2.NCh == 5
    assert em2.npts == em.npts


@needs_ts_fixture
def test_save_reload_trim_round_trip(tmp_path, monkeypatch):
    from io_utils.mat_io import MatLoader, MatSaver

    em = _load_fixture(monkeypatch)
    new_start = em.start_time + pd.Timedelta(seconds=500)
    new_end = em.end_time - pd.Timedelta(seconds=500)
    em.trim_time_range(new_start, new_end)

    out = tmp_path / "trim_rt.mat"
    MatSaver.save(em, str(out))

    em2 = MatLoader.load(str(out))
    assert em2.npts == em.npts
    assert em2.dt == em.dt
    assert em2.start_time == em.start_time
    assert em2.end_time == em.end_time
