"""预处理对话框的行为测试（offscreen Qt，不弹窗）。"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import xarray as xr
from PySide6.QtWidgets import QApplication

from core.em_data import EMData
from ui.dialogs.channel_management_dialog import ChannelManagementDialog
from ui.dialogs.decimate_dialog import DecimateDialog
from ui.dialogs.rename_channel_dialog import RenameChannelDialog
from ui.dialogs.station_metadata_dialog import StationMetadataDialog
from ui.dialogs.trim_dialog import TrimDialog
from ui.widgets.qt_time import pts_to_qdt


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class DemoData(EMData):
    """最小 EMData 存根：100 点 1s 采样，三通道。"""

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

        rng = np.random.default_rng(3)
        self.data = xr.Dataset(
            {ch: (("time",), rng.standard_normal(npts)) for ch in channels},
            coords={"time": self.datetime_index},
        )

    def restore_data(self, ch: str):
        return None


def _spy_accept(dialog):
    """把 dialog.accept 替换为记录器，返回已调用列表。"""
    calls = []
    dialog.accept = lambda: calls.append(True)
    return calls


# ── StationMetadataDialog ─────────────────────────────────────


def test_station_metadata_dialog_apply():
    _app()
    em = DemoData()
    dlg = StationMetadataDialog(em)

    dlg._name_edit.setText("NEWSTA")
    dlg._lat_spin.setValue(31.5)
    dlg._lon_spin.setValue(121.25)
    dlg._elev_spin.setValue(12.0)
    dlg._e_units_edit.setText("uV/km")
    dlg._m_units_edit.setText("pT")

    calls = _spy_accept(dlg)
    dlg._on_accept()
    assert calls, "有效输入应接受"

    dlg.apply_to(em)
    assert em.name == "NEWSTA"
    assert em.latitude == 31.5
    assert em.longitude == 121.25
    assert em.elevation == 12.0
    assert em.e_units == "uV/km"
    assert em.m_units == "pT"
    dlg.close()


def test_station_metadata_dialog_rejects_empty_name():
    _app()
    dlg = StationMetadataDialog(DemoData())
    dlg._name_edit.setText("   ")
    calls = _spy_accept(dlg)
    dlg._on_accept()
    assert not calls
    dlg.close()


def test_station_metadata_dialog_keeps_unset_coordinates():
    _app()
    em = DemoData()
    em.latitude = None
    dlg = StationMetadataDialog(em)

    # 纬度字段应为空（哨兵值），只改名称
    dlg._name_edit.setText("ONLYNAME")
    calls = _spy_accept(dlg)
    dlg._on_accept()
    assert calls
    dlg.apply_to(em)
    assert em.name == "ONLYNAME"
    assert em.latitude is None
    dlg.close()


# ── DecimateDialog ────────────────────────────────────────────


def test_decimate_dialog_preview_updates():
    _app()
    em = DemoData(100)
    dlg = DecimateDialog(em)

    dlg._factor_spin.setValue(5)
    preview = dlg._preview_label.text()
    assert "5s" in preview or "5 s" in preview.replace("\xa0", " ")
    assert "20 点" in preview
    assert dlg.factor() == 5
    dlg.close()


def test_decimate_dialog_rejects_excessive_factor():
    _app()
    dlg = DecimateDialog(DemoData(100))
    dlg._factor_spin.setValue(20)  # 100 → 5 点 < 10
    calls = _spy_accept(dlg)
    dlg._on_accept()
    assert not calls
    dlg.close()


def test_decimate_dialog_accepts_valid_factor():
    _app()
    dlg = DecimateDialog(DemoData(100))
    dlg._factor_spin.setValue(5)
    calls = _spy_accept(dlg)
    dlg._on_accept()
    assert calls
    dlg.close()


def test_decimate_dialog_preview_shows_limit_hint(monkeypatch):
    """预览应标注与建议上限（与 xlsx 导入共用）的比较及建议因子。"""
    _app()
    import ui.dialogs.decimate_dialog as dd

    monkeypatch.setattr(dd, "MAX_POINTS_PER_CHANNEL", 30)
    dlg = DecimateDialog(DemoData(100))

    dlg._factor_spin.setValue(2)  # 100 → 50 点 > 30：超限 + 建议因子 ceil(99/29)=4
    preview = dlg._preview_label.text()
    assert "仍高于建议上限" in preview
    assert "≥ 4" in preview

    dlg._factor_spin.setValue(4)  # 100 → 25 点 ≤ 30：达标
    preview = dlg._preview_label.text()
    assert "低于建议上限" in preview
    dlg.close()


# ── TrimDialog ────────────────────────────────────────────────


def test_trim_dialog_time_range_round_trip():
    _app()
    em = DemoData(100)
    dlg = TrimDialog(em)

    new_start = em.start_time + pd.Timedelta(seconds=10)
    new_end = em.start_time + pd.Timedelta(seconds=50)
    dlg._start_edit.setDateTime(pts_to_qdt(new_start, em.timezone))
    dlg._end_edit.setDateTime(pts_to_qdt(new_end, em.timezone))

    start, end = dlg.time_range()
    # 显示按台站时区，绝对时刻应与设定一致
    assert start == new_start
    assert end == new_end

    calls = _spy_accept(dlg)
    dlg._on_accept()
    assert calls
    dlg.close()


def test_trim_dialog_rejects_empty_range():
    _app()
    em = DemoData(100)
    dlg = TrimDialog(em)

    mid = em.start_time + pd.Timedelta(seconds=50)
    dlg._start_edit.setDateTime(pts_to_qdt(mid, em.timezone))
    dlg._end_edit.setDateTime(pts_to_qdt(mid, em.timezone))

    calls = _spy_accept(dlg)
    dlg._on_accept()
    assert not calls
    dlg.close()


def test_trim_dialog_applies_to_em_data():
    _app()
    em = DemoData(100)
    dlg = TrimDialog(em)

    new_start = em.start_time + pd.Timedelta(seconds=10)
    new_end = em.start_time + pd.Timedelta(seconds=49)
    dlg._start_edit.setDateTime(pts_to_qdt(new_start, em.timezone))
    dlg._end_edit.setDateTime(pts_to_qdt(new_end, em.timezone))

    start, end = dlg.time_range()
    em.trim_time_range(start, end)
    assert em.npts == 40
    assert em.start_time == new_start
    assert em.end_time == new_end
    dlg.close()


# ── RenameChannelDialog ───────────────────────────────────────


def test_rename_dialog_accepts_new_name():
    _app()
    em = DemoData()
    dlg = RenameChannelDialog(em, "Ex")
    dlg._name_edit.setText("Ex1")
    calls = _spy_accept(dlg)
    dlg._on_accept()
    assert calls
    assert dlg.new_name() == "Ex1"
    dlg.close()


def test_rename_dialog_rejects_collision():
    _app()
    em = DemoData()
    dlg = RenameChannelDialog(em, "Ex")
    dlg._name_edit.setText("Hy")
    calls = _spy_accept(dlg)
    dlg._on_accept()
    assert not calls
    dlg.close()


def test_rename_dialog_rejects_empty():
    _app()
    dlg = RenameChannelDialog(DemoData(), "Ex")
    dlg._name_edit.setText("  ")
    calls = _spy_accept(dlg)
    dlg._on_accept()
    assert not calls
    dlg.close()


# ── ChannelManagementDialog ───────────────────────────────────


def _fill_table(dlg, names):
    """把表格各行通道名设为 names（与原始行数一致）。"""
    for row, name in enumerate(names):
        dlg._table.item(row, 0).setText(name)


def test_channel_management_applies_rename_and_delete():
    _app()
    em = DemoData(channels=("Ex", "Hy", "Bz"))
    dlg = ChannelManagementDialog(em)

    _fill_table(dlg, ["Ex1", "Hy", "Bz"])
    dlg._delete_checks[2].setChecked(True)  # 删除 Bz

    applied_renames, applied_deletes = dlg.apply_to(em)
    assert (applied_renames, applied_deletes) == (1, 1)
    assert list(em.chid) == ["Ex1", "Hy"]
    dlg.close()


def test_channel_management_chain_rename_ordering():
    """Ex→Hy 且 Hy→Hz：需先改 Hy→Hz 再改 Ex→Hy，apply 应能完成。"""
    _app()
    em = DemoData(channels=("Ex", "Hy", "Bz"))
    dlg = ChannelManagementDialog(em)

    _fill_table(dlg, ["Hy", "Hz", "Bz"])  # Ex→Hy, Hy→Hz

    calls = _spy_accept(dlg)
    dlg._on_accept()
    assert calls

    applied_renames, applied_deletes = dlg.apply_to(em)
    assert (applied_renames, applied_deletes) == (2, 0)
    assert list(em.chid) == ["Hy", "Hz", "Bz"]
    dlg.close()


def test_channel_management_rejects_cycle():
    _app()
    em = DemoData(channels=("Ex", "Hy", "Bz"))
    dlg = ChannelManagementDialog(em)

    _fill_table(dlg, ["Hy", "Ex", "Bz"])  # Ex↔Hy 互换
    calls = _spy_accept(dlg)
    dlg._on_accept()
    assert not calls
    dlg.close()


def test_channel_management_rejects_all_deleted():
    _app()
    dlg = ChannelManagementDialog(DemoData(channels=("Ex",)))
    dlg._delete_checks[0].setChecked(True)
    calls = _spy_accept(dlg)
    dlg._on_accept()
    assert not calls
    dlg.close()


def test_channel_management_rejects_duplicate_names():
    _app()
    dlg = ChannelManagementDialog(DemoData())
    _fill_table(dlg, ["A", "A", "Bz"])
    calls = _spy_accept(dlg)
    dlg._on_accept()
    assert not calls
    dlg.close()
