"""数据树右键菜单结构与预处理器接线的测试（offscreen Qt）。"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import xarray as xr
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from base.data_manager import DataManager
from core.em_data import EMData
from ui.data_tree_view import DataTreeViewer


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class DemoData(EMData):
    def __init__(self, channels=("Ex", "Hy", "Bz")):
        super().__init__("/tmp/demo.mat")
        self.name = "demo"
        self.timezone = "Asia/Shanghai"
        self.dt = pd.Timedelta(seconds=1)
        self.start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        self.npts = 50
        self.end_time = self.start_time + self.dt * (self.npts - 1)
        self.datetime_index = pd.date_range(
            start=self.start_time, periods=self.npts, freq=self.dt,
        )
        self.kp_data = None
        self.e_units = "mV/km"
        self.m_units = "nT"
        self.latitude = 30.0
        self.longitude = 120.0
        self.elevation = 50.0

        rng = np.random.default_rng(11)
        self.data = xr.Dataset(
            {ch: (("time",), rng.standard_normal(self.npts)) for ch in channels},
            coords={"time": self.datetime_index},
        )

    def restore_data(self, ch: str):
        return None


def _menu_texts(menu) -> list[str]:
    return [a.text() for a in menu.actions() if not a.isSeparator()]


def test_dataset_menu_contains_preprocessing_items():
    _app()
    viewer = DataTreeViewer(DataManager())
    menu, actions = viewer._build_dataset_context_menu()

    texts = _menu_texts(menu)
    assert texts == [
        "保存", "另存为", "关闭数据",
        "编辑台站元数据…", "降采样…", "裁剪时间范围…", "通道管理…",
    ]
    # 每个 action 都有处理器
    assert all(callable(actions[a]) for a in menu.actions() if not a.isSeparator())
    menu.close()
    viewer.close()


def test_channel_menu_contains_channel_items():
    _app()
    viewer = DataTreeViewer(DataManager())
    menu, actions = viewer._build_channel_context_menu()

    texts = _menu_texts(menu)
    assert texts == ["显示曲线", "序列对比", "重命名通道…", "删除通道"]
    assert all(callable(actions[a]) for a in menu.actions() if not a.isSeparator())
    menu.close()
    viewer.close()


def test_dataset_menu_dispatches_to_handler():
    _app()
    viewer = DataTreeViewer(DataManager())

    called = []
    viewer._open_decimate_dialog = lambda data_id: called.append(data_id)

    menu, actions = viewer._build_dataset_context_menu()
    target = next(a for a in menu.actions() if a.text() == "降采样…")
    actions[target](42)

    assert called == [42]
    menu.close()
    viewer.close()


def test_confirm_delete_channel_emits_structural_change(monkeypatch):
    _app()
    dm = DataManager()
    key = dm.add(DemoData())
    viewer = DataTreeViewer(dm)

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Yes),
    )
    received = []
    dm.data_changed.connect(lambda k, s: received.append((k, s)))

    viewer._confirm_delete_channel(key, "Hy")

    assert received == [(key, True)]
    assert list(dm.get(key).chid) == ["Ex", "Bz"]
    viewer.close()


def test_confirm_delete_channel_refuses_cancel(monkeypatch):
    _app()
    dm = DataManager()
    key = dm.add(DemoData())
    viewer = DataTreeViewer(dm)

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.No),
    )
    received = []
    dm.data_changed.connect(lambda k, s: received.append((k, s)))

    viewer._confirm_delete_channel(key, "Hy")

    assert received == []
    assert list(dm.get(key).chid) == ["Ex", "Hy", "Bz"]
    viewer.close()


def test_confirm_delete_last_channel_warns(monkeypatch):
    _app()
    dm = DataManager()
    key = dm.add(DemoData(channels=("Ex",)))
    viewer = DataTreeViewer(dm)

    warnings = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *args, **kwargs: warnings.append(args)),
    )

    viewer._confirm_delete_channel(key, "Ex")

    assert warnings, "应提示不能删除最后一个通道"
    assert list(dm.get(key).chid) == ["Ex"]
    viewer.close()


def test_rename_channel_handler_emits_structural_change(monkeypatch):
    _app()
    dm = DataManager()
    key = dm.add(DemoData())
    viewer = DataTreeViewer(dm)

    monkeypatch.setattr(
        "ui.data_tree_view.RenameChannelDialog",
        _FakeRenameDialog("Ex1"),
    )
    received = []
    dm.data_changed.connect(lambda k, s: received.append((k, s)))

    viewer._open_rename_channel_dialog(key, "Ex")

    assert received == [(key, True)]
    assert list(dm.get(key).chid) == ["Ex1", "Hy", "Bz"]
    viewer.close()


class _FakeRenameDialog:
    """替换 RenameChannelDialog：直接给定新名并接受。"""

    def __init__(self, new_name: str):
        self._new_name = new_name

    def __call__(self, em_data, old_name, parent=None):
        return self

    def exec(self):
        return QDialog.DialogCode.Accepted

    def new_name(self) -> str:
        return self._new_name
