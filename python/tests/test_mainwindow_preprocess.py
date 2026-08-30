"""MainWindow.on_data_changed 路由的集成测试（offscreen Qt）。

验证就地预处理（结构/元数据变更）经 DataManager.data_changed
到达主窗口后的刷新行为：结构变更重建活跃视图、元数据变更只
刷新摘要、非活跃数据集的变更推迟到下次激活。
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import xarray as xr
from PySide6.QtWidgets import QApplication

from base.data_manager import DataManager
from core.em_data import EMData
from ui.mainwindow import MainWindow


class DemoData(EMData):
    def __init__(self, name: str, channels: list[str], npts: int = 20):
        super().__init__("/tmp/demo.mat")
        self.name = name
        self.timezone = "Asia/Shanghai"
        self.npts = npts
        self.start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        self.dt = pd.Timedelta(seconds=5)
        self.datetime_index = pd.date_range(
            start=self.start_time, periods=npts, freq=self.dt,
        )
        self.end_time = self.datetime_index[-1]
        self.kp_data = None
        self.e_units = "mV/km"
        self.m_units = "nT"

        rng = np.random.default_rng(5)
        self.data = xr.Dataset(
            {ch: (("time",), rng.standard_normal(npts)) for ch in channels},
            coords={"time": self.datetime_index},
        )

    def restore_data(self, ch: str):
        return None


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_structural_change_rebuilds_active_viewer():
    _app()
    manager = DataManager()
    data = DemoData("demo", ["Ex1", "Hy1"], npts=20)
    key = manager.add(data)

    window = MainWindow(manager)
    manager.set_active(key)
    old_viewer = window.data_viewers.get(key)
    assert old_viewer is not None

    # 就地降采样后发出结构变更
    data.decimate(2)
    assert data.npts == 10
    manager.notify_changed(key, structural=True)

    new_viewer = window.data_viewers.get(key)
    assert new_viewer is not None
    assert new_viewer is not old_viewer
    assert new_viewer.em_data is data
    assert window.data_view_stack.currentWidget() is new_viewer
    # 处理页上下文同步
    assert data.dt == pd.Timedelta(seconds=10)
    window.close()


def test_metadata_change_keeps_viewer_instance():
    _app()
    manager = DataManager()
    data = DemoData("demo", ["Ex1", "Hy1"])
    key = manager.add(data)

    window = MainWindow(manager)
    manager.set_active(key)
    viewer = window.data_viewers.get(key)
    assert viewer is not None

    data.set_metadata(name="新台站名")
    manager.notify_changed(key, structural=False)

    # viewer 实例不变（不重建金字塔）
    assert window.data_viewers.get(key) is viewer
    # 状态栏刷新为新名称与新样点数
    status = window.dataset_status_label.text()
    assert "新台站名" in status
    assert str(data.npts) in status
    window.close()


def test_structural_change_on_inactive_dataset_defers_rebuild():
    _app()
    manager = DataManager()
    data1 = DemoData("one", ["Ex1"], npts=20)
    data2 = DemoData("two", ["Ex1"], npts=20)
    key1 = manager.add(data1)
    key2 = manager.add(data2)

    window = MainWindow(manager)
    manager.set_active(key1)
    assert key2 not in window.data_viewers  # active-only：非活跃无 viewer

    data2.decimate(5)
    manager.notify_changed(key2, structural=True)

    # 仍未创建 viewer，旧现场快照（若有）被丢弃
    assert key2 not in window.data_viewers
    assert key2 not in window.viewer_state

    # 激活后按新数据构建
    manager.set_active(key2)
    viewer = window.data_viewers.get(key2)
    assert viewer is not None
    assert viewer.em_data is data2
    assert data2.npts == 4
    window.close()


def test_structural_change_preserves_view_state_of_active_viewer():
    """重建 viewer 后应恢复此前的视口现场（经 viewer_state 快照）。"""
    _app()
    manager = DataManager()
    data = DemoData("demo", ["Ex1", "Hy1"], npts=20)
    key = manager.add(data)

    window = MainWindow(manager)
    manager.set_active(key)
    viewer = window.data_viewers.get(key)
    assert viewer is not None

    # 改变视口现场
    viewer.x_view_size = pd.Timedelta(seconds=50)
    viewer.x_view_start = pd.Timedelta(seconds=0)

    manager.notify_changed(key, structural=True)

    new_viewer = window.data_viewers.get(key)
    assert new_viewer is not viewer
    # 视口尺寸现场被快照并恢复（受数据时长夹取）
    assert new_viewer.x_view_size == pd.Timedelta(seconds=50)
    window.close()
