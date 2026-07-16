#!/usr/bin/env python3
"""
MT时间序列异常样本采集工具

人工提取大地电磁时间序列并识别异常（正常/突变/阶跃/趋势/瞬态），
以1024点固定窗口截取波形，存储为HDF5格式用于AI训练。

用法: python mt_anomaly_collector.py
"""

from __future__ import annotations

import os
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import sys
import numpy as np
import pandas as pd
import h5py
import pytz
from scipy.io import loadmat
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QGroupBox, QSplitter, QSizePolicy,
    QAbstractItemView, QStatusBar, QStyle, QSpinBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut

import pyqtgraph as pg

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

WINDOW_SIZE = 1024
LABEL_INFO = {
    0: {"name": "正常", "color": "#27ae60"},
    1: {"name": "突变", "color": "#e74c3c"},
    2: {"name": "阶跃", "color": "#f39c12"},
    3: {"name": "趋势", "color": "#8e44ad"},
    4: {"name": "瞬态", "color": "#16a085"},
}

_TZ_CST_ = pytz.timezone("Asia/Shanghai")

# ---------------------------------------------------------------------------
# 自定义 PlotWidget（Shift 锁定轴向 + 双击跳转）
# ---------------------------------------------------------------------------


class ShiftablePlotWidget(pg.PlotWidget):
    """支持 Shift 轴向锁定和双击跳转 LinearRegionItem 的 PlotWidget."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._region_item: Optional[pg.LinearRegionItem] = None
        self._data_x0: Optional[float] = None
        self._data_x1: Optional[float] = None

    def set_region_reference(
        self, region: Optional[pg.LinearRegionItem], x_min: float, x_max: float
    ):
        self._region_item = region
        self._data_x0 = x_min
        self._data_x1 = x_max

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Shift and not event.isAutoRepeat():
            self.getViewBox().setMouseEnabled(x=True, y=False)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Shift and not event.isAutoRepeat():
            self.getViewBox().setMouseEnabled(x=True, y=True)
        super().keyReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if (event.button() == Qt.MouseButton.LeftButton
                and self._region_item is not None
                and self._data_x0 is not None):
            vb = self.getViewBox()
            scene_pos = self.mapToScene(event.pos())
            view_pos = vb.mapSceneToView(scene_pos)
            click_x = float(view_pos.x())

            r_left, r_right = self._region_item.getRegion()
            half = (r_right - r_left) / 2.0

            new_left = click_x - half
            new_right = click_x + half

            if new_left < self._data_x0:
                new_left = self._data_x0
                new_right = self._data_x0 + (r_right - r_left)
            if new_right > self._data_x1:
                new_right = self._data_x1
                new_left = self._data_x1 - (r_right - r_left)

            self._region_item.setRegion((new_left, new_right))
            return
        super().mouseDoubleClickEvent(event)


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------


class AnomalyCollector(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MT异常样本采集工具")
        self.resize(1280, 800)

        # ---- 数据状态 ----
        self._all_data: dict[str, np.ndarray] = {}     # ch_name → 1D array
        self.channel_names: list[str] = []
        self.timestamps: Optional[np.ndarray] = None    # Unix秒
        self.dt_sec: float = 0.0
        self.fixed_width_sec: float = 0.0               # window_size * dt
        self.window_size: int = WINDOW_SIZE              # 可调，默认 1024

        self.samples: list[dict] = []

        # ---- PyQtGraph items ----
        self.plot_curve: Optional[pg.PlotCurveItem] = None
        self.region_item: Optional[pg.LinearRegionItem] = None

        self._init_ui()
        self._connect_signals()
        self._init_shortcuts()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # -- 顶栏 --
        top_bar = QWidget()
        top_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        top = QHBoxLayout(top_bar)
        top.setContentsMargins(0, 0, 0, 0)

        self.btn_open = QPushButton("打开MAT文件")
        self.btn_open.setFixedHeight(24)
        top.addWidget(self.btn_open)

        self.btn_open_h5 = QPushButton("打开HDF5")
        self.btn_open_h5.setFixedHeight(24)
        top.addWidget(self.btn_open_h5)

        top.addWidget(QLabel("通道:"))
        self.combo_ch = QComboBox()
        self.combo_ch.setFixedWidth(140)
        self.combo_ch.setFixedHeight(24)
        self.combo_ch.setEnabled(False)
        top.addWidget(self.combo_ch)

        top.addWidget(QLabel("窗口:"))
        self.spin_window = QSpinBox()
        self.spin_window.setRange(256, 4096)
        self.spin_window.setSingleStep(64)
        self.spin_window.setValue(WINDOW_SIZE)
        self.spin_window.setFixedWidth(80)
        self.spin_window.setFixedHeight(24)
        self.spin_window.setEnabled(False)
        top.addWidget(self.spin_window)

        self.label_info = QLabel("请先打开MAT文件")
        top.addWidget(self.label_info)
        top.addStretch()

        self.label_pos = QLabel("")
        top.addWidget(self.label_pos)

        root.addWidget(top_bar)

        # -- 主体：绘图 + 右侧面板 --
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.plot_widget = ShiftablePlotWidget(axisItems={"bottom": pg.DateAxisItem()})
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.hideButtons()
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.setLabel("left", "幅值")
        self.plot_item.setLabel("bottom", "")
        self.plot_item.vb.disableAutoRange(axis="y")
        splitter.addWidget(self.plot_widget)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        rl.setSpacing(8)

        # 标注按钮组
        grp_btn = QGroupBox("异常类型标注")
        bl = QVBoxLayout(grp_btn)
        bl.setSpacing(5)

        self.label_btns: dict[int, QPushButton] = {}
        for lid, info in LABEL_INFO.items():
            btn = QPushButton(f"&{lid}  {info['name']}")
            btn.setFixedHeight(38)
            btn.setStyleSheet(self._btn_style(info["color"]))
            btn.setEnabled(False)
            self.label_btns[lid] = btn
            bl.addWidget(btn)

        rl.addWidget(grp_btn)

        # 样本列表
        grp_list = QGroupBox("已采集样本")
        ll = QVBoxLayout(grp_list)

        self.sample_list = QListWidget()
        self.sample_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.sample_list.setAlternatingRowColors(True)
        ll.addWidget(self.sample_list)

        bl2 = QHBoxLayout()
        self.btn_del = QPushButton("删除选中")
        self.btn_del.setEnabled(False)
        bl2.addWidget(self.btn_del)

        self.btn_save = QPushButton("保存 HDF5")
        self.btn_save.setFixedHeight(34)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet(self._btn_style("#2980b9"))
        bl2.addWidget(self.btn_save)

        ll.addLayout(bl2)
        rl.addWidget(grp_list)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

        # 状态栏
        self.status_label = QLabel("就绪")
        self.statusBar().addWidget(self.status_label)

    @staticmethod
    def _btn_style(base_color: str) -> str:
        return f"""
            QPushButton {{
                background-color: {base_color};
                color: white;
                font-weight: bold;
                font-size: 13px;
                border: none;
                border-radius: 4px;
            }}
            QPushButton:hover {{ opacity: 0.88; }}
            QPushButton:disabled {{ background-color: #bdc3c7; }}
        """

    # ------------------------------------------------------------------
    # 信号 / 快捷键
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.btn_open.clicked.connect(self._on_open)
        self.btn_open_h5.clicked.connect(self._on_open_h5)
        self.combo_ch.currentIndexChanged.connect(self._on_channel_changed)
        self.spin_window.valueChanged.connect(self._on_window_size_changed)
        self.btn_del.clicked.connect(self._on_delete)
        self.btn_save.clicked.connect(self._on_save)
        self.sample_list.itemSelectionChanged.connect(
            lambda: self.btn_del.setEnabled(len(self.sample_list.selectedItems()) > 0)
        )
        for lid, btn in self.label_btns.items():
            btn.clicked.connect(lambda _checked, lid=lid: self._on_collect(lid))

    def _init_shortcuts(self):
        for lid in LABEL_INFO:
            QShortcut(QKeySequence(str(lid)), self, activated=lambda lid=lid: self._on_collect(lid))

    # ------------------------------------------------------------------
    # 文件加载
    # ------------------------------------------------------------------

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开MAT文件", "", "MAT文件 (*.mat);;所有文件 (*)"
        )
        if not path:
            return

        try:
            mat = loadmat(path)
            ts = mat["tsStruct"]

            st = ts["startTime"][0, 0][0]
            self.dt_sec = float(ts["dt"][0, 0][0, 0])
            npts = int(ts["npts"][0, 0][0, 0])
            self.fixed_width_sec = self.window_size * self.dt_sec

            start_ts = pd.Timestamp(
                int(st[0]), int(st[1]), int(st[2]), int(st[3]),
                int(st[4]), int(st[5]), tz=_TZ_CST_,
            )
            self.timestamps = np.array(
                [start_ts.timestamp() + i * self.dt_sec for i in range(npts)],
                dtype=np.float64,
            )

            data = ts["data"][0, 0]  # (NCh, npts)
            chid = ts["chid"][0, 0][0]
            self.channel_names = [str(chid[i][0]) for i in range(len(chid))]

            self._all_data.clear()
            for i, ch in enumerate(self.channel_names):
                self._all_data[ch] = data[i, :].astype(np.float64, copy=False)

            name = str(ts["name"][0, 0][0])
            self.label_info.setText(
                f"台站: {name}  |  "
                f"点数: {npts}  |  "
                f"采样率: {1 / self.dt_sec:.1f} Hz  |  "
                f"通道数: {len(self.channel_names)}"
            )

            self.combo_ch.blockSignals(True)
            self.combo_ch.clear()
            self.combo_ch.addItems(self.channel_names)
            self.combo_ch.setEnabled(True)
            self.spin_window.setValue(self.window_size)
            self.spin_window.setEnabled(True)
            self.combo_ch.blockSignals(False)

            # 清除旧样本
            self.samples.clear()
            self._refresh_list()
            self.btn_save.setEnabled(False)

            self.status_label.setText(f"已加载: {os.path.basename(path)}")

            # 自动加载第一个通道
            if self.channel_names:
                self._load_channel(0)

        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"无法读取MAT文件:\n{e}")

    def _on_open_h5(self):
        """从已有 HDF5 文件加载样本，追加到当前列表中."""
        path, _ = QFileDialog.getOpenFileName(
            self, "打开HDF5文件", "", "HDF5文件 (*.h5 *.hdf5);;所有文件 (*)",
        )
        if not path:
            return

        try:
            with h5py.File(path, "r") as f:
                keys = sorted(k for k in f.keys() if k.startswith("sample_"))
                if not keys:
                    QMessageBox.warning(self, "无样本", "HDF5 中没有找到 sample_* 组。")
                    return

                loaded = 0
                for key in keys:
                    grp = f[key]
                    waveform = grp["waveform"][:].astype(np.float64)
                    label = int(grp["label"][()])
                    channel = str(grp.attrs.get("channel", "?"))
                    start_time = str(grp.attrs.get("start_time", "?"))
                    start_index = int(grp.attrs.get("start_index", -1))

                    self.samples.append({
                        "waveform": waveform,
                        "label": label,
                        "channel": channel,
                        "start_index": start_index,
                        "start_time": start_time,
                    })
                    loaded += 1

            self._refresh_list()
            self.btn_save.setEnabled(True)
            self.status_label.setText(
                f"已从 {os.path.basename(path)} 加载 {loaded} 个样本"
                f" (总计 {len(self.samples)} 个)"
            )
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"无法读取HDF5文件:\n{e}")

    # ------------------------------------------------------------------
    # 通道切换 & 绘图
    # ------------------------------------------------------------------

    def _on_channel_changed(self, idx: int):
        if idx >= 0:
            self._load_channel(idx)

    def _load_channel(self, idx: int):
        ch = self.channel_names[idx]
        y = self._all_data[ch]
        x = self.timestamps

        self.plot_widget.set_region_reference(None, 0.0, 0.0)
        self.plot_item.clear()
        self.plot_curve = None
        self.region_item = None

        # 自动 Y 范围
        ymin = float(np.nanmin(y))
        ymax = float(np.nanmax(y))
        yr = ymax - ymin or 1.0

        self.plot_curve = pg.PlotCurveItem(
            x, y,
            pen=pg.mkPen(color="b", width=1),
            clipToView=True,
            connect="finite",
            autoDownsample=True,
            downsampleMethod="peak",
        )
        self.plot_item.addItem(self.plot_curve)
        self.plot_item.setYRange(ymin - 0.05 * yr, ymax + 0.05 * yr, padding=0)
        self.plot_item.setLabel("left", ch)

        # LinearRegionItem 初始放在时间轴起点
        x0, x1 = float(x[0]), float(x[-1])
        r_left = x0
        r_right = min(x0 + self.fixed_width_sec, x1)

        self.region_item = pg.LinearRegionItem(
            values=[r_left, r_right],
            movable=True,
            bounds=[x0, x1],
        )
        self.region_item.setBrush(pg.mkBrush(135, 206, 250, 70))
        edge_pen = pg.mkPen(color=(65, 105, 225), width=1.5)
        self.region_item.lines[0].setPen(edge_pen)
        self.region_item.lines[1].setPen(edge_pen)
        # 锁定两端不可单独拖动，只能整体平移 → 宽度自然固定为 1024 点
        self.region_item.lines[0].setMovable(False)
        self.region_item.lines[1].setMovable(False)
        self.plot_item.addItem(self.region_item)
        self.plot_widget.set_region_reference(self.region_item, x0, x1)

        # 初始 X 视图：显示数据前段（约 10 倍窗口宽度）
        view_w = min(self.fixed_width_sec * 10, x1 - x0)
        self.plot_item.setXRange(x0, x0 + view_w, padding=0)

        # 更新位置标签
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

        for btn in self.label_btns.values():
            btn.setEnabled(True)

        self.status_label.setText(f"当前通道: {ch}  |  窗口宽度: {self.window_size} 点 ({self.fixed_width_sec:.3f} 秒)")

    def _on_mouse_moved(self, pos):
        """在状态栏显示鼠标对应的数据坐标"""
        if self.plot_curve is None:
            return
        vb = self.plot_item.vb
        if vb is None:
            return
        try:
            scene_pos = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            t = pd.Timestamp(float(scene_pos.x()), unit="s", tz=_TZ_CST_)
            self.label_pos.setText(f"鼠标: {t}  |  Y: {scene_pos.y():.4f}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 窗口宽度调节
    # ------------------------------------------------------------------

    def _on_window_size_changed(self, value: int):
        self.window_size = value
        self.fixed_width_sec = value * self.dt_sec
        if self.region_item is not None and self.timestamps is not None:
            center = sum(self.region_item.getRegion()) / 2.0
            half = self.fixed_width_sec / 2.0
            x0, x1 = float(self.timestamps[0]), float(self.timestamps[-1])
            new_left = center - half
            new_right = center + half
            if new_left < x0:
                new_left = x0
                new_right = x0 + self.fixed_width_sec
            if new_right > x1:
                new_right = x1
                new_left = x1 - self.fixed_width_sec
            self.region_item.setRegion((new_left, new_right))
            ch = self.combo_ch.currentText()
            self.status_label.setText(
                f"当前通道: {ch}  |  窗口宽度: {value} 点 ({self.fixed_width_sec:.3f} 秒)"
            )

    # ------------------------------------------------------------------
    # 样本采集
    # ------------------------------------------------------------------

    def _on_collect(self, label_id: int):
        if self.region_item is None or self.timestamps is None:
            return

        ch = self.combo_ch.currentText()
        y = self._all_data.get(ch)
        if y is None:
            return

        left, _ = self.region_item.getRegion()
        start_idx = int(np.searchsorted(self.timestamps, left, side="left"))
        start_idx = max(0, min(start_idx, len(y) - self.window_size))
        end_idx = start_idx + self.window_size

        waveform = y[start_idx:end_idx].copy()
        t0 = pd.Timestamp(self.timestamps[start_idx], unit="s", tz=_TZ_CST_)

        self.samples.append({
            "waveform": waveform,
            "label": label_id,
            "channel": ch,
            "start_index": start_idx,
            "start_time": str(t0),
        })

        self._refresh_list()
        self.btn_save.setEnabled(True)

        info = LABEL_INFO[label_id]
        self.status_label.setText(
            f"✓ 采集 [{info['name']}]  #{len(self.samples)}  |  "
            f"通道: {ch}  |  起始: {t0}  |  索引: {start_idx}"
        )

    def _refresh_list(self):
        self.sample_list.clear()
        for i, s in enumerate(self.samples):
            info = LABEL_INFO[s["label"]]
            text = f"#{i + 1:04d}  [{info['name']}]  {s['channel']}  起始={s['start_time']}"
            item = QListWidgetItem(text)
            item.setForeground(Qt.GlobalColor.black)
            self.sample_list.addItem(item)

    def _on_delete(self):
        rows = sorted({self.sample_list.row(it) for it in self.sample_list.selectedItems()}, reverse=True)
        for r in rows:
            del self.samples[r]
        self._refresh_list()
        if not self.samples:
            self.btn_save.setEnabled(False)
        self.status_label.setText(f"已删除选中，剩余 {len(self.samples)} 个样本")

    # ------------------------------------------------------------------
    # 保存 HDF5
    # ------------------------------------------------------------------

    def _on_save(self):
        if not self.samples:
            QMessageBox.warning(self, "无样本", "没有可保存的样本。")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "保存HDF5文件", "anomaly_samples.h5",
            "HDF5文件 (*.h5 *.hdf5);;所有文件 (*)",
        )
        if not path:
            return

        try:
            with h5py.File(path, "w") as f:
                for i, s in enumerate(self.samples):
                    grp = f.create_group(f"sample_{i:04d}")
                    grp.create_dataset("waveform", data=s["waveform"], dtype=np.float64)
                    grp.create_dataset("label", data=s["label"], dtype=np.int32)
                    grp.attrs["channel"] = s["channel"]
                    grp.attrs["start_time"] = s["start_time"]
                    grp.attrs["start_index"] = s["start_index"]

            self.status_label.setText(f"已保存 {len(self.samples)} 个样本 → {path}")
            QMessageBox.information(
                self, "保存成功",
                f"已保存 {len(self.samples)} 个样本到:\n{path}",
            )
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"写入HDF5文件失败:\n{e}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main():
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")

    window = AnomalyCollector()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
