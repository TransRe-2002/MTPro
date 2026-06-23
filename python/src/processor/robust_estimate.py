from __future__ import annotations

import logging
from typing import Optional, Dict

import numpy as np
import pandas as pd
import pyqtgraph as pg
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import detrend

# 配置 matplotlib 中文字体，优先使用方正黑体
matplotlib.rcParams["font.sans-serif"] = [
    "FZHei-B01", "Microsoft YaHei", "Noto Sans CJK JP", "Sarasa UI SC",
] + matplotlib.rcParams["font.sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 避免负号显示异常

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QPushButton, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QSpinBox, QDoubleSpinBox, QLabel, QMenu, QMessageBox, QSplitter, QFrame,
    QDateTimeEdit, QApplication, QStackedWidget,
)
from PySide6.QtCore import Qt, Signal, QDateTime

from base.data_manager import DataManager
from core.em_data import EMData
from processor.step_plot_widget import StepPlotWidget
from utils.time_convert import pts_to_qdt, qdt_to_pts

pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

logger = logging.getLogger(__name__)

# 默认 robust 估计参数
DEFAULT_OPTION = {
    "window_size": 1024,
    "step_pct": 50.0,
    "huber_threshold": 1.5,
    "max_iterations": 25,
    "tolerance": 1e-8,
}

# 各通道绑图颜色 (Ex/Ey 为电场用暖色，Hx/Hy/Hz 为磁场用冷色)
CH_COLOURS = {
    "Ex": (220, 50, 50),   # 红
    "Ey": (200, 120, 0),   # 橙
    "Hx": (30, 80, 220),   # 蓝
    "Hy": (30, 160, 80),   # 绿
    "Hz": (140, 40, 180),  # 紫
}


def _detect_channels(channel_names: list[str]) -> dict[str, Optional[str]]:
    """根据通道名称的命名规则自动匹配 Ex/Ey/Hx/Hy/Hz 五道。

    优先匹配第一个符合规则且尚未被占用的通道，避免重复映射。
    """
    mapping: dict[str, Optional[str]] = {
        "Ex": None, "Ey": None, "Hx": None, "Hy": None, "Hz": None,
    }
    for ch in channel_names:
        lower = ch.lower()
        if lower.startswith("e") and "x" in lower and mapping["Ex"] is None:
            mapping["Ex"] = ch
        elif lower.startswith("e") and "y" in lower and mapping["Ey"] is None:
            mapping["Ey"] = ch
        elif (lower.startswith("h") or lower.startswith("b")) and "x" in lower and mapping["Hx"] is None:
            mapping["Hx"] = ch
        elif (lower.startswith("h") or lower.startswith("b")) and "y" in lower and mapping["Hy"] is None:
            mapping["Hy"] = ch
        elif (lower.startswith("h") or lower.startswith("b")) and "z" in lower and mapping["Hz"] is None:
            mapping["Hz"] = ch
    return mapping


def _is_e_channel(name: str) -> bool:
    """判断通道是否属于电场分量（名称以 'e' 开头）。"""
    return name.lower().startswith("e")


def _has_magnetic_channel(data: EMData) -> bool:
    """检查 EMData 是否包含至少一个磁场通道（名称以 'h' 或 'b' 开头）。"""
    return any(ch.lower().startswith(("h", "b")) for ch in data.channel_names())


class RobustEstimateWidget(QWidget):
    """Robust 阻抗估计交互界面。

    左侧为控制面板（台站选择、时间范围、通道映射、估计参数），
    右侧上部为时序波形预览，下部为计算结果表格和对比图。
    """
    result_ready = Signal(object)  # 计算完成后触发，携带结果对象

    # 信号总线
    data_requested = Signal(int, str)
    result_submitted = Signal(int, str, object)

    def __init__(
        self,
        data_id: int = 0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._data_id = data_id
        self._pkg: dict = {}
        self._result = None  # 最近一次计算结果
        self._spectrum = None  # emtf_rs 原始 SpectrumResult
        self._bands = None     # to_bands 聚合后的 BandedResult
        self._region = None
        self._curves: dict[str, pg.PlotDataItem] = {}  # 当前显示的波形曲线
        self._updating_range = False  # 防止信号递归的标志位
        self._ts_array: Optional[np.ndarray] = None  # 预计算的时间戳缓存，避免每次拖拽区域时重建百万级数组

        # ── 波形预览 plot widget ──
        self.plot_widget = StepPlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'E (mV/km)')
        self.plot_widget.setLabel('bottom', '时间')
        self._legend = self.plot_widget.addLegend(offset=(0, 10))

        self.plot_item = self.plot_widget.getPlotItem()

        # 双 Y 轴：左侧显示电场 (mV/km)，右侧显示磁场 (nT)
        # 使用两个 ViewBox 分别控制各自 Y 轴范围，X 轴通过 setXLink 联动
        self.plot_widget.showAxis('right')
        self.plot_widget.setLabel('right', 'B (nT)')
        self._b_viewbox = pg.ViewBox()
        self.plot_item.scene().addItem(self._b_viewbox)
        self.plot_item.getAxis('right').linkToView(self._b_viewbox)
        self._b_viewbox.setXLink(self.plot_item)  # B ViewBox 跟随主 ViewBox 的 X 范围
        self.plot_item.vb.sigResized.connect(self._update_dual_geometry)  # 窗口大小变化时同步几何
        self.plot_widget.zoom_callback = self._sync_b_viewbox_zoom  # 滚轮缩放时等比同步 B ViewBox 的 Y 轴

        # 时间范围选取区域（LinearRegionItem 是纯图形控件，不锚定数据点）
        self.region_item = pg.LinearRegionItem(values=[0, 1], movable=True)
        border_pen = pg.mkPen(color=(65, 105, 225), width=1)
        self.region_item.lines[0].setPen(border_pen)
        self.region_item.lines[1].setPen(border_pen)
        self.region_item.setBrush(pg.mkBrush(135, 206, 250, 50))  # 半透明蓝色填充
        self.plot_widget.addItem(self.region_item)
        self.region_item.sigRegionChanged.connect(self._on_region_changed)  # 拖拽中持续更新
        self.region_item.sigRegionChangeFinished.connect(self._on_region_change_finished)  # 拖拽结束后做重操作

        # ── 台站选择器 ──
        station_layout = QHBoxLayout()
        station_layout.addWidget(QLabel("台站:"))
        self.station_combo = QComboBox()
        self.station_combo.currentIndexChanged.connect(self._on_station_changed)
        station_layout.addWidget(self.station_combo, 1)
        self.refresh_plot_btn = QPushButton("刷新绘图")
        self.refresh_plot_btn.clicked.connect(self._refresh_plot)
        station_layout.addWidget(self.refresh_plot_btn)

        # ── 时间范围编辑区 ──
        time_group = QGroupBox("时间范围")
        time_layout = QFormLayout(time_group)
        time_row = QHBoxLayout()
        self.start_dt = QDateTimeEdit()
        self.start_dt.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.start_dt.setCalendarPopup(True)
        self.start_dt.setTimeSpec(Qt.TimeSpec.UTC)
        self.start_dt.dateTimeChanged.connect(self._on_dt_start_changed)
        self.end_dt = QDateTimeEdit()
        self.end_dt.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.end_dt.setCalendarPopup(True)
        self.end_dt.setTimeSpec(Qt.TimeSpec.UTC)
        self.end_dt.dateTimeChanged.connect(self._on_dt_end_changed)
        time_row.addWidget(QLabel("起:"))
        time_row.addWidget(self.start_dt, 1)
        time_row.addWidget(QLabel("止:"))
        time_row.addWidget(self.end_dt, 1)
        time_layout.addRow(time_row)

        # 样点索引微调框（与日期时间双向同步）
        idx_row = QHBoxLayout()
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, 0)
        self.start_spin.setToolTip("起始样点索引")
        self.start_spin.valueChanged.connect(self._on_spin_start_changed)
        self.end_spin = QSpinBox()
        self.end_spin.setRange(0, 0)
        self.end_spin.setToolTip("结束样点索引（不包含）")
        self.end_spin.valueChanged.connect(self._on_spin_end_changed)
        self.full_button = QPushButton("全量")
        self.full_button.setFixedWidth(48)
        self.full_button.clicked.connect(self._reset_range)
        idx_row.addWidget(QLabel("样点起:"))
        idx_row.addWidget(self.start_spin)
        idx_row.addWidget(QLabel("止:"))
        idx_row.addWidget(self.end_spin)
        idx_row.addWidget(self.full_button)
        time_layout.addRow(idx_row)

        self.range_info = QLabel("")  # 显示当前选中范围的点数和时长
        self.warning_label = QLabel("")  # 当选中数据过短时显示警告
        self.warning_label.setStyleSheet("color: #cc6600; font-weight: bold;")
        time_layout.addRow(self.range_info)
        time_layout.addRow(self.warning_label)

        # ── 通道映射 ──
        channel_group = QGroupBox("通道映射")
        channel_form = QFormLayout(channel_group)
        self.ch_combo_ex = QComboBox()
        self.ch_combo_ey = QComboBox()
        self.ch_combo_hx = QComboBox()
        self.ch_combo_hy = QComboBox()
        self.ch_combo_hz = QComboBox()
        self.ch_combo_hz.addItem("(不使用)", None)  # Hz 为可选通道
        # 通道变更时自动刷新波形图
        self.ch_combo_ex.currentIndexChanged.connect(lambda: self._on_channel_changed())
        self.ch_combo_ey.currentIndexChanged.connect(lambda: self._on_channel_changed())
        self.ch_combo_hx.currentIndexChanged.connect(lambda: self._on_channel_changed())
        self.ch_combo_hy.currentIndexChanged.connect(lambda: self._on_channel_changed())
        self.ch_combo_hz.currentIndexChanged.connect(lambda: self._on_channel_changed())
        channel_form.addRow("Ex:", self.ch_combo_ex)
        channel_form.addRow("Ey:", self.ch_combo_ey)
        channel_form.addRow("Hx:", self.ch_combo_hx)
        channel_form.addRow("Hy:", self.ch_combo_hy)
        channel_form.addRow("Hz:", self.ch_combo_hz)

        # ── 估计参数 ──
        param_group = QGroupBox("估计参数")
        param_form = QFormLayout(param_group)

        self.sample_rate_label = QLabel("—")
        param_form.addRow("采样率 (Hz):", self.sample_rate_label)

        self.window_combo = QComboBox()
        self._window_values = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
        for v in self._window_values:
            self.window_combo.addItem(str(v), v)
        default_idx = 0  # 默认 1024 样点
        self.window_combo.setCurrentIndex(default_idx)
        self.window_combo.currentIndexChanged.connect(self._update_warning)
        param_form.addRow("窗口大小 (样点):", self.window_combo)

        self.step_pct_spin = QDoubleSpinBox()
        self.step_pct_spin.setRange(1.0, 100.0)
        self.step_pct_spin.setDecimals(0)
        self.step_pct_spin.setValue(50.0)
        self.step_pct_spin.setSuffix("%")
        param_form.addRow("步长 (%窗口):", self.step_pct_spin)

        self.huber_spin = QDoubleSpinBox()
        self.huber_spin.setRange(0.1, 10.0)
        self.huber_spin.setSingleStep(0.1)
        self.huber_spin.setValue(DEFAULT_OPTION["huber_threshold"])
        param_form.addRow("Huber threshold:", self.huber_spin)

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 200)
        self.max_iter_spin.setValue(DEFAULT_OPTION["max_iterations"])
        param_form.addRow("最大迭代:", self.max_iter_spin)

        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(1e-16, 1.0)
        self.tol_spin.setDecimals(12)
        self.tol_spin.setSingleStep(1e-8)
        self.tol_spin.setValue(DEFAULT_OPTION["tolerance"])
        param_form.addRow("收敛容差:", self.tol_spin)

        self.bands_spin = QSpinBox()
        self.bands_spin.setRange(10, 200)
        self.bands_spin.setValue(40)
        self.bands_spin.setToolTip("对数均分频带数量，影响视电阻率/相位图的频点密度")
        self.bands_spin.valueChanged.connect(self._on_bands_changed)
        param_form.addRow("频带数量:", self.bands_spin)

        # ── 运行按钮 ──
        btn_layout = QHBoxLayout()
        self.run_button = QPushButton("运行 Robust 估计")
        self.run_button.clicked.connect(self._run_estimation)
        btn_layout.addStretch()
        btn_layout.addWidget(self.run_button)

        self.status_label = QLabel("")

        # ── 结果表格 ──
        # 阻抗张量结果表
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "freq (Hz)", "period (s)",
            "Zxx", "Zxy", "Zyx", "Zyy",
            "Zxx SE", "Zxy SE", "Zyx SE", "Zyy SE",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_context_menu)
        self.table.keyPressEvent = self._table_key_press_event

        self.export_edi_btn = QPushButton("输出阻抗")
        self.export_edi_btn.clicked.connect(self._export_edi)

        # 视电阻率与相位结果表
        self.resistivity_table = QTableWidget()
        self.resistivity_table.setColumnCount(10)
        self.resistivity_table.setHorizontalHeaderLabels([
            "freq (Hz)", "period (s)",
            "ρxy (Ω·m)", "ρyx (Ω·m)",
            "φxy (°)", "φyx (°)",
            "ρxy SE", "ρyx SE",
            "φxy SE (°)", "φyx SE (°)",
        ])
        self.resistivity_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.resistivity_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.resistivity_table.customContextMenuRequested.connect(self._on_table_context_menu)
        self.resistivity_table.keyPressEvent = self._table_key_press_event

        # 结果标签页
        self.result_tabs = QTabWidget()
        tab_corner = QWidget()
        tab_corner_layout = QHBoxLayout(tab_corner)
        tab_corner_layout.setContentsMargins(0, 0, 4, 0)
        tab_corner_layout.addWidget(self.export_edi_btn)
        self.result_tabs.setCornerWidget(tab_corner)  # 右上角导出按钮
        self.result_tabs.addTab(self.table, "阻抗结果")
        self.result_tabs.addTab(self.resistivity_table, "视电阻率与相位")

        # 第三标签页：matplotlib 视电阻率/相位对比图
        self.mpl_fig = Figure(figsize=(6, 5), tight_layout=True)
        self.mpl_canvas = FigureCanvas(self.mpl_fig)
        self.mpl_canvas.setParent(self)
        self.result_tabs.addTab(self.mpl_canvas, "对比图")

        # 计算前显示占位提示，计算后切换到结果标签页
        self._placeholder = QLabel("此处显示计算结果信息")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888; font-size: 14px;")
        self._result_stack = QStackedWidget()
        self._result_stack.addWidget(self._placeholder)  # index 0: 占位
        self._result_stack.addWidget(self.result_tabs)   # index 1: 结果
        self._result_stack.setCurrentIndex(0)

        # ── 组装布局 ──
        # 左侧控制面板
        control_panel = QVBoxLayout()
        control_panel.setContentsMargins(4, 4, 4, 4)
        control_panel.addLayout(station_layout)
        control_panel.addWidget(time_group)
        control_panel.addWidget(channel_group)
        control_panel.addWidget(param_group)
        control_panel.addLayout(btn_layout)
        control_panel.addWidget(self.status_label)
        control_panel.addStretch()

        control_widget = QWidget()
        control_widget.setLayout(control_panel)
        control_widget.setFixedWidth(340)

        # 右侧：上方波形图 + 下方结果区，用竖向 Splitter 分隔
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(self.plot_widget)
        right_splitter.addWidget(self._result_stack)
        right_splitter.setStretchFactor(0, 0)  # 波形区不随窗口拉伸
        right_splitter.setStretchFactor(1, 1)  # 结果区占据额外空间
        right_splitter.setSizes([200, 600])

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(control_widget)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([340, 960])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(main_splitter)

        # ── 初始填充 ── 数据到达后由 _on_data_ready 触发

    # ─── 数据接收 ──────────────────────────────────────────────

    def _on_data_ready(self, data_id: int, channel: str, pkg: dict) -> None:
        if data_id != self._data_id:
            return
        self._pkg = pkg
        if not self._pkg.get('stations'):
            return
        self._populate_stations()
        self._apply_station_data()

    # ─── 双 Y 轴同步 ──────────────────────────────────────────────

    def _update_dual_geometry(self):
        """主 ViewBox 尺寸变化时，同步 B ViewBox 的几何位置。

        B ViewBox 叠加在主 ViewBox 上方，需要保持相同的大小和位置。
        """
        self._b_viewbox.setGeometry(self.plot_item.vb.sceneBoundingRect())

    def _sync_b_viewbox_zoom(self, old_main_yrange, new_main_yrange):
        """滚轮缩放时等比同步 B ViewBox 的 Y 轴范围。

        当主 ViewBox（电场，左轴）通过滚轮缩放时，计算其 Y 范围的缩放比例，
        然后对 B ViewBox（磁场，右轴）按相同比例缩放，保证左右轴视觉上同步。
        """
        old_range = old_main_yrange[1] - old_main_yrange[0]
        new_range = new_main_yrange[1] - new_main_yrange[0]
        if old_range <= 1e-16:
            return
        scale = new_range / old_range
        b_yrange = self._b_viewbox.viewRange()[1]
        b_center = (b_yrange[0] + b_yrange[1]) / 2.0
        b_half = (b_yrange[1] - b_yrange[0]) / 2.0
        self._b_viewbox.setYRange(b_center - b_half * scale, b_center + b_half * scale, padding=0)

    # ─── 台站 / 通道管理 ───────────────────────────────────────────

    def _populate_stations(self):
        prev_key = self.station_combo.currentData()
        self.station_combo.blockSignals(True)
        self.station_combo.clear()
        for s in self._pkg.get('stations', []):
            if s['has_magnetic']:
                self.station_combo.addItem(f"#{s['id']} {s['name']}", s['id'])
        self.station_combo.blockSignals(False)
        if prev_key is not None and self.station_combo.findData(prev_key) >= 0:
            self.station_combo.setCurrentIndex(self.station_combo.findData(prev_key))
        elif self.station_combo.count() > 0:
            self._on_station_changed(0)

    def _on_station_changed(self, _idx: int):
        """台站切换时请求新数据。"""
        new_id = self.station_combo.currentData()
        if new_id is not None:
            self._data_id = new_id
            self.data_requested.emit(new_id, "")
        # 数据到达后 _on_data_ready → _apply_station_data 完成 UI 更新

    def _apply_station_data(self):
        """根据 _pkg 更新通道映射、时间范围和波形。"""
        pkg = self._pkg
        channels: list[str] = pkg.get('channel_names', [])
        mapping = _detect_channels(channels)

        for combo, key in [
            (self.ch_combo_ex, "Ex"), (self.ch_combo_ey, "Ey"),
            (self.ch_combo_hx, "Hx"), (self.ch_combo_hy, "Hy"),
        ]:
            combo.blockSignals(True)
            combo.clear()
            for ch in channels:
                combo.addItem(ch, ch)
            auto_ch = mapping.get(key)
            if auto_ch is not None:
                idx = combo.findText(auto_ch)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)

        self.ch_combo_hz.blockSignals(True)
        self.ch_combo_hz.clear()
        self.ch_combo_hz.addItem("(不使用)", None)
        for ch in channels:
            self.ch_combo_hz.addItem(ch, ch)
        self.ch_combo_hz.setCurrentIndex(0)
        self.ch_combo_hz.blockSignals(False)

        dt = pkg.get('dt', 1.0)
        n = pkg.get('npts', 0)
        start_ts = pkg.get('start_time', 0)
        end_ts = start_ts + dt * n if n > 0 else start_ts

        self.sample_rate_label.setText(f"{1.0 / dt:.4f}" if dt > 0 else "N/A")
        self.start_spin.blockSignals(True)
        self.start_spin.setRange(0, n - 1)
        self.start_spin.setValue(0)
        self.start_spin.blockSignals(False)
        self.end_spin.blockSignals(True)
        self.end_spin.setRange(1, n)
        self.end_spin.setValue(n)
        self.end_spin.blockSignals(False)

        start_dt_val = pts_to_qdt(pd.Timestamp(start_ts, unit='s', tz='Asia/Shanghai'))
        end_dt_val = pts_to_qdt(pd.Timestamp(end_ts, unit='s', tz='Asia/Shanghai'))
        self.start_dt.blockSignals(True)
        self.start_dt.setDateTime(start_dt_val)
        self.start_dt.setMinimumDateTime(start_dt_val)
        self.start_dt.setMaximumDateTime(end_dt_val)
        self.start_dt.blockSignals(False)
        self.end_dt.blockSignals(True)
        self.end_dt.setDateTime(end_dt_val)
        self.end_dt.setMinimumDateTime(start_dt_val)
        self.end_dt.setMaximumDateTime(end_dt_val)
        self.end_dt.blockSignals(False)

        self._refresh_plot()

    def _on_channel_changed(self):
        """通道映射变更时刷新波形图。"""
        self._refresh_plot()

    # ─── 波形绑图 ─────────────────────────────────────────────────────────

    def _selected_channels(self) -> dict[str, Optional[str]]:
        """获取当前各分量选中的实际通道名。"""
        return {
            "Ex": self.ch_combo_ex.currentData(),
            "Ey": self.ch_combo_ey.currentData(),
            "Hx": self.ch_combo_hx.currentData(),
            "Hy": self.ch_combo_hy.currentData(),
            "Hz": self.ch_combo_hz.currentData(),
        }

    def _refresh_plot(self):
        """刷新波形预览：清除旧曲线，按当前通道选择重新绑图。"""
        pkg = self._pkg
        if not pkg:
            return
        all_y = pkg.get('all_y', {})
        x = pkg.get('x', np.array([]))
        if len(x) == 0:
            return

        # 清除旧曲线和图例
        for curve in list(self._curves.values()):
            curve_view = curve.getViewBox()
            if curve_view is self._b_viewbox:
                self._b_viewbox.removeItem(curve)
            else:
                self.plot_item.vb.removeItem(curve)
        self._curves.clear()
        self._legend.clear()

        # 构建时间戳数组（同时缓存，供拖拽区域时二分查找使用）
        t = np.array([t.timestamp() for t in pkg['datetime_index']])
        self._ts_array = t

        channels = self._selected_channels()

        for tag, ch_name in channels.items():
            if ch_name is None:
                continue
            y = all_y.get(ch_name)
            if y is None:
                continue
            # 去均值以便在同一坐标系下对比波形形态
            y = y - np.nanmean(y)

            colour = CH_COLOURS.get(tag, (100, 100, 100))
            pen = pg.mkPen(color=colour, width=1)
            label = f"{tag} ({ch_name})"

            if _is_e_channel(ch_name):
                # 电场 → 左轴（主 ViewBox）
                curve = pg.PlotDataItem(t, y, pen=pen, name=label,
                                        autoDownsample=True, clipToView=True,
                                        connect='finite')  # NaN 处断开，不连接
                curve.setParentItem(self.plot_item.vb)
                curve.forgetViewBox()
                self.plot_item.vb.addItem(curve)
            else:
                # 磁场 → 右轴（B ViewBox）
                curve = pg.PlotDataItem(t, y, pen=pen, name=label,
                                        autoDownsample=True, clipToView=True,
                                        connect='finite')
                curve.setParentItem(self._b_viewbox)
                curve.forgetViewBox()
                self._b_viewbox.addItem(curve)
            self._legend.addItem(curve, label)
            self._curves[tag] = curve

        # 将选取区域设为全量数据范围
        if len(t) > 1:
            self.region_item.blockSignals(True)
            self.region_item.setRegion((t[0], t[-1]))
            self.region_item.blockSignals(False)

        self._update_range_info()
        self._update_warning()

    # ─── 区域 / 范围选择 ─────────────────────────────────────────────

    def _on_region_changed(self):
        """LinearRegionItem 拖拽过程中持续调用，只更新轻量控件（微调框）。

        注意：此方法通过预缓存的 _ts_array 做二分查找，
        避免每次鼠标移动都重建百万级时间戳数组导致的卡顿。
        """
        if self._updating_range:
            return
        pkg = self._pkg
        if not pkg:
            return
        left, right = self.region_item.getRegion()
        start, end = self._ts_range_to_indices(left, right)
        self._updating_range = True
        self.start_spin.blockSignals(True)
        self.start_spin.setValue(start)
        self.start_spin.blockSignals(False)
        self.end_spin.blockSignals(True)
        self.end_spin.setValue(end)
        self.end_spin.blockSignals(False)
        self._updating_range = False

    def _on_region_change_finished(self):
        """拖拽结束后的重操作：同步日期时间编辑器和范围信息。"""
        if self._updating_range:
            return
        pkg = self._pkg
        if not pkg:
            return
        left, right = self.region_item.getRegion()
        start, end = self._ts_range_to_indices(left, right)
        self._set_dt_from_indices(start, end)
        self._update_range_info()
        self._update_warning()

    def _on_spin_start_changed(self, val: int):
        """起始样点微调框变更 → 同步日期时间和选取区域。"""
        if self._updating_range:
            return
        self._set_region_from_indices(val, self.end_spin.value())
        self._set_dt_from_indices(val, self.end_spin.value())
        self._update_range_info()
        self._update_warning()

    def _on_spin_end_changed(self, val: int):
        """结束样点微调框变更 → 同步日期时间和选取区域。"""
        if self._updating_range:
            return
        self._set_region_from_indices(self.start_spin.value(), val)
        self._set_dt_from_indices(self.start_spin.value(), val)
        self._update_range_info()
        self._update_warning()

    def _on_dt_start_changed(self, qdt: QDateTime):
        """起始日期时间变更 → 同步样点索引和选取区域。"""
        if self._updating_range:
            return
        pkg = self._pkg
        if not pkg:
            return
        t = qdt_to_pts(qdt)
        idx = np.searchsorted(pkg['datetime_index'], t)
        idx = max(0, min(idx, pkg['npts'] - 1))
        self._set_spins(idx, self.end_spin.value())
        self._set_region_from_indices(idx, self.end_spin.value())

    def _on_dt_end_changed(self, qdt: QDateTime):
        """结束日期时间变更 → 同步样点索引和选取区域。"""
        if self._updating_range:
            return
        pkg = self._pkg
        if not pkg:
            return
        t = qdt_to_pts(qdt)
        idx = np.searchsorted(pkg['datetime_index'], t)
        idx = max(1, min(idx, pkg['npts']))
        self._set_spins(self.start_spin.value(), idx)
        self._set_region_from_indices(self.start_spin.value(), idx)

    def _ts_range_to_indices(self, left: float, right: float) -> tuple[int, int]:
        """将时间戳范围转换为样点索引范围，使用预缓存的 _ts_array 做二分查找。"""
        data = self._current_data()
        if data is None or self._ts_array is None:
            return 0, 0
        t_arr = self._ts_array
        start = int(np.searchsorted(t_arr, left))
        end = int(np.searchsorted(t_arr, right))
        return max(0, start), min(pkg['npts'], end)

    def _set_spins(self, start: int, end: int):
        """同步设置起止样点微调框，同时更新范围信息。"""
        self._updating_range = True
        self.start_spin.blockSignals(True)
        self.start_spin.setValue(start)
        self.start_spin.blockSignals(False)
        self.end_spin.blockSignals(True)
        self.end_spin.setValue(end)
        self.end_spin.blockSignals(False)
        self._updating_range = False
        self._update_range_info()
        self._update_warning()

    def _set_dt_from_indices(self, start: int, end: int):
        """将样点索引转换为日期时间并同步编辑器。"""
        pkg = self._pkg
        if not pkg:
            return
        idx = pkg['datetime_index']
        n = len(idx)
        start = max(0, min(start, n - 1))
        end = max(1, min(end, n))
        self._updating_range = True
        self.start_dt.blockSignals(True)
        self.start_dt.setDateTime(pts_to_qdt(idx[start]))
        self.start_dt.blockSignals(False)
        self.end_dt.blockSignals(True)
        self.end_dt.setDateTime(pts_to_qdt(idx[end - 1]))
        self.end_dt.blockSignals(False)
        self._updating_range = False

    def _set_region_from_indices(self, start: int, end: int):
        """将样点索引转换为时间戳并更新 LinearRegionItem 位置。"""
        data = self._current_data()
        if data is None or self._ts_array is None:
            return
        t_arr = self._ts_array
        if end > 0 and start < len(t_arr):
            self._updating_range = True
            self.region_item.blockSignals(True)
            self.region_item.setRegion((t_arr[start], t_arr[min(end, len(t_arr) - 1)]))
            self.region_item.blockSignals(False)
            self._updating_range = False

    def _reset_range(self):
        """恢复选取范围为全量数据。"""
        pkg = self._pkg
        if not pkg:
            return
        n = pkg['npts']
        self._set_spins(0, n)
        self._set_dt_from_indices(0, n)
        self._set_region_from_indices(0, n)
        self._update_range_info()
        self._update_warning()

    def _update_range_info(self):
        """更新右下角的范围信息提示。"""
        pkg = self._pkg
        if not pkg:
            return
        start = self.start_spin.value()
        end = self.end_spin.value()
        n = end - start
        dt = pkg['dt']
        duration = n * dt
        self.range_info.setText(f"选中 {n} 样点 ({start}:{end}), 时长 {duration:.0f}s")

    def _update_warning(self):
        """当选中数据长度不足 26 个窗口时显示警告。"""
        window_size = self.window_combo.currentData()
        min_required = 26 * window_size
        start = self.start_spin.value()
        end = self.end_spin.value()
        n = end - start
        if n < min_required:
            self.warning_label.setText(
                f"警告: 选中 {n} 样点 < 26×窗口={min_required}，"
                f"建议增加数据长度或减小窗口"
            )
        else:
            self.warning_label.setText("")

    # ─── robust 估计 ───────────────────────────────────────────────────

    def _run_estimation(self):
        """运行 robust 阻抗估计。

        流程：从选中范围提取各通道数据 → 预处理（去趋势）→ 调用 emtf_rs 估计
        → 展示结果表格和对比图。

        注意：这里只做去趋势（detrend），不做插值。emtf_rs 内部有自己的插值逻辑，
        此处重复插值反而会掩盖数据质量问题。
        """
        pkg = self._pkg
        if not pkg:
            QMessageBox.warning(self, "错误", "请先选择一个台站。")
            return
        all_y = pkg.get('all_y', {})

        ch_ex = self.ch_combo_ex.currentData()
        ch_ey = self.ch_combo_ey.currentData()
        ch_hx = self.ch_combo_hx.currentData()
        ch_hy = self.ch_combo_hy.currentData()
        ch_hz = self.ch_combo_hz.currentData()

        for label, ch in [("Ex", ch_ex), ("Ey", ch_ey), ("Hx", ch_hx), ("Hy", ch_hy)]:
            if ch is None:
                QMessageBox.warning(self, "错误", f"请选择 {label} 通道。")
                return

        try:
            import emtf_rs

            start = self.start_spin.value()
            end = self.end_spin.value()
            if end <= start:
                QMessageBox.warning(self, "错误", "结束索引必须大于起始索引。")
                return

            ex_arr = all_y.get(ch_ex, np.array([]))[start:end]
            ey_arr = all_y.get(ch_ey, np.array([]))[start:end]
            hx_arr = all_y.get(ch_hx, np.array([]))[start:end]
            hy_arr = all_y.get(ch_hy, np.array([]))[start:end]
            hz_arr = None
            if ch_hz is not None:
                hz_arr = all_y.get(ch_hz, np.array([]))[start:end]

            def _preprocess(arr: np.ndarray, name: str) -> np.ndarray:
                """对通道数据做去趋势（detrend），以移除仪器漂移等低频伪影。

                如果数据中存在 NaN 断点则跳过去趋势，避免 scipy.detrend 报错。
                emtf_rs 内部会自行处理 NaN 和去趋势。
                """
                n_nan = np.sum(~np.isfinite(arr))
                if n_nan > 0:
                    logger.warning("%s: %d NaN in selected range (%.2f%%), "
                                   "detrend skipped", name, n_nan,
                                   100 * n_nan / len(arr))
                    return arr
                return detrend(arr)

            ex_arr = _preprocess(ex_arr, ch_ex)
            ey_arr = _preprocess(ey_arr, ch_ey)
            hx_arr = _preprocess(hx_arr, ch_hx)
            hy_arr = _preprocess(hy_arr, ch_hy)
            if hz_arr is not None:
                hz_arr = _preprocess(hz_arr, str(ch_hz))

            sample_rate = 1.0 / pkg['dt']
            window_size = self.window_combo.currentData()
            n_samples = end - start
            if window_size > n_samples:
                QMessageBox.warning(
                    self, "参数错误",
                    f"窗口大小 ({window_size}) 超过选中数据点数 ({n_samples})，"
                    f"请减小窗口大小或扩大数据范围。"
                )
                return
            step_size = max(1, int(window_size * self.step_pct_spin.value() / 100.0))
            huber = self.huber_spin.value()
            max_iter = self.max_iter_spin.value()
            tolerance = self.tol_spin.value()

            self.status_label.setText("正在计算...")
            self.run_button.setEnabled(False)
            QApplication.processEvents()  # 强制刷新 UI，显示"正在计算"

            result = emtf_rs.estimate_single_station_spectrum(
                ex_arr, ey_arr, hx_arr, hy_arr,
                sample_rate_hz=sample_rate,
                window_size=window_size,
                hz=hz_arr,
                step_size=step_size,
                huber_threshold=huber,
                max_iterations=max_iter,
                tolerance=tolerance,
            )

            self._spectrum = result
            self._result = result
            self._build_bands()
            self._display_result(result)
            self.result_ready.emit(result)
            self.status_label.setText(
                f"完成 — {len(result.frequencies_hz)} 个频率, {result.segment_count} 个时间窗"
            )

        except Exception as e:
            logger.exception("Robust estimation failed")
            QMessageBox.critical(self, "估计失败", str(e))
            self.status_label.setText("失败")
        finally:
            self.run_button.setEnabled(True)

    def _build_bands(self):
        """从 spectrum 构建 BandedResult，period 范围 2*dt ~ w*dt 对数均分。"""
        import emtf_rs

        if self._spectrum is None:
            self._bands = None
            return
        data = self._current_data()
        if data is None:
            self._bands = None
            return
        dt = pkg['dt']
        w = self.window_combo.currentData()
        n_bands = self.bands_spin.value()
        periods = np.logspace(np.log10(2 * dt), np.log10(w * dt), n_bands)
        band_configs = [
            emtf_rs.BandConfig(
                name=f"band_{i+1}",
                min_period=float(periods[i]),
                max_period=float(periods[i+1]),
            )
            for i in range(len(periods) - 1)
        ]
        self._bands = self._spectrum.to_bands(band_configs)

    def _on_bands_changed(self):
        """频带数量变更时重建 bands 并刷新对比图。"""
        if self._spectrum is None:
            return
        self._build_bands()
        self._draw_comparison_plot()

    def _display_result(self, result):
        """将 SpectrumResult 填充到表格中，并用 to_bands 后的 BandedResult 画图。"""
        self._result_stack.setCurrentIndex(1)
        freqs = np.array(result.frequencies_hz)
        n = len(freqs)

        def _c2s(c: complex) -> str:
            return f"{c.real:.4e}{c.imag:+.4e}j"

        # ── 阻抗张量结果表 ──
        self.table.setRowCount(n)
        for i in range(n):
            f = freqs[i]
            period = 1.0 / f if f > 0 else float("inf")
            self.table.setItem(i, 0, QTableWidgetItem(f"{f:.6f}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{period:.3f}"))
            self.table.setItem(i, 2, QTableWidgetItem(_c2s(result.zxx[i])))
            self.table.setItem(i, 3, QTableWidgetItem(_c2s(result.zxy[i])))
            self.table.setItem(i, 4, QTableWidgetItem(_c2s(result.zyx[i])))
            self.table.setItem(i, 5, QTableWidgetItem(_c2s(result.zyy[i])))
            self.table.setItem(i, 6, QTableWidgetItem(f"{result.zxx_se[i]:.4e}"))
            self.table.setItem(i, 7, QTableWidgetItem(f"{result.zxy_se[i]:.4e}"))
            self.table.setItem(i, 8, QTableWidgetItem(f"{result.zyx_se[i]:.4e}"))
            self.table.setItem(i, 9, QTableWidgetItem(f"{result.zyy_se[i]:.4e}"))

        # ── 视电阻率与相位结果表 ──
        self.resistivity_table.setRowCount(n)
        for i in range(n):
            f = freqs[i]
            period = 1.0 / f if f > 0 else float("inf")
            self.resistivity_table.setItem(i, 0, QTableWidgetItem(f"{f:.6f}"))
            self.resistivity_table.setItem(i, 1, QTableWidgetItem(f"{period:.3f}"))
            self.resistivity_table.setItem(i, 2, QTableWidgetItem(f"{result.rho_xy[i]:.4e}"))
            self.resistivity_table.setItem(i, 3, QTableWidgetItem(f"{result.rho_yx[i]:.4e}"))
            self.resistivity_table.setItem(i, 4, QTableWidgetItem(f"{result.phi_xy[i]:.4f}"))
            self.resistivity_table.setItem(i, 5, QTableWidgetItem(f"{result.phi_yx[i]:.4f}"))
            self.resistivity_table.setItem(i, 6, QTableWidgetItem(f"{result.rho_xy_se[i]:.4e}"))
            self.resistivity_table.setItem(i, 7, QTableWidgetItem(f"{result.rho_yx_se[i]:.4e}"))
            self.resistivity_table.setItem(i, 8, QTableWidgetItem(f"{result.phi_xy_se[i]:.4f}"))
            self.resistivity_table.setItem(i, 9, QTableWidgetItem(f"{result.phi_yx_se[i]:.4f}"))

        # ── matplotlib 对比图 ──
        self._draw_comparison_plot()

    def _draw_comparison_plot(self):
        """用 matplotlib 绘制视电阻率和相位的误差棒对比图。"""
        if self._bands is None:
            self.mpl_fig.clear()
            self.mpl_canvas.draw()
            return

        result = self._bands

        p = np.array(result.periods)
        finite = np.isfinite(p) & (p > 0)
        if not finite.any():
            self.mpl_fig.clear()
            self.mpl_canvas.draw()
            return

        p = p[finite]
        rho_xy = np.array(result.rho_xy)[finite]
        rho_xy_se = np.array(result.rho_xy_se)[finite]
        phi_xy = np.array(result.phi_xy)[finite]
        phi_xy_se = np.array(result.phi_xy_se)[finite]
        rho_yx = np.array(result.rho_yx)[finite]
        rho_yx_se = np.array(result.rho_yx_se)[finite]
        phi_yx = np.array(result.phi_yx)[finite]
        phi_yx_se = np.array(result.phi_yx_se)[finite]

        self.mpl_fig.clear()
        ax1, ax2 = self.mpl_fig.subplots(2, 1)

        ax1.errorbar(p, rho_xy, yerr=rho_xy_se,
                     fmt='o', color='red', capsize=3, markersize=5,
                     label='ρxy (Ω·m)')
        ax1.errorbar(p, rho_yx, yerr=rho_yx_se,
                     fmt='s', color='blue', capsize=3, markersize=5,
                     label='ρyx (Ω·m)')
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_ylabel('Apparent Resistivity (Ω·m)')
        ax1.set_xlabel('Period (s)')
        ax1.set_title('视电阻率')
        ax1.legend(loc='upper right', fontsize=7, framealpha=0.4, edgecolor='none')
        ax1.grid(True, alpha=0.3)

        ax2.errorbar(p, phi_xy, yerr=phi_xy_se,
                     fmt='o', color='red', capsize=3, markersize=5,
                     label='φxy (°)')
        ax2.errorbar(p, phi_yx, yerr=phi_yx_se,
                     fmt='s', color='blue', capsize=3, markersize=5,
                     label='φyx (°)')
        ax2.set_xscale('log')
        ax2.set_xlabel('Period (s)')
        ax2.set_ylabel('Phase (°)')
        ax2.set_title('相位')
        ax2.legend(loc='upper right', fontsize=7, framealpha=0.4, edgecolor='none')
        ax2.grid(True, alpha=0.3)

        self.mpl_canvas.draw()

    def result(self):
        """返回最近一次计算结果。"""
        return self._result

    # ─── 表格剪贴板 / 导出 ──────────────────────────────────────

    def _table_key_press_event(self, event):
        """重载键盘事件，Ctrl+C 复制表格选中区域。"""
        if event.matches(Qt.KeyboardShortcut.StandardKey.Copy):
            self._copy_selection()
        else:
            QTableWidget.keyPressEvent(self.table, event)

    def _on_table_context_menu(self, pos):
        """右键菜单：复制 / 输出阻抗。"""
        menu = QMenu()
        copy_action = menu.addAction("复制")
        edi_action = menu.addAction("输出阻抗...")
        action = menu.exec(self.table.viewport().mapToGlobal(pos))
        if action is copy_action:
            self._copy_selection()
        elif action is edi_action:
            self._export_edi()

    def _copy_selection(self):
        """将当前标签页的选中内容以 TSV 格式复制到剪贴板。"""
        table = self.result_tabs.currentWidget()
        if not isinstance(table, QTableWidget):
            return
        selected = table.selectedRanges()
        if not selected:
            return
        rows = set()
        for r in selected:
            for row in range(r.topRow(), r.bottomRow() + 1):
                rows.add(row)
        lines = []
        lines.append("\t".join(
            table.horizontalHeaderItem(c).text()
            for c in range(table.columnCount())
        ))
        for row in sorted(rows):
            cells = []
            for col in range(table.columnCount()):
                item = table.item(row, col)
                cells.append(item.text() if item is not None else "")
            lines.append("\t".join(cells))
        QApplication.clipboard().setText("\n".join(lines))

    def _export_edi(self):
        """将当前结果聚合为频带后导出 EDI + Z-file 到指定文件夹。"""
        from PySide6.QtWidgets import QFileDialog

        if self._bands is None:
            QMessageBox.warning(self, "无结果", "请先运行估计。")
            return

        folder = QFileDialog.getExistingDirectory(self, "选择导出文件夹")
        if not folder:
            return

        self._bands.export_to_folder(
            folder,
            station_name=self._pkg.get('name', 'unknown'),
            mode="ss",
        )
        self.status_label.setText(f"已导出到 {folder}")