from __future__ import annotations

import logging
import importlib.util
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import pyqtgraph as pg
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import detrend

from ui.theme import configure_matplotlib_canvas, configure_matplotlib_fonts, configure_pyqtgraph_theme

configure_matplotlib_fonts()
configure_pyqtgraph_theme()

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QPushButton, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QSpinBox, QDoubleSpinBox, QLabel, QMenu, QMessageBox, QSplitter, QFrame,
    QDateTimeEdit, QApplication, QStackedWidget,
)
from PySide6.QtCore import Qt, Signal, QDateTime, QTimer

from base.data_manager import DataManager
from core.station_metadata import StationMetadata
from core.em_data import EMData
from emtf.io.zfile import emtf_phase_deg
from output_tools.shared.emtf import (
    BACKEND_EMTF_PYWRAP,
    BACKEND_EMTF_RS,
    CH_COLOURS,
    DEFAULT_OPTION,
    FFT_WINDOW_VALUES,
    WINDOW_VALUES,
    build_band_configs,
    build_spectrum_kwargs,
    detect_channels,
    emtf_period_grid,
    has_magnetic_channel,
    is_e_channel,
    load_emtf_rs_backend,
    native_emtf_rs_available,
    preprocess_channel,
    set_form_row_visible,
)
from plugins.common.plotting import StepPlotWidget
from ui.dialogs.figure_export import default_export_name, export_matplotlib_figure_with_dialog
from utils.decimate import LODPyramid
from utils.time_convert import pts_to_qdt, qdt_to_pts

logger = logging.getLogger(__name__)

# ─── 供本文件内部代码使用的短别名（保持向后兼容）──────────────────
_detect_channels = detect_channels
_is_e_channel = is_e_channel
_has_magnetic_channel = has_magnetic_channel
_set_form_row_visible = set_form_row_visible
_native_emtf_rs_available = native_emtf_rs_available


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
        self.setObjectName("RobustEstimateWidget")
        self.setStyleSheet(
            "QWidget#RobustEstimateWidget, "
            "QWidget#robustControlWidget, "
            "QSplitter#robustMainSplitter, "
            "QSplitter#robustRightSplitter, "
            "QStackedWidget#robustResultStack { background: palette(window); }"
            "QTabWidget::pane, QTableWidget { background: palette(base); }"
            "QLabel#robustPlaceholder { background: palette(window); color: palette(text); }"
        )
        self._data_id = data_id
        self._pkg: dict = {}
        self._station_metadata: Optional[StationMetadata] = None
        self._result = None  # 最近一次计算结果
        self._spectrum = None  # emtf_rs 原始 SpectrumResult
        self._bands = None     # to_bands 聚合后的 BandedResult
        self._region = None
        self._curves: dict[str, pg.PlotDataItem] = {}  # 当前显示的波形曲线
        self._lod_pyramids: dict[str, LODPyramid] = {}  # 每通道的 LOD 金字塔
        self._updating_range = False  # 防止信号递归的标志位
        self._ts_array: Optional[np.ndarray] = None  # 预计算的时间戳缓存，避免每次拖拽区域时重建百万级数组

        # LOD 视口刷新防抖定时器（60fps）
        self._lod_timer = QTimer(self)
        self._lod_timer.setSingleShot(True)
        self._lod_timer.setInterval(16)
        self._lod_timer.timeout.connect(self._lod_update_curves)

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
        self.plot_item.vb.sigXRangeChanged.connect(self._on_lod_viewport_changed)  # LOD 视口变化
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
        self._param_form = param_form

        self.backend_combo = QComboBox()
        self.backend_combo.addItem("EMTF pywrap", BACKEND_EMTF_PYWRAP)
        self.backend_combo.addItem("emtf-rs", BACKEND_EMTF_RS)
        if not _native_emtf_rs_available():
            self.backend_combo.setItemText(1, "emtf-rs（未安装）")
            self.backend_combo.model().item(1).setEnabled(False)
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        param_form.addRow("估计后端:", self.backend_combo)

        self.sample_rate_label = QLabel("—")
        param_form.addRow("采样率 (Hz):", self.sample_rate_label)

        self.window_combo = QComboBox()
        self.window_combo.currentIndexChanged.connect(self._update_warning)
        param_form.addRow("目标周期窗样点数:", self.window_combo)

        self.fft_window_combo = QComboBox()
        for value in FFT_WINDOW_VALUES:
            self.fft_window_combo.addItem(str(value), value)
        self.fft_window_combo.setCurrentIndex(FFT_WINDOW_VALUES.index(128))
        self.fft_window_combo.setToolTip("写入 EMTF decset.cfg 的 nwin，每段 FFT 短窗的样点数")
        param_form.addRow("FFT短窗点数:", self.fft_window_combo)

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

        self._rs_only_fields = [
            self.step_pct_spin,
            self.huber_spin,
            self.max_iter_spin,
            self.tol_spin,
        ]
        self._on_backend_changed()

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
        self.export_figure_btn = QPushButton("输出图像")
        self.export_figure_btn.clicked.connect(self._export_current_figure)
        self.export_figure_btn.setVisible(False)

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

        self.tipper_table = QTableWidget()
        self.tipper_table.setColumnCount(6)
        self.tipper_table.setHorizontalHeaderLabels([
            "freq (Hz)", "period (s)",
            "Tzx", "Tzy",
            "Tzx SE", "Tzy SE",
        ])
        self.tipper_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.tipper_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tipper_table.customContextMenuRequested.connect(self._on_table_context_menu)
        self.tipper_table.keyPressEvent = self._table_key_press_event
        self._tipper_tab_index: Optional[int] = None

        self.tipper_fig = Figure(figsize=(6, 4), tight_layout=True)
        self.tipper_canvas = FigureCanvas(self.tipper_fig)
        self.tipper_canvas.setParent(self)
        configure_matplotlib_canvas(self.tipper_canvas, self.tipper_fig, self)
        self._tipper_plot_tab_index: Optional[int] = None

        # 结果标签页
        self.result_tabs = QTabWidget()
        tab_corner = QWidget()
        tab_corner_layout = QHBoxLayout(tab_corner)
        tab_corner_layout.setContentsMargins(0, 0, 4, 0)
        tab_corner_layout.addWidget(self.export_figure_btn)
        tab_corner_layout.addWidget(self.export_edi_btn)
        self.tipper_mode_combo = QComboBox(self)
        self.tipper_mode_combo.addItem("倾子：实部/虚部", "re_im")
        self.tipper_mode_combo.addItem("倾子：幅值/相位", "amp_phase")
        self.tipper_mode_combo.currentIndexChanged.connect(self._draw_tipper_plot)
        self.tipper_mode_combo.setVisible(False)
        tab_corner_layout.addWidget(self.tipper_mode_combo)
        self.result_tabs.setCornerWidget(tab_corner)  # 右上角导出按钮
        self.result_tabs.currentChanged.connect(self._sync_result_actions)
        self.result_tabs.addTab(self.table, "阻抗结果")
        self.result_tabs.addTab(self.resistivity_table, "视电阻率与相位")

        # 第三标签页：matplotlib 视电阻率/相位对比图
        self.mpl_fig = Figure(figsize=(6, 5), tight_layout=True)
        self.mpl_canvas = FigureCanvas(self.mpl_fig)
        self.mpl_canvas.setParent(self)
        configure_matplotlib_canvas(self.mpl_canvas, self.mpl_fig, self)
        self.result_tabs.addTab(self.mpl_canvas, "视电阻率与相位图")

        # 计算前显示占位提示，计算后切换到结果标签页
        self._placeholder = QLabel("此处显示计算结果信息")
        self._placeholder.setObjectName("robustPlaceholder")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("font-size: 14px;")
        self._result_stack = QStackedWidget()
        self._result_stack.setObjectName("robustResultStack")
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
        control_widget.setObjectName("robustControlWidget")
        control_widget.setLayout(control_panel)
        control_widget.setFixedWidth(340)

        # 右侧：上方波形图 + 下方结果区，用竖向 Splitter 分隔
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setObjectName("robustRightSplitter")
        right_splitter.addWidget(self.plot_widget)
        right_splitter.addWidget(self._result_stack)
        right_splitter.setStretchFactor(0, 0)  # 波形区不随窗口拉伸
        right_splitter.setStretchFactor(1, 1)  # 结果区占据额外空间
        right_splitter.setSizes([200, 600])

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setObjectName("robustMainSplitter")
        main_splitter.addWidget(control_widget)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([340, 960])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(main_splitter)
        self._sync_result_actions()

        # ── 初始填充 ── 数据到达后由 _on_data_ready 触发

    def _current_data(self) -> Optional[dict]:
        """返回当前数据包，无数据时返回 None。"""
        return self._pkg if self._pkg else None

    # ─── 数据接收 ──────────────────────────────────────────────

    def _on_data_ready(self, data_id: int, channel: str, pkg: dict) -> None:
        if data_id != self._data_id:
            return
        self._pkg = pkg
        self._station_metadata = pkg['station_metadata']
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
        self._sync_tipper_tab_visibility()
        self._refresh_plot()

    def _hz_channel_selected(self) -> bool:
        return self.ch_combo_hz.currentData() is not None

    def _sync_tipper_tab_visibility(self) -> None:
        should_show = self._hz_channel_selected()
        table_index = self.result_tabs.indexOf(self.tipper_table)
        plot_index = self.result_tabs.indexOf(self.tipper_canvas)
        if should_show:
            self.tipper_mode_combo.setVisible(True)
            if table_index < 0:
                self._tipper_tab_index = self.result_tabs.addTab(self.tipper_table, "倾子响应")
            if plot_index < 0:
                self._tipper_plot_tab_index = self.result_tabs.addTab(self.tipper_canvas, "倾子响应图")
        else:
            self.tipper_mode_combo.setVisible(False)
            if plot_index >= 0:
                self.result_tabs.removeTab(plot_index)
                self._tipper_plot_tab_index = None
            table_index = self.result_tabs.indexOf(self.tipper_table)
            if table_index >= 0:
                self.result_tabs.removeTab(table_index)
            self._tipper_tab_index = None
            self.tipper_table.setRowCount(0)
            self.tipper_fig.clear()
            self.tipper_canvas.draw()
        self._sync_result_actions()

    def _current_matplotlib_export_target(self):
        current = self.result_tabs.currentWidget()
        mpl_canvas = getattr(self, "mpl_canvas", None)
        tipper_canvas = getattr(self, "tipper_canvas", None)
        if current is mpl_canvas:
            return self.mpl_fig, self.mpl_canvas, "resistivity_phase"
        if current is tipper_canvas:
            return self.tipper_fig, self.tipper_canvas, "tipper"
        return None

    def _sync_result_actions(self, _index: int = -1) -> None:
        self.export_figure_btn.setVisible(self._current_matplotlib_export_target() is not None)

    def _export_current_figure(self) -> None:
        target = self._current_matplotlib_export_target()
        if target is None:
            return
        figure, canvas, plot_slug = target
        station = self._station_metadata.name if self._station_metadata is not None else self._pkg.get("name", "station")
        path = export_matplotlib_figure_with_dialog(
            self,
            figure,
            canvas,
            default_directory=Path.home(),
            default_name=default_export_name(station, plot_slug),
        )
        if path is not None:
            self.status_label.setText(f"已输出图像到 {path}")

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
        """刷新波形预览：清除旧曲线，按当前通道选择重新绑图。

        每通道构建 LODPyramid（Rust 加速），曲线初始填入概览级数据；
        后续 pan/zoom 时由 sigXRangeChanged → _lod_update_curves 动态查询。
        """
        pkg = self._pkg
        if not pkg:
            return
        all_y = pkg.get('all_y', {})
        x = pkg.get('x', np.array([]))
        if len(x) == 0:
            return

        # 清除旧曲线、金字塔和图例
        for curve in list(self._curves.values()):
            curve_view = curve.getViewBox()
            if curve_view is self._b_viewbox:
                self._b_viewbox.removeItem(curve)
            else:
                self.plot_item.vb.removeItem(curve)
        self._curves.clear()
        self._lod_pyramids.clear()
        self._legend.clear()

        # 构建时间戳数组（同时缓存，供拖拽区域时二分查找使用）
        t = np.array([t.timestamp() for t in pkg['datetime_index']])
        self._ts_array = t

        channels = self._selected_channels()
        pw = max(self.plot_widget.width(), 800)

        for tag, ch_name in channels.items():
            if ch_name is None:
                continue
            y = all_y.get(ch_name)
            if y is None:
                continue
            # 去均值以便在同一坐标系下对比波形形态
            y = y - np.nanmean(y)

            # 构建 LOD 金字塔（Rust build_lod_pyramid_rs 加速）
            pyramid = LODPyramid(t, y)
            self._lod_pyramids[tag] = pyramid

            # 初始数据：取概览级，避免百万点全量塞入 PlotDataItem
            ox, oy = pyramid.get_full_overview(pw)

            colour = CH_COLOURS.get(tag, (100, 100, 100))
            pen = pg.mkPen(color=colour, width=1)
            label = f"{tag} ({ch_name})"

            if _is_e_channel(ch_name):
                # 电场 → 左轴（主 ViewBox）
                curve = pg.PlotDataItem(ox, oy, pen=pen, name=label,
                                        autoDownsample=False, clipToView=False,
                                        connect='finite')
                curve.setParentItem(self.plot_item.vb)
                curve.forgetViewBox()
                self.plot_item.vb.addItem(curve)
            else:
                # 磁场 → 右轴（B ViewBox）
                curve = pg.PlotDataItem(ox, oy, pen=pen, name=label,
                                        autoDownsample=False, clipToView=False,
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

    # ─── LOD 视口刷新 ──────────────────────────────────────────────

    def _on_lod_viewport_changed(self):
        """ViewBox X 范围变化 → 启动防抖定时器，16ms 后批量刷新所有曲线。"""
        self._lod_timer.start()

    def _lod_update_curves(self):
        """从各通道的 LODPyramid 查询当前视口适配分辨率的数据并 setData。"""
        if not self._lod_pyramids:
            return
        vb = self.plot_item.vb
        x_start, x_end = vb.viewRange()[0]
        pw = max(self.plot_widget.width(), 200)
        for tag, pyramid in self._lod_pyramids.items():
            curve = self._curves.get(tag)
            if curve is None:
                continue
            lx, ly = pyramid.get(x_start, x_end, pw)
            curve.setData(lx, ly)

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
        return max(0, start), min(self._pkg['npts'], end)

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
        if self._current_backend() == BACKEND_EMTF_PYWRAP:
            self.warning_label.setText("")
            return
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

    def _current_backend(self) -> str:
        return str(self.backend_combo.currentData())

    def _on_backend_changed(self):
        backend = self._current_backend()
        self._set_window_values(WINDOW_VALUES)
        rs_visible = backend == BACKEND_EMTF_RS
        for field in self._rs_only_fields:
            _set_form_row_visible(self._param_form, field, rs_visible)
        _set_form_row_visible(self._param_form, self.fft_window_combo, not rs_visible)
        if backend == BACKEND_EMTF_PYWRAP:
            self.warning_label.setText("")
        else:
            self.bands_spin.setToolTip("对数均分频带数量，影响视电阻率/相位图的频点密度")

    def _set_window_values(self, values: list[int]) -> None:
        current = self.window_combo.currentData()
        if current in values and self.window_combo.count() == len(values):
            return
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        for value in values:
            self.window_combo.addItem(str(value), value)
        if current in values:
            self.window_combo.setCurrentIndex(values.index(current))
        else:
            self.window_combo.setCurrentIndex(1 if len(values) > 1 else 0)
        self.window_combo.blockSignals(False)
        self._update_warning()

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
        backend = self._current_backend()
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
            if backend == BACKEND_EMTF_RS:
                import emtf_rs
            else:
                import emtf as emtf_rs

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

            ex_arr = preprocess_channel(ex_arr, ch_ex)
            ey_arr = preprocess_channel(ey_arr, ch_ey)
            hx_arr = preprocess_channel(hx_arr, ch_hx)
            hy_arr = preprocess_channel(hy_arr, ch_hy)
            if hz_arr is not None:
                hz_arr = preprocess_channel(hz_arr, str(ch_hz))

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

            self.status_label.setText("正在计算...")
            self.run_button.setEnabled(False)
            QApplication.processEvents()  # 强制刷新 UI，显示"正在计算"

            station_name = self._station_metadata.name if self._station_metadata else "STATION"
            kwargs = build_spectrum_kwargs(
                sample_rate=sample_rate,
                window_size=window_size,
                step_pct=self.step_pct_spin.value(),
                huber_threshold=self.huber_spin.value(),
                max_iterations=self.max_iter_spin.value(),
                tolerance=self.tol_spin.value(),
                hz_arr=hz_arr,
                backend=backend,
                fft_window_size=self.fft_window_combo.currentData() if backend == BACKEND_EMTF_PYWRAP else None,
                n_bands=self.bands_spin.value(),
                station_name=station_name,
            )
            result = emtf_rs.estimate_single_station_spectrum(
                ex_arr, ey_arr, hx_arr, hy_arr, **kwargs
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
        if self._spectrum is None:
            self._bands = None
            return
        if self._current_backend() == BACKEND_EMTF_PYWRAP:
            from emtf.io import from_result

            self._bands = from_result(self._spectrum, station_name=self._spectrum.station_name, mode="ss")
            return
        data = self._current_data()
        if data is None:
            self._bands = None
            return
        dt = self._pkg['dt']
        w = self.window_combo.currentData()
        n_bands = self.bands_spin.value()
        periods = np.logspace(np.log10(2 * dt), np.log10(w * dt), n_bands)
        band_configs = build_band_configs(periods)
        self._bands = self._spectrum.to_bands(band_configs)

    def _emtf_period_grid(self, sample_rate: float, window_size: int, fft_window_size: int):
        return emtf_period_grid(sample_rate, window_size, fft_window_size, self.bands_spin.value())

    def _on_bands_changed(self):
        """频带数量变更时重建 bands 并刷新对比图。"""
        if self._spectrum is None:
            return
        self._build_bands()
        self._display_tipper_result()
        self._draw_comparison_plot()

    def _display_tipper_result(self):
        """Fill the tipper table from the banded result used by EDI/Z-file export."""
        self._sync_tipper_tab_visibility()
        bands = self._bands
        if bands is None or not self._hz_channel_selected():
            self.tipper_table.setRowCount(0)
            self._draw_tipper_plot()
            return

        tzx = getattr(bands, "tzx", None)
        tzy = getattr(bands, "tzy", None)
        tzx_se = getattr(bands, "tzx_se", None)
        tzy_se = getattr(bands, "tzy_se", None)
        if tzx is None or tzy is None or tzx_se is None or tzy_se is None:
            self.tipper_table.setRowCount(0)
            self._draw_tipper_plot()
            return

        freqs = np.asarray(getattr(bands, "center_frequencies_hz", []), dtype=float)
        periods = np.asarray(getattr(bands, "periods", []), dtype=float)
        tzx = np.asarray(tzx)
        tzy = np.asarray(tzy)
        tzx_se = np.asarray(tzx_se, dtype=float)
        tzy_se = np.asarray(tzy_se, dtype=float)
        n = min(len(freqs), len(periods), len(tzx), len(tzy), len(tzx_se), len(tzy_se))

        def _c2s(c: complex) -> str:
            return f"{c.real:.4e}{c.imag:+.4e}j"

        self.tipper_table.setRowCount(n)
        for i in range(n):
            self.tipper_table.setItem(i, 0, QTableWidgetItem(f"{freqs[i]:.6f}"))
            self.tipper_table.setItem(i, 1, QTableWidgetItem(f"{periods[i]:.3f}"))
            self.tipper_table.setItem(i, 2, QTableWidgetItem(_c2s(tzx[i])))
            self.tipper_table.setItem(i, 3, QTableWidgetItem(_c2s(tzy[i])))
            self.tipper_table.setItem(i, 4, QTableWidgetItem(f"{tzx_se[i]:.4e}"))
            self.tipper_table.setItem(i, 5, QTableWidgetItem(f"{tzy_se[i]:.4e}"))
        self._draw_tipper_plot()

    def _draw_tipper_plot(self):
        """Draw Tzx/Tzy real and imaginary components with error bars."""
        result = self._bands
        if result is None or not self._hz_channel_selected():
            self.tipper_fig.clear()
            configure_matplotlib_canvas(self.tipper_canvas, self.tipper_fig, self)
            self.tipper_canvas.draw()
            return

        tzx = getattr(result, "tzx", None)
        tzy = getattr(result, "tzy", None)
        tzx_se = getattr(result, "tzx_se", None)
        tzy_se = getattr(result, "tzy_se", None)
        if tzx is None or tzy is None or tzx_se is None or tzy_se is None:
            self.tipper_fig.clear()
            configure_matplotlib_canvas(self.tipper_canvas, self.tipper_fig, self)
            self.tipper_canvas.draw()
            return

        p = np.asarray(result.periods, dtype=float)
        tzx = np.asarray(tzx, dtype=np.complex128)
        tzy = np.asarray(tzy, dtype=np.complex128)
        tzx_se = np.asarray(tzx_se, dtype=float)
        tzy_se = np.asarray(tzy_se, dtype=float)
        n = min(len(p), len(tzx), len(tzy), len(tzx_se), len(tzy_se))
        finite = (
            np.isfinite(p[:n]) & (p[:n] > 0)
            & np.isfinite(tzx[:n].real) & np.isfinite(tzx[:n].imag)
            & np.isfinite(tzy[:n].real) & np.isfinite(tzy[:n].imag)
        )
        if not finite.any():
            self.tipper_fig.clear()
            configure_matplotlib_canvas(self.tipper_canvas, self.tipper_fig, self)
            self.tipper_canvas.draw()
            return

        p = p[:n][finite]
        tzx = tzx[:n][finite]
        tzy = tzy[:n][finite]
        tzx_se = tzx_se[:n][finite]
        tzy_se = tzy_se[:n][finite]

        self.tipper_fig.clear()
        ax1, ax2 = self.tipper_fig.subplots(2, 1)
        amp_phase = getattr(self, "tipper_mode_combo", None)
        amp_phase = amp_phase is not None and amp_phase.currentData() == "amp_phase"
        if amp_phase:
            ax1.errorbar(p, np.abs(tzx), yerr=tzx_se, fmt='o', color='purple',
                         capsize=3, markersize=5, label='|Tzx|')
            ax1.errorbar(p, np.abs(tzy), yerr=tzy_se, fmt='s', color='green',
                         capsize=3, markersize=5, label='|Tzy|')
        else:
            ax1.errorbar(p, tzx.real, yerr=tzx_se, fmt='o', color='purple',
                         capsize=3, markersize=5, label='Re(Tzx)')
            ax1.errorbar(p, tzy.real, yerr=tzy_se, fmt='s', color='green',
                         capsize=3, markersize=5, label='Re(Tzy)')
        ax1.set_xscale('log')
        ax1.set_ylabel('Amplitude' if amp_phase else 'Real')
        ax1.set_title('倾子响应幅值' if amp_phase else '倾子响应实部')
        ax1.legend(loc='best', fontsize=7, framealpha=0.4, edgecolor='none')
        ax1.grid(True, alpha=0.3)

        if amp_phase:
            ax2.errorbar(p, emtf_phase_deg(tzx), yerr=np.degrees(tzx_se / np.maximum(np.abs(tzx), 1e-20)), fmt='o', color='purple',
                         capsize=3, markersize=5, label='arg(Tzx)')
            ax2.errorbar(p, emtf_phase_deg(tzy), yerr=np.degrees(tzy_se / np.maximum(np.abs(tzy), 1e-20)), fmt='s', color='green',
                         capsize=3, markersize=5, label='arg(Tzy)')
        else:
            ax2.errorbar(p, tzx.imag, yerr=tzx_se, fmt='o', color='purple',
                         capsize=3, markersize=5, label='Im(Tzx)')
            ax2.errorbar(p, tzy.imag, yerr=tzy_se, fmt='s', color='green',
                         capsize=3, markersize=5, label='Im(Tzy)')
        ax2.set_xscale('log')
        ax2.set_xlabel('Period (s)')
        ax2.set_ylabel('Phase (°)' if amp_phase else 'Imaginary')
        ax2.set_title('倾子响应相位' if amp_phase else '倾子响应虚部')
        ax2.legend(loc='best', fontsize=7, framealpha=0.4, edgecolor='none')
        ax2.grid(True, alpha=0.3)
        configure_matplotlib_canvas(self.tipper_canvas, self.tipper_fig, self)
        self.tipper_canvas.draw()

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
        phi_xy_values = emtf_phase_deg(np.asarray(result.zxy))
        phi_yx_values = emtf_phase_deg(np.asarray(result.zyx))
        for i in range(n):
            f = freqs[i]
            period = 1.0 / f if f > 0 else float("inf")
            self.resistivity_table.setItem(i, 0, QTableWidgetItem(f"{f:.6f}"))
            self.resistivity_table.setItem(i, 1, QTableWidgetItem(f"{period:.3f}"))
            self.resistivity_table.setItem(i, 2, QTableWidgetItem(f"{result.rho_xy[i]:.4e}"))
            self.resistivity_table.setItem(i, 3, QTableWidgetItem(f"{result.rho_yx[i]:.4e}"))
            self.resistivity_table.setItem(i, 4, QTableWidgetItem(f"{phi_xy_values[i]:.4f}"))
            self.resistivity_table.setItem(i, 5, QTableWidgetItem(f"{phi_yx_values[i]:.4f}"))
            self.resistivity_table.setItem(i, 6, QTableWidgetItem(f"{result.rho_xy_se[i]:.4e}"))
            self.resistivity_table.setItem(i, 7, QTableWidgetItem(f"{result.rho_yx_se[i]:.4e}"))
            self.resistivity_table.setItem(i, 8, QTableWidgetItem(f"{result.phi_xy_se[i]:.4f}"))
            self.resistivity_table.setItem(i, 9, QTableWidgetItem(f"{result.phi_yx_se[i]:.4f}"))

        # ── 倾子响应结果表（与导出一致，使用频带聚合结果） ──
        self._display_tipper_result()

        # ── matplotlib 对比图 ──
        self._draw_comparison_plot()

    def _draw_comparison_plot(self):
        """用 matplotlib 绘制视电阻率和相位的误差棒对比图。"""
        if self._bands is None:
            self.mpl_fig.clear()
            configure_matplotlib_canvas(self.mpl_canvas, self.mpl_fig, self)
            self.mpl_canvas.draw()
            return

        result = self._bands

        p = np.array(result.periods)
        finite = np.isfinite(p) & (p > 0)
        if not finite.any():
            self.mpl_fig.clear()
            configure_matplotlib_canvas(self.mpl_canvas, self.mpl_fig, self)
            self.mpl_canvas.draw()
            return

        p = p[finite]
        rho_xy = np.array(result.rho_xy)[finite]
        rho_xy_se = np.array(result.rho_xy_se)[finite]
        phi_xy = emtf_phase_deg(np.asarray(result.zxy))[finite]
        phi_xy_se = np.array(result.phi_xy_se)[finite]
        rho_yx = np.array(result.rho_yx)[finite]
        rho_yx_se = np.array(result.rho_yx_se)[finite]
        phi_yx = emtf_phase_deg(np.asarray(result.zyx))[finite]
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
        ax1.legend(loc='best', fontsize=7, framealpha=0.4, edgecolor='none')
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
        # 与 EMTF MATLAB 的 set_lims/pltrhom 对齐：相位绘图区固定为
        # [-180, 180]。phi_se 仍保留原始标准误，不在数据层截断。
        ax2.set_ylim(-180.0, 180.0)
        ax2.legend(loc='best', fontsize=7, framealpha=0.4, edgecolor='none')
        ax2.grid(True, alpha=0.3)

        configure_matplotlib_canvas(self.mpl_canvas, self.mpl_fig, self)
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
        table = self.sender()
        if not isinstance(table, QTableWidget):
            table = self.table
        menu = QMenu()
        copy_action = menu.addAction("复制")
        edi_action = menu.addAction("输出阻抗...")
        action = menu.exec(table.viewport().mapToGlobal(pos))
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
        """将当前频带结果导出为用户指定路径的 EDI 文件。"""
        from PySide6.QtWidgets import QFileDialog
        from pathlib import Path

        if self._bands is None:
            QMessageBox.warning(self, "无结果", "请先运行估计。")
            return

        if not self._pkg:
            QMessageBox.warning(self, "无数据", "当前没有台站数据。")
            return
        if self._station_metadata is None:
            QMessageBox.warning(self, "无数据", "当前台站缺少元数据。")
            return

        station_meta = self._station_metadata
        station_name = station_meta.name.strip() or "unknown"
        default_path = f"{station_name}.edi"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "导出 EDI 阻抗文件",
            default_path,
            "EDI 文件 (*.edi);;所有文件 (*)",
        )
        if not filename:
            return

        path = Path(filename)
        if path.suffix.lower() != ".edi":
            path = path.with_suffix(".edi")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            lat = station_meta.latitude if station_meta.latitude is not None else 0.0
            lon = station_meta.longitude if station_meta.longitude is not None else 0.0
            elev = station_meta.elevation if station_meta.elevation is not None else 0.0
            if self._current_backend() == BACKEND_EMTF_RS:
                import emtf_rs

                meta = emtf_rs.PyStationMetadata(
                    station_name=station_name,
                    project_name=station_name,
                    latitude=lat,
                    longitude=lon,
                    elevation=elev,
                    processing_mode="ss",
                    sample_rate_hz=1.0 / self._pkg["dt"],
                )
                edi_text = self._bands.format_edi_str(meta)
            else:
                edi_text = self._bands.format_edi_str(
                    station_name=station_name,
                    mode="ss",
                    lat=lat,
                    lon=lon,
                    elev=elev,
                )
            path.write_text(edi_text, encoding="ascii")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", f"无法导出 EDI 文件：{exc}")
            return

        self.status_label.setText(f"已导出 EDI 到 {path}")
