from __future__ import annotations

import logging
import importlib.util
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pandas as pd
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import detrend

from core.station_metadata import StationMetadata
from core.em_data import EMData
from emtf.io.zfile import emtf_phase_deg
from emtf.time_alignment import matching_remote_slice

from ui.theme import (
    add_matplotlib_reset_button,
    configure_matplotlib_canvas,
    configure_matplotlib_fonts,
    configure_pyqtgraph_theme,
)

configure_matplotlib_fonts()
configure_pyqtgraph_theme()

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QPushButton, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QSpinBox, QDoubleSpinBox, QLabel, QMenu, QMessageBox, QSplitter, QFrame,
    QDateTimeEdit, QApplication, QStackedWidget, QDialog,
)
from PySide6.QtCore import Qt, Signal, QDateTime, QTimer

from ui.impedance_plot import draw_impedance_response, draw_tipper_response, draw_c_response
from output_tools.shared.c_response import c_response_from_bands
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
    has_electric_channel,
    has_magnetic_channel,
    is_e_channel,
    is_h_channel,
    native_emtf_rs_available,
    preprocess_channel,
    set_form_row_visible,
)
from plugins.common.plotting import StepPlotWidget
from ui.dialogs.figure_export import default_export_name, export_matplotlib_figure_with_dialog
from utils.decimate import LODPyramid
from base.qt_bridge import run as run_in_pool
from utils.time_convert import pts_to_qdt, qdt_to_pts
from utils.geo import station_distance

logger = logging.getLogger(__name__)


def _estimate_remote_task(backend: str, arrays: tuple, channel_names: tuple, kwargs: dict, generation: int):
    try:
        ex, ey, local_hx, local_hy, remote_hx, remote_hy, hz = arrays
        processed = tuple(
            preprocess_channel(values, name)
            for values, name in zip(
                (ex, ey, local_hx, local_hy, remote_hx, remote_hy),
                channel_names[:6],
            )
        )
        call_kwargs = dict(kwargs)
        call_kwargs["hz"] = preprocess_channel(hz, str(channel_names[6])) if hz is not None else None
        if backend == BACKEND_EMTF_RS:
            from base.emtf_native import estimate_native_emtf
            result = estimate_native_emtf(*processed, remote_reference=True, **call_kwargs)
        else:
            import emtf
            result = emtf.estimate_remote_reference_spectrum(*processed, **call_kwargs)
        return generation, result, None
    except BaseException as error:
        return generation, None, error


def _build_lod_task(side: str, entries: list[tuple], pixel_width: int, generation: int):
    built = []
    for tag, channel_name, electric, timestamps, values in entries:
        values = np.asarray(values, dtype=np.float64)
        mean = np.nanmean(values)
        centered = values - (mean if np.isfinite(mean) else 0.0)
        pyramid = LODPyramid(timestamps, centered)
        built.append((tag, channel_name, electric, pyramid, pyramid.get_full_overview(pixel_width)))
    return side, generation, built

# ─── 供本文件内部代码使用的短别名（保持向后兼容）──────────────────
_detect_channels = detect_channels
_is_e_channel = is_e_channel
_is_h_channel = is_h_channel
_has_magnetic_channel = has_magnetic_channel
_has_electric_channel = has_electric_channel
_set_form_row_visible = set_form_row_visible
_native_emtf_rs_available = native_emtf_rs_available


class RemoteReferenceControlDialog(QDialog):
    """远参考法调参面板（独立浮窗）。"""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        w = parent  # RemoteReferenceWidget
        self.setWindowTitle("远参考法 — 参数调整")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 第一行：台站并排
        station_row = QHBoxLayout()
        station_row.setSpacing(8)
        station_row.addWidget(w._local_station_widget)
        station_row.addWidget(w._remote_station_widget)
        layout.addLayout(station_row)

        # 第二行：距离 + 采样率
        info_row = QHBoxLayout()
        info_row.setSpacing(12)
        info_row.addWidget(w.distance_label)
        info_row.addWidget(w.sample_rate_warn)
        info_row.addStretch()
        layout.addLayout(info_row)

        # 下半部分：左右两栏
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        # 左栏：时间范围 + 通道映射
        left_col = QVBoxLayout()
        left_col.setSpacing(6)
        left_col.addWidget(w._time_group)
        left_col.addWidget(w._local_ch_group)
        left_col.addWidget(w._remote_ch_group)
        left_col.addStretch()
        bottom.addLayout(left_col)

        # 右栏：估计参数 + 按钮 + 状态
        right_col = QVBoxLayout()
        right_col.setSpacing(6)
        right_col.addWidget(w._param_group)
        right_col.addWidget(w._btn_widget)
        right_col.addWidget(w.status_label)
        right_col.addStretch()
        bottom.addLayout(right_col)

        layout.addLayout(bottom)


class _PkgWrapper:
    """将 Broker 数据包包装为 EMData 兼容接口。"""
    __slots__ = ('_pkg',)

    def __init__(self, pkg: dict):
        self._pkg = pkg

    @property
    def npts(self) -> int: return self._pkg.get('npts', 0)
    @property
    def dt(self): return pd.Timedelta(seconds=self._pkg.get('dt', 1.0))
    @property
    def start_time(self):
        return pd.Timestamp(self._pkg.get('start_time', 0), unit='s', tz='Asia/Shanghai')
    @property
    def end_time(self):
        s, dt, n = self._pkg.get('start_time', 0), self._pkg.get('dt', 1), self._pkg.get('npts', 0)
        return pd.Timestamp(s + dt * n, unit='s', tz='Asia/Shanghai') if n > 0 else self.start_time
    @property
    def datetime_index(self): return self._pkg.get('datetime_index')
    @property
    def station_metadata(self) -> StationMetadata: return self._pkg['station_metadata']
    @property
    def name(self) -> str: return self.station_metadata.name
    @property
    def latitude(self): return self.station_metadata.latitude
    @property
    def longitude(self): return self.station_metadata.longitude
    @property
    def elevation(self): return self.station_metadata.elevation
    def channel_names(self) -> list[str]: return self._pkg.get('channel_names', [])
    def get_channel_series(self, ch: str) -> pd.Series:
        arr = self._pkg.get('all_y', {}).get(ch, np.array([]))
        idx = self.datetime_index
        return pd.Series(arr, index=idx if idx is not None else range(len(arr)), name=ch)


class RemoteReferenceWidget(QWidget):
    """远参考法阻抗估计交互界面。

    上部为本地站和远参考站波形图左右并排，
    下部为计算结果表格和对比图，
    控制参数通过独立浮窗调节。
    """
    result_ready = Signal(object)
    snapshot_save_requested = Signal(object)
    preview_requested = Signal()

    data_requested = Signal(int, str)
    result_submitted = Signal(int, str, object)

    def __init__(
        self,
        data_id: int = 0,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("RemoteReferenceWidget")
        self.setStyleSheet(
            "QWidget#RemoteReferenceWidget, "
            "QWidget#remotePlotRowWidget, "
            "QSplitter#remoteMainSplitter, "
            "QStackedWidget#remoteResultStack { background: palette(window); }"
            "QTabWidget::pane, QTableWidget { background: palette(base); }"
            "QLabel#remotePlaceholder { background: palette(window); color: palette(text); }"
        )
        self._data_id = data_id
        self._local_pkg: dict = {}
        self._remote_pkg: dict = {}
        self._result = None
        self._spectrum = None  # emtf_rs 原始 SpectrumResult
        self._bands = None     # to_bands 聚合后的 BandedResult
        self._calculation_context: dict = {}
        self._saved_current = True
        self._pending_context: dict = {}
        self._estimation_future = None
        self._task_generation = 0
        self._local_plot_generation = 0
        self._remote_plot_generation = 0
        self._curves_local: dict[str, pg.PlotDataItem] = {}
        self._curves_remote: dict[str, pg.PlotDataItem] = {}
        self._lod_local: dict[str, LODPyramid] = {}   # 本地站每通道 LOD 金字塔
        self._lod_remote: dict[str, LODPyramid] = {}  # 远参考站每通道 LOD 金字塔
        self._updating_range = False
        self._syncing_region = False  # 防止两个 LinearRegionItem 递归同步
        self._syncing_xrange = False  # 防止两个图 X 轴递归同步
        self._ts_array: np.ndarray | None = None
        self._ts_remote_array: np.ndarray | None = None
        self._control_dialog: RemoteReferenceControlDialog | None = None

        # LOD 视口刷新防抖定时器（60fps）
        self._lod_timer = QTimer(self)
        self._lod_timer.setSingleShot(True)
        self._lod_timer.setInterval(16)
        self._lod_timer.timeout.connect(self._lod_update_curves)

        # ── 本地站波形预览（双 Y 轴） ──
        self.local_plot = StepPlotWidget()
        self.local_plot.showGrid(x=True, y=True, alpha=0.3)
        self.local_plot.setLabel('left', 'E (mV/km)')
        self.local_plot.setLabel('bottom', '时间')
        self._local_legend = self.local_plot.addLegend(offset=(0, 10))
        self._local_plot_item = self.local_plot.getPlotItem()

        self.local_plot.showAxis('right')
        self.local_plot.setLabel('right', 'B (nT)')
        self._local_b_viewbox = pg.ViewBox()
        self._local_plot_item.scene().addItem(self._local_b_viewbox)
        self._local_plot_item.getAxis('right').linkToView(self._local_b_viewbox)
        self._local_b_viewbox.setXLink(self._local_plot_item)
        self._local_plot_item.vb.sigResized.connect(self._update_local_dual_geometry)
        self.local_plot.zoom_callback = self._sync_local_b_viewbox_zoom

        # 本地站 LinearRegionItem
        self.region_item = pg.LinearRegionItem(values=[0, 1], movable=True)
        border_pen = pg.mkPen(color=(65, 105, 225), width=1)
        self.region_item.lines[0].setPen(border_pen)
        self.region_item.lines[1].setPen(border_pen)
        self.region_item.setBrush(pg.mkBrush(135, 206, 250, 50))
        self.local_plot.addItem(self.region_item)
        self.region_item.sigRegionChanged.connect(self._on_local_region_changed)
        self.region_item.sigRegionChangeFinished.connect(self._on_region_change_finished)

        # ── 远参考站波形预览（单 Y 轴，仅磁场） ──
        self.remote_plot = StepPlotWidget()
        self.remote_plot.showGrid(x=True, y=True, alpha=0.3)
        self.remote_plot.setLabel('left', 'B (nT)')
        self.remote_plot.setLabel('bottom', '时间')
        self._remote_legend = self.remote_plot.addLegend(offset=(0, 10))
        self._remote_plot_item = self.remote_plot.getPlotItem()

        # 远参考站 LinearRegionItem
        self.remote_region_item = pg.LinearRegionItem(values=[0, 1], movable=True)
        self.remote_region_item.lines[0].setPen(border_pen)
        self.remote_region_item.lines[1].setPen(border_pen)
        self.remote_region_item.setBrush(pg.mkBrush(135, 206, 250, 50))
        self.remote_plot.addItem(self.remote_region_item)
        self.remote_region_item.sigRegionChanged.connect(self._on_remote_region_changed)
        self.remote_region_item.sigRegionChangeFinished.connect(self._on_region_change_finished)

        # X 轴双向同步
        self._local_plot_item.vb.sigRangeChanged.connect(self._on_local_xrange_changed)
        self._remote_plot_item.vb.sigRangeChanged.connect(self._on_remote_xrange_changed)

        # LOD 视口刷新：任一图 X 范围变化 → 防抖批量查询两图所有曲线
        self._local_plot_item.vb.sigXRangeChanged.connect(self._on_lod_viewport_changed)
        self._remote_plot_item.vb.sigXRangeChanged.connect(self._on_lod_viewport_changed)

        # ── 本地站选择器 ──
        self._local_station_widget = QWidget()
        local_station_layout = QHBoxLayout(self._local_station_widget)
        local_station_layout.setContentsMargins(0, 0, 0, 0)
        local_station_layout.addWidget(QLabel("本地站:"))
        self.local_station_combo = QComboBox()
        self.local_station_combo.currentIndexChanged.connect(self._on_local_station_changed)
        local_station_layout.addWidget(self.local_station_combo, 1)

        # ── 远参考站选择器 ──
        self._remote_station_widget = QWidget()
        remote_station_layout = QHBoxLayout(self._remote_station_widget)
        remote_station_layout.setContentsMargins(0, 0, 0, 0)
        remote_station_layout.addWidget(QLabel("远参考站:"))
        self.remote_station_combo = QComboBox()
        self.remote_station_combo.currentIndexChanged.connect(self._on_remote_station_changed)
        remote_station_layout.addWidget(self.remote_station_combo, 1)

        # ── 台站距离 ──
        self.distance_label = QLabel("")
        self.distance_label.setStyleSheet("color: #336699; font-weight: bold;")
        self.sample_rate_warn = QLabel("")
        self.sample_rate_warn.setStyleSheet("color: #cc6600; font-weight: bold;")

        # ── 时间范围编辑区 ──
        self._time_group = QGroupBox("时间范围")
        time_layout = QFormLayout(self._time_group)
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

        self.range_info = QLabel("")
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: #cc6600; font-weight: bold;")
        time_layout.addRow(self.range_info)
        time_layout.addRow(self.warning_label)

        # ── 通道映射（本地站） ──
        self._local_ch_group = QGroupBox("通道映射（本地站）")
        local_ch_form = QFormLayout(self._local_ch_group)
        self.ch_combo_ex = QComboBox()
        self.ch_combo_ey = QComboBox()
        self.ch_combo_hx = QComboBox()
        self.ch_combo_hy = QComboBox()
        self.ch_combo_hz = QComboBox()
        self.ch_combo_hz.addItem("(不使用)", None)
        for combo in [self.ch_combo_ex, self.ch_combo_ey, self.ch_combo_hx, self.ch_combo_hy, self.ch_combo_hz]:
            combo.currentIndexChanged.connect(lambda: self._on_channel_changed())
        local_ch_form.addRow("Ex:", self.ch_combo_ex)
        local_ch_form.addRow("Ey:", self.ch_combo_ey)
        local_ch_form.addRow("Hx:", self.ch_combo_hx)
        local_ch_form.addRow("Hy:", self.ch_combo_hy)
        local_ch_form.addRow("Hz:", self.ch_combo_hz)

        # ── 通道映射（远参考站） ──
        self._remote_ch_group = QGroupBox("通道映射（远参考站）")
        remote_ch_form = QFormLayout(self._remote_ch_group)
        self.ch_combo_remote_hx = QComboBox()
        self.ch_combo_remote_hy = QComboBox()
        self.ch_combo_remote_hx.currentIndexChanged.connect(lambda: self._on_channel_changed())
        self.ch_combo_remote_hy.currentIndexChanged.connect(lambda: self._on_channel_changed())
        remote_ch_form.addRow("Hx:", self.ch_combo_remote_hx)
        remote_ch_form.addRow("Hy:", self.ch_combo_remote_hy)

        # ── 估计参数 ──
        self._param_group = QGroupBox("估计参数")
        param_form = QFormLayout(self._param_group)
        self._param_form = param_form

        self.backend_combo = QComboBox()
        self.backend_combo.addItem("EMTF pywrap", BACKEND_EMTF_PYWRAP)
        self.backend_combo.addItem("emtf-rs", BACKEND_EMTF_RS)
        self.backend_combo.setItemData(1,
            "已对齐原版降采样、预白化、窗函数和 FC 选带；保留双精度输入。"
            "稳健误差棒和有效观测数尚未验证与原版等价。", Qt.ItemDataRole.ToolTipRole)
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
        for control in (self.window_combo, self.fft_window_combo):
            control.currentIndexChanged.connect(self._mark_parameters_changed)
        for control in (self.step_pct_spin, self.huber_spin, self.max_iter_spin, self.tol_spin):
            control.valueChanged.connect(self._mark_parameters_changed)

        # ── 运行按钮 ──
        self._btn_widget = QWidget()
        btn_layout = QHBoxLayout(self._btn_widget)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        self.run_button = QPushButton("运行远参考估计")
        self.run_button.clicked.connect(self._run_estimation)
        self.save_preview_btn = QPushButton("保存到阻抗预览")
        self.save_preview_btn.setEnabled(False)
        self.save_preview_btn.clicked.connect(self._save_to_preview)
        self.view_preview_btn = QPushButton("查看阻抗预览")
        self.view_preview_btn.setVisible(False)
        self.view_preview_btn.clicked.connect(self.preview_requested)
        self.cancel_button = QPushButton("取消计算")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_estimation)
        btn_layout.addStretch()
        btn_layout.addWidget(self.run_button)
        btn_layout.addWidget(self.cancel_button)
        btn_layout.addWidget(self.save_preview_btn)
        btn_layout.addWidget(self.view_preview_btn)

        self.status_label = QLabel("")

        # ── 结果表格 ──
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

        self.tune_btn = QPushButton("参数调整")
        self.tune_btn.clicked.connect(self._show_control_dialog)

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
        self._tipper_tab_index: int | None = None

        self.tipper_fig = Figure(figsize=(6, 4), tight_layout=True)
        self.tipper_canvas = FigureCanvas(self.tipper_fig)
        self.tipper_canvas.setParent(self)
        configure_matplotlib_canvas(self.tipper_canvas, self.tipper_fig, self)
        self._tipper_plot_tab_index: int | None = None

        self.c_response_table = QTableWidget()
        self.c_response_table.setColumnCount(6)
        self.c_response_table.setHorizontalHeaderLabels([
            "freq (Hz)", "period (s)", "Cxx (m)", "Cxy (m)", "Cyx (m)", "Cyy (m)",
        ])
        self.c_response_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.c_response_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.c_response_table.customContextMenuRequested.connect(self._on_table_context_menu)
        self.c_response_table.keyPressEvent = self._table_key_press_event
        self.c_response_fig = Figure(figsize=(8, 6), tight_layout=True)
        self.c_response_canvas = FigureCanvas(self.c_response_fig)
        self.c_response_canvas.setParent(self)
        configure_matplotlib_canvas(self.c_response_canvas, self.c_response_fig, self)

        self.result_tabs = QTabWidget()
        tab_corner = QWidget()
        tab_corner_layout = QHBoxLayout(tab_corner)
        tab_corner_layout.setContentsMargins(0, 0, 4, 0)
        tab_corner_layout.addWidget(self.tune_btn)
        self.tipper_mode_combo = QComboBox(self)
        self.tipper_mode_combo.addItem("倾子：实部/虚部", "re_im")
        self.tipper_mode_combo.addItem("倾子：幅值/相位", "amp_phase")
        self.tipper_mode_combo.currentIndexChanged.connect(self._draw_tipper_plot)
        self.tipper_mode_combo.setVisible(False)
        self.c_response_mode_combo = QComboBox(self)
        self.c_response_mode_combo.addItem("C响应：实部/虚部", "re_im")
        self.c_response_mode_combo.addItem("C响应：幅值/相位", "amp_phase")
        self.c_response_mode_combo.currentIndexChanged.connect(self._draw_c_response_plot)
        self.c_response_mode_combo.setVisible(False)
        tab_corner_layout.addWidget(self.c_response_mode_combo)
        tab_corner_layout.addWidget(self.tipper_mode_combo)
        self.tipper_reset_button = add_matplotlib_reset_button(self.tipper_canvas, tab_corner_layout)
        self.c_response_reset_button = add_matplotlib_reset_button(self.c_response_canvas, tab_corner_layout)
        self.result_tabs.setCornerWidget(tab_corner)
        self.result_tabs.currentChanged.connect(self._sync_result_actions)
        self.result_tabs.currentChanged.connect(self._sync_response_mode_visibility)
        self.result_tabs.addTab(self.table, "阻抗结果")
        self.result_tabs.addTab(self.resistivity_table, "视电阻率与相位")

        self.mpl_fig = Figure(figsize=(6, 5), tight_layout=True)
        self.mpl_canvas = FigureCanvas(self.mpl_fig)
        self.mpl_canvas.setParent(self)
        configure_matplotlib_canvas(self.mpl_canvas, self.mpl_fig, self)
        self.mpl_reset_button = add_matplotlib_reset_button(self.mpl_canvas, tab_corner_layout)
        self.result_tabs.addTab(self.mpl_canvas, "视电阻率与相位图")
        self.result_tabs.addTab(self.c_response_table, "C 响应")
        self.result_tabs.addTab(self.c_response_canvas, "C 响应图")

        self._placeholder = QLabel("此处显示计算结果信息")
        self._placeholder.setObjectName("remotePlaceholder")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("font-size: 14px;")
        self._result_stack = QStackedWidget()
        self._result_stack.setObjectName("remoteResultStack")
        self._result_stack.addWidget(self._placeholder)
        self._result_stack.addWidget(self.result_tabs)
        self._result_stack.setCurrentIndex(0)

        # ── 组装布局 ──
        # 两个波形图左右并排
        self.local_plot.setLabel('top', '本地站')
        self.remote_plot.setLabel('top', '远参考站')
        plot_row = QHBoxLayout()
        plot_row.setContentsMargins(0, 0, 0, 0)
        plot_row.addWidget(self.local_plot)
        plot_row.addWidget(self.remote_plot)
        plot_widget = QWidget()
        plot_widget.setObjectName("remotePlotRowWidget")
        plot_widget.setLayout(plot_row)

        # 下方：结果 TabWidget
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setObjectName("remoteMainSplitter")
        main_splitter.addWidget(plot_widget)
        main_splitter.addWidget(self._result_stack)
        main_splitter.setSizes([360, 600])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(main_splitter)
        self._sync_result_actions()

        # ── 控制面板（独立浮窗） ──
        self._control_dialog = RemoteReferenceControlDialog(self)
        self._control_dialog_presented = False
        self._control_present_timer = QTimer(self)
        self._control_present_timer.setSingleShot(True)
        self._control_present_timer.timeout.connect(self._present_initial_controls)

        # ── 初始填充 ──
        self._populate_stations()
        if data_id != 0:
            idx = self.local_station_combo.findData(data_id)
            if idx >= 0:
                self.local_station_combo.setCurrentIndex(idx)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._control_dialog_presented:
            # Let the host finish showing/maximizing and activating its MDI
            # window before bringing the parameter dialog to the foreground.
            self._control_present_timer.start(0)

    def _present_initial_controls(self):
        if self.isVisible() and not self._control_dialog_presented:
            self._control_dialog_presented = True
            self._show_control_dialog()

    def _show_control_dialog(self):
        """打开/重新显示调参窗口。"""
        if self._control_dialog is not None:
            if self._control_dialog.isMinimized():
                self._control_dialog.showNormal()
            else:
                self._control_dialog.show()
            self._control_dialog.raise_()
            self._control_dialog.activateWindow()

    def closeEvent(self, event):
        if self._estimation_future is not None:
            reply = QMessageBox.question(
                self, "计算正在运行", "远参考计算仍在后台运行，关闭后将丢弃计算结果。仍要关闭吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._cancel_estimation()
        if self._bands is not None and not self._saved_current:
            reply = QMessageBox.question(
                self, "结果尚未保存", "当前计算结果尚未保存到阻抗预览，仍要关闭吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        if self._control_dialog is not None:
            self._control_dialog.close()
        super().closeEvent(event)

    # ─── 双 Y 轴同步（本地站） ─────────────────────────────────

    def _update_local_dual_geometry(self):
        self._local_b_viewbox.setGeometry(self._local_plot_item.vb.sceneBoundingRect())

    def _sync_local_b_viewbox_zoom(self, old_main_yrange, new_main_yrange):
        old_range = old_main_yrange[1] - old_main_yrange[0]
        new_range = new_main_yrange[1] - new_main_yrange[0]
        if old_range <= 1e-16:
            return
        scale = new_range / old_range
        b_yrange = self._local_b_viewbox.viewRange()[1]
        b_center = (b_yrange[0] + b_yrange[1]) / 2.0
        b_half = (b_yrange[1] - b_yrange[0]) / 2.0
        self._local_b_viewbox.setYRange(
            b_center - b_half * scale, b_center + b_half * scale, padding=0,
        )

    # ─── 远参考站 X 轴双向同步 ────────────────────────────────

    def _on_local_xrange_changed(self, _vb, range_info):
        if self._syncing_xrange:
            return
        x_range = range_info[0]
        self._syncing_xrange = True
        self._remote_plot_item.vb.blockSignals(True)
        self._remote_plot_item.setXRange(x_range[0], x_range[1], padding=0)
        self._remote_plot_item.vb.blockSignals(False)
        self._syncing_xrange = False

    def _on_remote_xrange_changed(self, _vb, range_info):
        if self._syncing_xrange:
            return
        x_range = range_info[0]
        self._syncing_xrange = True
        self._local_plot_item.vb.blockSignals(True)
        self._local_plot_item.setXRange(x_range[0], x_range[1], padding=0)
        # 主 VB 信号被阻断后 setXLink 不会自动同步 B ViewBox，需手动更新
        self._local_b_viewbox.setXRange(x_range[0], x_range[1], padding=0)
        self._local_plot_item.vb.blockSignals(False)
        self._syncing_xrange = False

    # ─── 台站管理 ──────────────────────────────────────────────────

    def _on_data_ready(self, data_id: int, channel: str, pkg: dict) -> None:
        local_id = self.local_station_combo.currentData()
        remote_id = self.remote_station_combo.currentData()

        if data_id == self._data_id and not self._local_pkg:
            # 初始请求：填充台站列表 + 选中当前台站
            self._local_pkg = pkg
            self._populate_stations()
            if self._data_id != 0:
                idx = self.local_station_combo.findData(self._data_id)
                if idx >= 0:
                    self.local_station_combo.blockSignals(True)
                    self.local_station_combo.setCurrentIndex(idx)
                    self.local_station_combo.blockSignals(False)
            self._apply_local_station_data()
            remote_id = self.remote_station_combo.currentData()
            if remote_id is not None:
                self.data_requested.emit(remote_id, "")
        elif data_id == local_id:
            self._local_pkg = pkg
            self._apply_local_station_data()
        elif data_id == remote_id:
            self._remote_pkg = pkg
            self._apply_remote_station_data()

    def on_source_restored(self, data_id: int) -> None:
        if data_id not in (
            self.local_station_combo.currentData(), self.remote_station_combo.currentData(),
        ):
            return
        self._cancel_estimation()
        self.status_label.setText("输入数据已从源文件还原；已有阻抗结果基于还原前的数据，请重新计算。")

    def _populate_stations(self):
        prev_local = self.local_station_combo.currentData()
        prev_remote = self.remote_station_combo.currentData()
        target_local = prev_local if prev_local is not None else self._data_id
        stations = self._local_pkg.get('stations', [])

        valid_local: list[tuple[str, int]] = []
        valid_remote: list[tuple[str, int]] = []
        for s in stations:
            name = f"#{s['id']} {s['name']}"
            if s['has_magnetic'] and s['has_electric']:
                valid_local.append((name, s['id']))
            if s['has_magnetic']:
                valid_remote.append((name, s['id']))

        self.local_station_combo.blockSignals(True)
        self.local_station_combo.clear()
        for name, key in valid_local:
            self.local_station_combo.addItem(name, key)
        self.local_station_combo.blockSignals(False)
        if target_local is not None and self.local_station_combo.findData(target_local) >= 0:
            self.local_station_combo.blockSignals(True)
            self.local_station_combo.setCurrentIndex(self.local_station_combo.findData(target_local))
            self.local_station_combo.blockSignals(False)
        elif self.local_station_combo.count() > 0:
            self.local_station_combo.blockSignals(True)
            self.local_station_combo.setCurrentIndex(0)
            self.local_station_combo.blockSignals(False)

        local_id = self.local_station_combo.currentData()
        self.remote_station_combo.blockSignals(True)
        self.remote_station_combo.clear()
        for name, key in valid_remote:
            if key != local_id:
                self.remote_station_combo.addItem(name, key)
        self.remote_station_combo.blockSignals(False)
        if prev_remote is not None and self.remote_station_combo.findData(prev_remote) >= 0:
            self.remote_station_combo.blockSignals(True)
            self.remote_station_combo.setCurrentIndex(self.remote_station_combo.findData(prev_remote))
            self.remote_station_combo.blockSignals(False)

    def _current_local_data(self):
        return _PkgWrapper(self._local_pkg) if self._local_pkg else None

    def _current_remote_data(self):
        return _PkgWrapper(self._remote_pkg) if self._remote_pkg else None

    def _update_station_distance(self):
        local = self._current_local_data()
        remote = self._current_remote_data()
        if local is None or remote is None:
            self.distance_label.setText("")
            return
        try:
            dist = station_distance(local, remote, unit="km")
            self.distance_label.setText(f"台站距离: {dist:.2f} km")
        except ValueError as e:
            self.distance_label.setText(f"(无法计算距离: {e})")

    def _update_sample_rate_warn(self):
        local = self._current_local_data()
        remote = self._current_remote_data()
        if local is None or remote is None:
            self.sample_rate_warn.setText("")
            return
        local_sr = 1.0 / local.dt.total_seconds()
        remote_sr = 1.0 / remote.dt.total_seconds()
        if abs(local_sr - remote_sr) > 1e-6:
            self.sample_rate_warn.setText(
                f"⚠ 采样率不一致: 本地 {local_sr:.4f} Hz, 远参考 {remote_sr:.4f} Hz"
            )
        else:
            self.sample_rate_warn.setText("")

    # ─── 本地站通道填充 ─────────────────────────────────────────────

    def _populate_local_channels(self):
        data = self._current_local_data()
        channels = data.channel_names() if data is not None else []
        mapping = _detect_channels(channels)

        for combo, key in [
            (self.ch_combo_ex, "Ex"),
            (self.ch_combo_ey, "Ey"),
            (self.ch_combo_hx, "Hx"),
            (self.ch_combo_hy, "Hy"),
        ]:
            combo.blockSignals(True)
            combo.clear()
            for ch in channels:
                combo.addItem(ch, ch)
            auto_ch = mapping.get(key)
            if auto_ch is not None:
                idx_ = combo.findText(auto_ch)
                if idx_ >= 0:
                    combo.setCurrentIndex(idx_)
            combo.blockSignals(False)

        self.ch_combo_hz.blockSignals(True)
        self.ch_combo_hz.clear()
        self.ch_combo_hz.addItem("(不使用)", None)
        for ch in channels:
            self.ch_combo_hz.addItem(ch, ch)
        self.ch_combo_hz.setCurrentIndex(0)
        self.ch_combo_hz.blockSignals(False)

    # ─── 远参考站通道填充 ────────────────────────────────────────────

    def _populate_remote_channels(self):
        data = self._current_remote_data()
        channels = data.channel_names() if data is not None else []
        mapping = _detect_channels(channels)

        for combo, key in [
            (self.ch_combo_remote_hx, "Hx"),
            (self.ch_combo_remote_hy, "Hy"),
        ]:
            combo.blockSignals(True)
            combo.clear()
            for ch in channels:
                if _is_h_channel(ch):
                    combo.addItem(ch, ch)
            auto_ch = mapping.get(key)
            if auto_ch is not None:
                idx_ = combo.findText(auto_ch)
                if idx_ >= 0:
                    combo.setCurrentIndex(idx_)
            combo.blockSignals(False)

    def _selected_remote_channels(self) -> dict[str, str | None]:
        return {
            "Hx": self.ch_combo_remote_hx.currentData(),
            "Hy": self.ch_combo_remote_hy.currentData(),
        }

    # ─── 台站切换 ──────────────────────────────────────────────────

    def _on_local_station_changed(self, _idx: int):
        self._mark_parameters_changed()
        new_id = self.local_station_combo.currentData()
        if new_id is not None:
            self.data_requested.emit(new_id, "")
        # UI 更新由 _on_data_ready → _apply_local_station_data 完成

    def _apply_local_station_data(self):
        data = self._current_local_data()
        self._populate_local_channels()
        if data is not None:
            self.sample_rate_label.setText(f"{1.0 / data.dt.total_seconds():.4f}")
            n = data.npts
            self.start_spin.blockSignals(True)
            self.start_spin.setRange(0, n - 1)
            self.start_spin.setValue(0)
            self.start_spin.blockSignals(False)
            self.end_spin.blockSignals(True)
            self.end_spin.setRange(1, n)
            self.end_spin.setValue(n)
            self.end_spin.blockSignals(False)
            self.start_dt.blockSignals(True)
            self.start_dt.setDateTime(pts_to_qdt(data.start_time))
            self.start_dt.setMinimumDateTime(pts_to_qdt(data.start_time))
            self.start_dt.setMaximumDateTime(pts_to_qdt(data.end_time))
            self.start_dt.blockSignals(False)
            self.end_dt.blockSignals(True)
            self.end_dt.setDateTime(pts_to_qdt(data.end_time))
            self.end_dt.setMinimumDateTime(pts_to_qdt(data.start_time))
            self.end_dt.setMaximumDateTime(pts_to_qdt(data.end_time))
            self.end_dt.blockSignals(False)
        self._update_station_distance()
        self._update_sample_rate_warn()
        self._refresh_plots()

    def _on_remote_station_changed(self, _idx: int):
        self._mark_parameters_changed()
        new_id = self.remote_station_combo.currentData()
        if new_id is not None:
            self.data_requested.emit(new_id, "")
        # UI 更新由 _on_data_ready → _apply_remote_station_data 完成

    def _apply_remote_station_data(self):
        self._populate_remote_channels()
        self._update_station_distance()
        self._update_sample_rate_warn()
        self._refresh_plots()

    def _on_channel_changed(self):
        self._mark_parameters_changed()
        self._sync_tipper_tab_visibility()
        self._refresh_plots()

    def _hz_channel_selected(self) -> bool:
        return self.ch_combo_hz.currentData() is not None

    def _sync_tipper_tab_visibility(self) -> None:
        should_show = self._hz_channel_selected()
        table_index = self.result_tabs.indexOf(self.tipper_table)
        plot_index = self.result_tabs.indexOf(self.tipper_canvas)
        if should_show:
            if table_index < 0:
                self._tipper_tab_index = self.result_tabs.addTab(self.tipper_table, "倾子响应")
            if plot_index < 0:
                self._tipper_plot_tab_index = self.result_tabs.addTab(self.tipper_canvas, "倾子响应图")
        else:
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
        self._sync_response_mode_visibility()

    def _current_matplotlib_export_target(self):
        current = self.result_tabs.currentWidget()
        mpl_canvas = getattr(self, "mpl_canvas", None)
        tipper_canvas = getattr(self, "tipper_canvas", None)
        c_response_canvas = getattr(self, "c_response_canvas", None)
        if current is mpl_canvas:
            return self.mpl_fig, self.mpl_canvas, "resistivity_phase"
        if current is tipper_canvas:
            return self.tipper_fig, self.tipper_canvas, "tipper"
        if current is c_response_canvas:
            return self.c_response_fig, self.c_response_canvas, "c_response"
        return None

    def _sync_response_mode_visibility(self, _index: int = -1) -> None:
        current = self.result_tabs.currentWidget()
        self.c_response_mode_combo.setVisible(current is self.c_response_canvas)
        self.tipper_mode_combo.setVisible(current is self.tipper_canvas and self._hz_channel_selected())
        self.c_response_reset_button.setVisible(current is self.c_response_canvas)
        self.tipper_reset_button.setVisible(current is self.tipper_canvas)
        mpl_reset_button = getattr(self, "mpl_reset_button", None)
        if mpl_reset_button is not None:
            mpl_reset_button.setVisible(current is getattr(self, "mpl_canvas", None))

    def _sync_result_actions(self, _index: int = -1) -> None:
        pass

    def _export_current_figure(self) -> None:
        target = self._current_matplotlib_export_target()
        if target is None:
            return
        figure, canvas, plot_slug = target
        local = self._current_local_data()
        station = local.name if local is not None else "station"
        path = export_matplotlib_figure_with_dialog(
            self,
            figure,
            canvas,
            default_directory=Path.home(),
            default_name=default_export_name(station, plot_slug),
        )
        if path is not None:
            self.status_label.setText(f"已输出图像到 {path}")

    # ─── 波形图 ────────────────────────────────────────────────────

    def _selected_local_channels(self) -> dict[str, str | None]:
        return {
            "Ex": self.ch_combo_ex.currentData(),
            "Ey": self.ch_combo_ey.currentData(),
            "Hx": self.ch_combo_hx.currentData(),
            "Hy": self.ch_combo_hy.currentData(),
            "Hz": self.ch_combo_hz.currentData(),
        }

    def _refresh_plots(self):
        self._refresh_local_plot()
        self._refresh_remote_plot()

    def _refresh_local_plot(self):
        data = self._current_local_data()
        for curve in list(self._curves_local.values()):
            curve_view = curve.getViewBox()
            if curve_view is self._local_b_viewbox:
                self._local_b_viewbox.removeItem(curve)
            else:
                self._local_plot_item.vb.removeItem(curve)
        self._curves_local.clear()
        self._lod_local.clear()
        self._local_legend.clear()

        if data is None:
            return

        datetime_index = pd.DatetimeIndex(data.datetime_index)
        t = datetime_index.asi8.astype(np.float64) / 1_000_000_000.0
        self._ts_array = t
        pw = max(self.local_plot.width(), 800)

        entries = []
        for tag, ch_name in self._selected_local_channels().items():
            if ch_name is None:
                continue
            y = data.get_channel_series(ch_name).to_numpy()
            entries.append((tag, ch_name, _is_e_channel(ch_name), t, y))
        self._local_plot_generation += 1
        run_in_pool(
            "compute", _build_lod_task, "local", entries, pw, self._local_plot_generation,
            on_done=self._on_lod_build_done, on_error=self._on_lod_build_failed, owner=self,
        )

        if len(t) > 1:
            self.region_item.blockSignals(True)
            self.region_item.setRegion((t[0], t[-1]))
            self.region_item.blockSignals(False)
            self.remote_region_item.blockSignals(True)
            self.remote_region_item.setRegion((t[0], t[-1]))
            self.remote_region_item.blockSignals(False)

        self._update_range_info()
        self._update_warning()

    def _refresh_remote_plot(self):
        data = self._current_remote_data()
        for curve in list(self._curves_remote.values()):
            self._remote_plot_item.vb.removeItem(curve)
        self._curves_remote.clear()
        self._lod_remote.clear()
        self._remote_legend.clear()

        if data is None:
            return

        datetime_index = pd.DatetimeIndex(data.datetime_index)
        t = datetime_index.asi8.astype(np.float64) / 1_000_000_000.0
        self._ts_remote_array = t
        pw = max(self.remote_plot.width(), 800)

        entries = []
        for tag, ch_name in self._selected_remote_channels().items():
            if ch_name is None:
                continue
            y = data.get_channel_series(ch_name).to_numpy()
            entries.append((tag, ch_name, False, t, y))
        self._remote_plot_generation += 1
        run_in_pool(
            "compute", _build_lod_task, "remote", entries, pw, self._remote_plot_generation,
            on_done=self._on_lod_build_done, on_error=self._on_lod_build_failed, owner=self,
        )

    def _on_lod_build_done(self, completed: tuple) -> None:
        side, generation, built = completed
        if side == "local" and generation != self._local_plot_generation:
            return
        if side == "remote" and generation != self._remote_plot_generation:
            return
        for tag, ch_name, electric, pyramid, (ox, oy) in built:
            colour = CH_COLOURS.get(tag, (100, 100, 100))
            curve = pg.PlotDataItem(
                ox, oy, pen=pg.mkPen(color=colour, width=1), name=f"{tag} ({ch_name})",
                autoDownsample=False, clipToView=False, connect="finite",
            )
            if side == "local":
                self._lod_local[tag] = pyramid
                viewbox = self._local_plot_item.vb if electric else self._local_b_viewbox
                curve.setParentItem(viewbox); curve.forgetViewBox(); viewbox.addItem(curve)
                self._local_legend.addItem(curve, f"{tag} ({ch_name})")
                self._curves_local[tag] = curve
            else:
                self._lod_remote[tag] = pyramid
                curve.setParentItem(self._remote_plot_item.vb); curve.forgetViewBox(); self._remote_plot_item.vb.addItem(curve)
                self._remote_legend.addItem(curve, f"{tag} ({ch_name})")
                self._curves_remote[tag] = curve

    def _on_lod_build_failed(self, error: BaseException) -> None:
        logger.error("Remote-reference waveform LOD build failed: %r", error, exc_info=error)

    # ─── LOD 视口刷新 ──────────────────────────────────────────────

    def _on_lod_viewport_changed(self):
        """任一 ViewBox X 范围变化 → 启动防抖定时器，16ms 后批量刷新两图所有曲线。"""
        self._lod_timer.start()

    def _lod_update_curves(self):
        """从各通道的 LODPyramid 查询当前视口适配分辨率的数据并 setData。"""
        # 本地站
        if self._lod_local:
            vb = self._local_plot_item.vb
            x_start, x_end = vb.viewRange()[0]
            pw = max(self.local_plot.width(), 200)
            for tag, pyramid in self._lod_local.items():
                curve = self._curves_local.get(tag)
                if curve is None:
                    continue
                lx, ly = pyramid.get(x_start, x_end, pw)
                curve.setData(lx, ly)

        # 远参考站
        if self._lod_remote:
            vb = self._remote_plot_item.vb
            x_start, x_end = vb.viewRange()[0]
            pw = max(self.remote_plot.width(), 200)
            for tag, pyramid in self._lod_remote.items():
                curve = self._curves_remote.get(tag)
                if curve is None:
                    continue
                lx, ly = pyramid.get(x_start, x_end, pw)
                curve.setData(lx, ly)

    # ─── 区域 / 范围选择 ──────────────────────────────────────────

    def _on_local_region_changed(self):
        """本地站 LinearRegionItem 拖拽 → 同步远参考站 region + 更新微调框。"""
        if self._updating_range or self._syncing_region:
            return
        data = self._current_local_data()
        if data is None:
            return
        left, right = self.region_item.getRegion()
        start, end = self._ts_range_to_indices(left, right)

        # 同步远参考站 region
        self._syncing_region = True
        self.remote_region_item.blockSignals(True)
        self.remote_region_item.setRegion((left, right))
        self.remote_region_item.blockSignals(False)
        self._syncing_region = False

        self._updating_range = True
        self.start_spin.blockSignals(True)
        self.start_spin.setValue(start)
        self.start_spin.blockSignals(False)
        self.end_spin.blockSignals(True)
        self.end_spin.setValue(end)
        self.end_spin.blockSignals(False)
        self._updating_range = False

    def _on_remote_region_changed(self):
        """远参考站 LinearRegionItem 拖拽 → 同步本地站 region + 更新微调框。"""
        if self._updating_range or self._syncing_region:
            return
        data = self._current_local_data()
        if data is None:
            return
        left, right = self.remote_region_item.getRegion()
        start, end = self._ts_range_to_indices(left, right)

        # 同步本地站 region
        self._syncing_region = True
        self.region_item.blockSignals(True)
        self.region_item.setRegion((left, right))
        self.region_item.blockSignals(False)
        self._syncing_region = False

        self._updating_range = True
        self.start_spin.blockSignals(True)
        self.start_spin.setValue(start)
        self.start_spin.blockSignals(False)
        self.end_spin.blockSignals(True)
        self.end_spin.setValue(end)
        self.end_spin.blockSignals(False)
        self._updating_range = False

    def _on_region_change_finished(self):
        if self._updating_range:
            return
        data = self._current_local_data()
        if data is None:
            return
        left, right = self.region_item.getRegion()
        start, end = self._ts_range_to_indices(left, right)
        self._set_dt_from_indices(start, end)
        self._update_range_info()
        self._update_warning()

    def _on_spin_start_changed(self, val: int):
        if self._updating_range:
            return
        self._set_region_from_indices(val, self.end_spin.value())
        self._set_dt_from_indices(val, self.end_spin.value())
        self._update_range_info()
        self._update_warning()
        self._mark_parameters_changed()

    def _on_spin_end_changed(self, val: int):
        if self._updating_range:
            return
        self._set_region_from_indices(self.start_spin.value(), val)
        self._set_dt_from_indices(self.start_spin.value(), val)
        self._update_range_info()
        self._update_warning()
        self._mark_parameters_changed()

    def _on_dt_start_changed(self, qdt: QDateTime):
        if self._updating_range:
            return
        data = self._current_local_data()
        if data is None:
            return
        t = qdt_to_pts(qdt)
        idx = np.searchsorted(data.datetime_index, t)
        idx = max(0, min(idx, data.npts - 1))
        self._set_spins(idx, self.end_spin.value())
        self._set_region_from_indices(idx, self.end_spin.value())

    def _on_dt_end_changed(self, qdt: QDateTime):
        if self._updating_range:
            return
        data = self._current_local_data()
        if data is None:
            return
        t = qdt_to_pts(qdt)
        idx = np.searchsorted(data.datetime_index, t)
        idx = max(1, min(idx, data.npts))
        self._set_spins(self.start_spin.value(), idx)
        self._set_region_from_indices(self.start_spin.value(), idx)

    def _ts_range_to_indices(self, left: float, right: float) -> tuple[int, int]:
        data = self._current_local_data()
        if data is None or self._ts_array is None:
            return 0, 0
        t_arr = self._ts_array
        start = int(np.searchsorted(t_arr, left))
        end = int(np.searchsorted(t_arr, right))
        return max(0, start), min(data.npts, end)

    def _set_spins(self, start: int, end: int):
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
        data = self._current_local_data()
        if data is None:
            return
        idx = data.datetime_index
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
        data = self._current_local_data()
        if data is None or self._ts_array is None:
            return
        t_arr = self._ts_array
        if end > 0 and start < len(t_arr):
            ll, rr = t_arr[start], t_arr[min(end, len(t_arr) - 1)]
            self._updating_range = True
            self.region_item.blockSignals(True)
            self.region_item.setRegion((ll, rr))
            self.region_item.blockSignals(False)
            self.remote_region_item.blockSignals(True)
            self.remote_region_item.setRegion((ll, rr))
            self.remote_region_item.blockSignals(False)
            self._updating_range = False

    def _reset_range(self):
        data = self._current_local_data()
        if data is None:
            return
        n = data.npts
        self._set_spins(0, n)
        self._set_dt_from_indices(0, n)
        self._set_region_from_indices(0, n)
        self._update_range_info()
        self._update_warning()

    def _update_range_info(self):
        data = self._current_local_data()
        if data is None:
            return
        start = self.start_spin.value()
        end = self.end_spin.value()
        n = end - start
        dt = data.dt.total_seconds()
        duration = n * dt
        self.range_info.setText(f"选中 {n} 样点 ({start}:{end}), 时长 {duration:.0f}s")

    def _update_warning(self):
        if not hasattr(self, "fft_window_combo"):
            return
        window_size = self.fft_window_combo.currentData()
        min_required = 26 * window_size
        start = self.start_spin.value()
        end = self.end_spin.value()
        n = end - start
        if n < min_required:
            self.warning_label.setText(
                f"提示: 选中 {n} 样点 < 26×FFT短窗={min_required}，"
                f"低频有效时间窗可能偏少"
            )
        else:
            self.warning_label.setText("")

    def _current_backend(self) -> str:
        return str(self.backend_combo.currentData())

    def _on_backend_changed(self):
        self._set_window_values(WINDOW_VALUES)
        for field in self._rs_only_fields:
            _set_form_row_visible(self._param_form, field, False)
        _set_form_row_visible(self._param_form, self.fft_window_combo, True)
        self.warning_label.setText("")
        self.bands_spin.setToolTip("两个后端共用 EMTF 降采样层和 FC 选带；更改后需重新计算")

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

    # ─── 远参考估计 ──────────────────────────────────────────────

    def _run_estimation(self):
        if self._estimation_future is not None:
            return
        if self._bands is not None and not self._saved_current:
            reply = QMessageBox.question(
                self, "替换未保存结果", "当前结果尚未保存到阻抗预览，重新计算会替换它。继续吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        local = self._current_local_data()
        remote = self._current_remote_data()
        if local is None:
            QMessageBox.warning(self, "错误", "请选择一个本地站。")
            return
        if remote is None:
            QMessageBox.warning(self, "错误", "请选择一个远参考站。")
            return
        backend = self._current_backend()

        local_sr = 1.0 / local.dt.total_seconds()
        remote_sr = 1.0 / remote.dt.total_seconds()
        if abs(local_sr - remote_sr) > 1e-6:
            QMessageBox.warning(
                self, "采样率不一致",
                f"本地站 ({local_sr:.4f} Hz) 和远参考站 ({remote_sr:.4f} Hz) 采样率不一致，无法计算。",
            )
            return

        ch_ex = self.ch_combo_ex.currentData()
        ch_ey = self.ch_combo_ey.currentData()
        ch_hx_local = self.ch_combo_hx.currentData()
        ch_hy_local = self.ch_combo_hy.currentData()
        ch_hz = self.ch_combo_hz.currentData()

        for label, ch in [("Ex", ch_ex), ("Ey", ch_ey), ("Hx", ch_hx_local), ("Hy", ch_hy_local)]:
            if ch is None:
                QMessageBox.warning(self, "错误", f"请选择本地站 {label} 通道。")
                return

        ch_hx_remote = self.ch_combo_remote_hx.currentData()
        ch_hy_remote = self.ch_combo_remote_hy.currentData()
        if ch_hx_remote is None or ch_hy_remote is None:
            QMessageBox.warning(self, "错误", "请选择远参考站 Hx 和 Hy 通道。")
            return
        local_channels = [ch_ex, ch_ey, ch_hx_local, ch_hy_local]
        if len(set(local_channels)) != len(local_channels):
            QMessageBox.warning(self, "错误", "本地站 Ex、Ey、Hx、Hy 必须映射到四个不同通道。")
            return
        if ch_hz is not None and ch_hz in local_channels:
            QMessageBox.warning(self, "错误", "Hz 必须与本地站的 Ex、Ey、Hx、Hy 通道不同。")
            return
        if ch_hx_remote == ch_hy_remote:
            QMessageBox.warning(self, "错误", "远参考 Hx 和 Hy 必须映射到不同通道。")
            return
        if self.local_station_combo.currentData() == self.remote_station_combo.currentData():
            QMessageBox.warning(self, "错误", "远参考站必须与本地站不同。")
            return

        try:
            start = self.start_spin.value()
            end = self.end_spin.value()
            if end <= start:
                QMessageBox.warning(self, "错误", "结束索引必须大于起始索引。")
                return

            # 本地站数据
            ex_arr = local.get_channel_series(ch_ex).to_numpy()[start:end]
            ey_arr = local.get_channel_series(ch_ey).to_numpy()[start:end]
            local_hx_arr = local.get_channel_series(ch_hx_local).to_numpy()[start:end]
            local_hy_arr = local.get_channel_series(ch_hy_local).to_numpy()[start:end]
            hz_arr = None
            if ch_hz is not None:
                hz_arr = local.get_channel_series(ch_hz).to_numpy()[start:end]

            # 当前后端按数组位置配对 FFT 窗口，先确认两站对应同一时间网格。
            remote_start, remote_end = matching_remote_slice(
                local.datetime_index,
                remote.datetime_index,
                start,
                end,
            )

            remote_hx_arr = remote.get_channel_series(ch_hx_remote).to_numpy()[remote_start:remote_end]
            remote_hy_arr = remote.get_channel_series(ch_hy_remote).to_numpy()[remote_start:remote_end]

            sample_rate = local_sr
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
            self.cancel_button.setEnabled(True)

            kwargs = build_spectrum_kwargs(
                sample_rate=sample_rate,
                window_size=window_size,
                step_pct=self.step_pct_spin.value(),
                huber_threshold=self.huber_spin.value(),
                max_iterations=self.max_iter_spin.value(),
                tolerance=self.tol_spin.value(),
                hz_arr=hz_arr,
                backend=backend,
                fft_window_size=self.fft_window_combo.currentData(),
                n_bands=self.bands_spin.value(),
                station_name=local.name,
            )
            start_time = local.datetime_index[start] if start < len(local.datetime_index) else ""
            end_time = local.datetime_index[end - 1] if end > 0 and end <= len(local.datetime_index) else ""
            station_meta = local.station_metadata
            context = {
                "method": "remote reference",
                "station_name": local.name,
                "remote_station_name": remote.name,
                "latitude": getattr(station_meta, "latitude", 0.0) or 0.0,
                "longitude": getattr(station_meta, "longitude", 0.0) or 0.0,
                "elevation": getattr(station_meta, "elevation", 0.0) or 0.0,
                "range_start": start_time,
                "range_end": end_time,
                "channels": {
                    "Ex": ch_ex, "Ey": ch_ey, "local Hx": ch_hx_local,
                    "local Hy": ch_hy_local, "Hz": ch_hz,
                    "remote Hx": ch_hx_remote, "remote Hy": ch_hy_remote,
                },
                "parameters": {
                    "start": start, "end": end, "backend": backend, "window_size": window_size,
                    "fft_window_size": self.fft_window_combo.currentData(),
                    "processing_scheme": "emtf_fc_bands_v1",
                    "ar_filter_length": 3,
                    "solver_settings": {"huber_threshold": 1.5, "max_iterations": 50, "tolerance": 0.002},
                    "local_station_id": self.local_station_combo.currentData(),
                    "remote_station_id": self.remote_station_combo.currentData(),
                    "step_pct": self.step_pct_spin.value(), "bands": self.bands_spin.value(),
                    "huber_threshold": self.huber_spin.value(),
                    "max_iterations": self.max_iter_spin.value(), "tolerance": self.tol_spin.value(),
                },
            }
            self._pending_context = context
            self._task_generation += 1
            generation = self._task_generation
            arrays = (ex_arr, ey_arr, local_hx_arr, local_hy_arr, remote_hx_arr, remote_hy_arr, hz_arr)
            channel_names = (ch_ex, ch_ey, ch_hx_local, ch_hy_local, ch_hx_remote, ch_hy_remote, ch_hz)
            self._estimation_future = run_in_pool(
                "compute", _estimate_remote_task, backend, arrays, channel_names, kwargs, generation,
                on_done=self._on_estimation_done,
                on_error=self._on_estimation_error,
                owner=self,
            )

        except Exception as e:
            logger.exception("Remote reference estimation failed")
            QMessageBox.critical(self, "估计失败", str(e))
            self.status_label.setText("失败")
            self.run_button.setEnabled(True)
            self.cancel_button.setEnabled(False)

    def _on_estimation_done(self, completed: tuple) -> None:
        generation, result, error = completed
        if generation != self._task_generation:
            return
        if error is not None:
            self._on_estimation_error(error)
            return
        self._estimation_future = None
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._spectrum = result
        self._result = result
        self._calculation_context = self._pending_context
        self._build_bands(
            backend=self._calculation_context["parameters"]["backend"],
            window_size=self._calculation_context["parameters"]["window_size"],
            n_bands=self._calculation_context["parameters"]["bands"],
        )
        self._display_result(result)
        self._saved_current = False
        self.save_preview_btn.setEnabled(self._bands is not None)
        self.save_preview_btn.setText("保存到阻抗预览")
        self.view_preview_btn.setVisible(False)
        self.status_label.setText(f"完成 — {len(result.frequencies_hz)} 个频率, {result.segment_count} 个时间窗")

    def _on_estimation_error(self, error: BaseException) -> None:
        if self._estimation_future is None:
            return
        self._estimation_future = None
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        logger.error("Remote reference estimation failed", exc_info=error)
        QMessageBox.critical(self, "估计失败", str(error))
        self.status_label.setText("失败")

    def _cancel_estimation(self) -> None:
        if self._estimation_future is None:
            return
        self._task_generation += 1
        self._estimation_future.cancel()
        self._estimation_future = None
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.status_label.setText("已取消；后台原生调用若已开始会在完成后丢弃结果")

    def _build_bands(self, backend=None, window_size=None, n_bands=None):
        """从 spectrum 构建 BandedResult，period 范围 2*dt ~ w*dt 对数均分。"""
        if self._spectrum is None:
            self._bands = None
            return
        backend = backend or self._current_backend()
        if backend == BACKEND_EMTF_PYWRAP or getattr(self._spectrum, "processing_scheme", "") == "emtf_fc_bands_v1":
            from emtf.io import from_result

            self._bands = from_result(self._spectrum, station_name=self._spectrum.station_name, mode="rr")
            return
        data = self._current_local_data()
        if data is None:
            self._bands = None
            return
        dt = data.dt.total_seconds()
        w = window_size or self.window_combo.currentData()
        n_bands = n_bands or self.bands_spin.value()
        periods = np.logspace(np.log10(2 * dt), np.log10(w * dt), n_bands)
        band_configs = build_band_configs(periods)
        self._bands = self._spectrum.to_bands(band_configs)

    def _emtf_period_grid(self, sample_rate: float, window_size: int, fft_window_size: int):
        return emtf_period_grid(sample_rate, window_size, fft_window_size, self.bands_spin.value())

    def _on_bands_changed(self):
        """频带数量变更时重建 bands 并刷新对比图。"""
        if self._spectrum is None:
            return
        self._mark_parameters_changed()

    def _mark_parameters_changed(self, *_args) -> None:
        if not self._calculation_context or self._spectrum is None:
            return
        if self._current_calculation_signature() != self._saved_calculation_signature():
            self.save_preview_btn.setEnabled(False)
            self.status_label.setText("参数、台站或通道已更改，请重新运行后再保存")

    def _current_calculation_signature(self) -> tuple:
        return (
            self.start_spin.value(), self.end_spin.value(), self._current_backend(),
            self.window_combo.currentData(), self.fft_window_combo.currentData(),
            self.step_pct_spin.value(), self.huber_spin.value(), self.max_iter_spin.value(),
            self.tol_spin.value(), self.bands_spin.value(), self.local_station_combo.currentData(),
            self.remote_station_combo.currentData(), self.ch_combo_ex.currentData(),
            self.ch_combo_ey.currentData(), self.ch_combo_hx.currentData(), self.ch_combo_hy.currentData(),
            self.ch_combo_hz.currentData(), self.ch_combo_remote_hx.currentData(),
            self.ch_combo_remote_hy.currentData(),
        )

    def _saved_calculation_signature(self) -> tuple:
        params = self._calculation_context.get("parameters", {})
        channels = self._calculation_context.get("channels", {})
        return (
            params.get("start"), params.get("end"), params.get("backend"), params.get("window_size"),
            params.get("fft_window_size"), params.get("step_pct"), params.get("huber_threshold"),
            params.get("max_iterations"), params.get("tolerance"), params.get("bands"),
            params.get("local_station_id"), params.get("remote_station_id"), channels.get("Ex"),
            channels.get("Ey"), channels.get("local Hx"), channels.get("local Hy"), channels.get("Hz"),
            channels.get("remote Hx"), channels.get("remote Hy"),
        )

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
        result = self._bands if self._hz_channel_selected() else None
        mode = self.tipper_mode_combo.currentData() if hasattr(self, "tipper_mode_combo") else "re_im"
        draw_tipper_response(self.tipper_fig, self.tipper_canvas, result, str(mode), self)

    def _display_c_response_result(self) -> None:
        bands = self._bands
        if bands is None:
            self.c_response_table.setRowCount(0)
            self._draw_c_response_plot()
            return
        components = c_response_from_bands(bands)
        periods = np.asarray(getattr(bands, "periods", []), dtype=float)
        freqs = np.asarray(getattr(bands, "center_frequencies_hz", []), dtype=float)
        if not components:
            self.c_response_table.setRowCount(0)
            self._draw_c_response_plot()
            return
        count = min([len(periods), len(freqs), *(len(values) for values in components.values())])
        valid = np.isfinite(periods[:count]) & (periods[:count] > 0) & np.isfinite(freqs[:count])
        self.c_response_table.setRowCount(int(valid.sum()))
        for row, index in enumerate(np.flatnonzero(valid)):
            self.c_response_table.setItem(row, 0, QTableWidgetItem(f"{freqs[index]:.6f}"))
            self.c_response_table.setItem(row, 1, QTableWidgetItem(f"{periods[index]:.3f}"))
            for column, name in enumerate(("cxx", "cxy", "cyx", "cyy"), start=2):
                value = components[name][index]
                self.c_response_table.setItem(
                    row, column, QTableWidgetItem(f"{value.real:.4e}{value.imag:+.4e}j")
                )
        self._draw_c_response_plot()

    def _draw_c_response_plot(self):
        mode = self.c_response_mode_combo.currentData() if hasattr(self, "c_response_mode_combo") else "re_im"
        draw_c_response(self.c_response_fig, self.c_response_canvas, self._bands, str(mode), self)

    def _display_result(self, result):
        """将 SpectrumResult 填充到表格中，并用 to_bands 后的 BandedResult 画图。"""
        self._result_stack.setCurrentIndex(1)
        freqs = np.array(result.frequencies_hz)
        n = len(freqs)

        def _c2s(c: complex) -> str:
            return f"{c.real:.4e}{c.imag:+.4e}j"

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

        self._display_tipper_result()
        self._display_c_response_result()

        self._draw_comparison_plot()

    def _draw_comparison_plot(self):
        draw_impedance_response(self.mpl_fig, self.mpl_canvas, self._bands, self)

    def result(self):
        return self._result

    # ─── 表格剪贴板 / 导出 ──────────────────────────────────────────

    def _table_key_press_event(self, event):
        if event.matches(Qt.KeyboardShortcut.StandardKey.Copy):
            self._copy_selection()
        else:
            QTableWidget.keyPressEvent(self.table, event)

    def _on_table_context_menu(self, pos):
        """右键菜单：复制。"""
        table = self.sender()
        if not isinstance(table, QTableWidget):
            table = self.table
        menu = QMenu()
        copy_action = menu.addAction("复制")
        action = menu.exec(table.viewport().mapToGlobal(pos))
        if action is copy_action:
            self._copy_selection()

    def _save_to_preview(self) -> None:
        if self._bands is None or not self._calculation_context:
            QMessageBox.warning(self, "无结果", "请先运行估计。")
            return
        if self._saved_current:
            return
        if self._current_calculation_signature() != self._saved_calculation_signature():
            QMessageBox.warning(self, "参数已更改", "参数、台站或通道已更改，请重新运行后再保存。")
            return
        payload = dict(self._calculation_context)
        payload["bands"] = self._bands
        self.snapshot_save_requested.emit(payload)
        self._saved_current = True
        self.save_preview_btn.setEnabled(False)
        self.save_preview_btn.setText("已保存到阻抗预览")
        self.view_preview_btn.setVisible(True)
        self.status_label.setText(self.status_label.text() + "；已暂存")

    def _copy_selection(self):
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

        local = self._current_local_data()
        if local is None:
            QMessageBox.warning(self, "无数据", "当前没有本地台站数据。")
            return

        station_meta = local.station_metadata
        station_name = station_meta.name.strip() or "unknown"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "导出 EDI 阻抗文件",
            f"{station_name}.edi",
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
                    processing_mode="rr",
                    sample_rate_hz=1.0 / local.dt.total_seconds(),
                )
                edi_text = self._bands.format_edi_str(meta)
            else:
                edi_text = self._bands.format_edi_str(
                    station_name=station_name,
                    mode="rr",
                    lat=lat,
                    lon=lon,
                    elev=elev,
                )
            path.write_text(edi_text, encoding="ascii")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", f"无法导出 EDI 文件：{exc}")
            return

        self.status_label.setText(f"已导出 EDI 到 {path}")
