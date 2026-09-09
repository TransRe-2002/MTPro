from __future__ import annotations

import logging
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter,
    QSizePolicy, QSlider, QDateTimeEdit, QScrollBar,
    QLabel, QPushButton, QFrame, QHBoxLayout, QGridLayout, QGroupBox,
    QCheckBox, QApplication,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from ui.brief_plot import BriefPlot
from utils.time_convert import pts_to_qdt, qdt_to_pts
from utils.series import dti_to_numpy, safe_nanmean
from utils.decimate import LODPyramid, PYRAMID_MAX_FINEST_POINTS
from utils.qt_bridge import run as run_in_pool
from utils.signals import block_signals

logger = logging.getLogger(__name__)
from core.em_data import EMData
from base.time_viewport_mixin import TimeViewportMixin

_DISPLAY_CHANNEL_PREFIXES = ("Ex", "Ey", "Hx", "Hy", "Hz", "Bx", "By", "Bz")

@dataclass
class PlotResources:
    widget: BriefPlot | None = None
    row_container: QWidget | None = None


class PlotRowContainer(QFrame):
    """通道行容器；通道顺序由数据模型统一管理。"""

    def __init__(self, channel: str, brief_plot: BriefPlot, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)

        # 通道名称显示在 BriefPlot 的 X 轴标签中，不再额外放置拖拽把手。
        row_layout = QHBoxLayout(self)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)
        row_layout.addWidget(brief_plot, 1)


class DataViewWidget(TimeViewportMixin, QWidget):

    # 异步显示构建完成后发出（主线程），供外部关闭进度框
    display_ready = Signal()

    # -------------------------------------------------------------------------
    # 初始化
    # -------------------------------------------------------------------------

    def __init__(self, parent=None):
        super().__init__(parent)

        self._viewport_debounce_init()  # Mixin: 初始化视口更新防抖

        self.em_data: EMData | None = None
        self.x_view_start: pd.Timedelta = pd.Timedelta(seconds=0)
        self.x_view_size: pd.Timedelta = pd.Timedelta(hours=1)
        self.duration: pd.Timedelta = pd.Timedelta(hours=0)
        self._build_gen: int = 0           # 显示构建代次（防中途切换/重初始化串台）
        self._pending_gen: int = -1
        self._pending_data: EMData | None = None
        self._channel_build_gen: dict[str, int] = {}

        # 通道索引保留在 plot_rows 中，但所有通道实际共用一个 PlotWidget。
        self.plot_rows: dict[str, PlotResources] = {}
        # 所有通道共用一个绘图控件；plot_rows 保留为处理工具的兼容索引。
        self.plot_widget = BriefPlot(label=None)
        # 每通道的 LOD 金字塔：显示层只画视口适配分辨率的数据，
        # 全分辨率数据仍留在 EMData 中供处理工具使用
        self._pyramids: dict[str, LODPyramid] = {}
        self._channel_centers: dict[str, float] = {}
        self._channel_scales: dict[str, float] = {}
        self._stack_offsets: dict[str, float] = {}
        self._display_gain = 1.0
        # 有序通道列表（控制显示顺序）
        self._channel_order: list[str] = []

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setOpaqueResize(False)

        self.slide_time = QSlider(Qt.Orientation.Horizontal)
        self.slide_time.setFixedWidth(300)
        self.slide_time.setDisabled(True)
        self.start_time = QDateTimeEdit()
        self.start_time.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.start_time.setCalendarPopup(True)
        self.start_time.setTimeSpec(Qt.TimeSpec.UTC)
        self.end_time = QDateTimeEdit()
        self.end_time.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.end_time.setCalendarPopup(True)
        self.end_time.setTimeSpec(Qt.TimeSpec.UTC)
        self.btn_plot = QPushButton("时间序列跳转")
        self.btn_plot.setDisabled(True)
        self.scroll_x = QScrollBar(Qt.Orientation.Horizontal)
        self.scroll_x.setDisabled(True)

        # ── 通道可见性 CheckBox 组 ──
        self.checkbox_group = QGroupBox("通道显示 / Channel Visibility")
        self.checkbox_group.setFont(QFont("Arial", 9))
        # 使用网格布局，每行最多 5 个 CheckBox，自动换行
        self.checkbox_layout = QGridLayout(self.checkbox_group)
        self.checkbox_layout.setContentsMargins(8, 4, 8, 4)
        self.checkbox_layout.setSpacing(6)
        self._checkbox_columns = 9  # 一行最多9个，紧凑排列
        self._checkboxes: dict[str, QCheckBox] = {}

        self.init_ui()
        self.connect_signals()
        self.plot_widget.vertical_gain_changed.connect(self._on_vertical_gain_changed)
        self._shutdown_done = False

    def init_ui(self):
        plot_panel = QWidget()
        plot_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setSpacing(2)
        plot_layout.setContentsMargins(4, 4, 4, 4)

        # ── 时间导航控制组 ─────────────────────────────────────
        control_group = QGroupBox("时间窗口导航 / Time Window Navigation", plot_panel)
        control_group.setFont(QFont("Arial", 9))
        control_layout = QVBoxLayout(control_group)
        control_layout.setContentsMargins(8, 6, 8, 6)
        control_layout.setSpacing(4)

        # 第一行：窗口长度滑块
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        lbl_win = QLabel("窗口长度:", control_group)
        lbl_win.setFont(QFont("Arial", 9))
        lbl_win.setFixedWidth(64)
        row1.addWidget(lbl_win)
        row1.addWidget(self.slide_time)
        row1.addStretch(1)
        control_layout.addLayout(row1)

        # 第二行：起始/结束时间 + 跳转按钮
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        lbl_start = QLabel("起始:", control_group)
        lbl_start.setFont(QFont("Arial", 9))
        lbl_start.setFixedWidth(36)
        lbl_end = QLabel("结束:", control_group)
        lbl_end.setFont(QFont("Arial", 9))
        lbl_end.setFixedWidth(36)
        row2.addWidget(lbl_start)
        row2.addWidget(self.start_time)
        row2.addSpacing(8)
        row2.addWidget(lbl_end)
        row2.addWidget(self.end_time)
        row2.addSpacing(12)
        row2.addWidget(self.btn_plot)
        row2.addStretch(1)
        control_layout.addLayout(row2)

        # ── 波形图区 ──────────────────────────────────────────
        plot_frame = QFrame(plot_panel)
        plot_frame.setFrameShape(QFrame.Shape.StyledPanel)
        plot_frame.setFrameShadow(QFrame.Shadow.Sunken)
        plot_frame_layout = QVBoxLayout(plot_frame)
        plot_frame_layout.setContentsMargins(2, 2, 2, 2)
        plot_frame_layout.setSpacing(2)
        self.splitter.addWidget(self.plot_widget)
        plot_frame_layout.addWidget(self.splitter, 1)
        plot_frame_layout.addWidget(self.scroll_x)

        # 导航栏和通道显示并排
        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        top_row.addWidget(control_group)
        top_row.addWidget(self.checkbox_group, 1)
        plot_layout.addLayout(top_row)
        plot_layout.addWidget(plot_frame, 1)

        self.setLayout(plot_layout)

    def connect_signals(self):
        self.slide_time.valueChanged.connect(self.on_x_view_size_changed)   # Mixin
        self.slide_time.sliderPressed.connect(self._on_scroll_drag_start)
        self.slide_time.sliderReleased.connect(self._on_scroll_drag_end)
        self.scroll_x.valueChanged.connect(self.on_x_view_start_changed)   # Mixin
        self.scroll_x.sliderPressed.connect(self._on_scroll_drag_start)
        self.scroll_x.sliderReleased.connect(self._on_scroll_drag_end)
        self.btn_plot.clicked.connect(self.on_btn_plot_clicked)
        self.start_time.dateTimeChanged.connect(self.on_time_changed)       # Mixin
        self.end_time.dateTimeChanged.connect(self.on_time_changed)         # Mixin

    # -------------------------------------------------------------------------
    # 数据加载
    # -------------------------------------------------------------------------

    def init_data(
        self, data: EMData,
        progress_callback: Callable[[str], None] | None = None,
        view_state: dict | None = None,
        build_plots: bool = True,
    ):
        """初始化/更换数据。

        build_plots=True 时同步构建金字塔与曲线（测试/兼容路径）；
        False 时只建控件与状态，金字塔走 start_display_build() 异步
        构建（主窗口激活路径，避免大文件冻住 UI）。
        """
        self._build_gen += 1
        self._pending_gen = -1
        self._pending_data = None
        self._channel_build_gen.clear()
        self.em_data = data
        if self.em_data is None:
            self._pyramids.clear()
            self._channel_centers.clear()
            self._channel_scales.clear()
            for resources in self.plot_rows.values():
                if resources.widget is not None:
                    resources.widget.clear_curves()
                    resources.widget.hide()
            return

        self.duration = self.em_data.end_time - self.em_data.start_time

        self.set_time_range(                                                 # Mixin
            pts_to_qdt(self.em_data.start_time),
            pts_to_qdt(self.em_data.end_time),
        )

        # 恢复上次的视口现场（在 slider/scroll 设置之前，使其反映恢复值）
        if view_state is not None:
            self._restore_viewport_state(view_state)

        # 同步 slider/scroll 状态时屏蔽信号：setValue 被滑块 min/max
        # 夹取后会触发 valueChanged → mixin 槽函数，会把刚恢复的
        # 视口现场又改回夹取值
        with block_signals(self.slide_time):
            self.slide_time.setMinimum(int(pd.Timedelta(hours=1).total_seconds()))
            self.slide_time.setMaximum(int(self.duration.total_seconds()))
            self.slide_time.setValue(int(self.x_view_size.total_seconds()))
        self.slide_time.setEnabled(True)

        with block_signals(self.scroll_x):
            self.scroll_x.setMinimum(0)
            self.scroll_x.setMaximum(int((self.duration - self.x_view_size).total_seconds()))
            self.scroll_x.setValue(int(self.x_view_start.total_seconds()))
            self.scroll_x.setPageStep(int(self.x_view_size.total_seconds()))
        self.scroll_x.setEnabled(True)

        self._build_channel_order()
        if view_state is not None:
            self._restore_channel_order(view_state)
        self._sync_plot_rows()
        if build_plots:
            self._refresh_plots(progress_callback)
        self._rebuild_checkboxes()
        if view_state is not None:
            self._restore_hidden_channels(view_state)
            self._restore_splitter_sizes(view_state)
        self._on_viewport_changed()

    def _visible_wave_channels(self) -> list[str]:
        return [
            ch for ch in self._channel_order
            if ch != "Kp" and self._checkboxes.get(ch, None) is not None
            and self._checkboxes[ch].isChecked()
        ]

    def _update_stacked_layout(self) -> None:
        """Repack visible E/H curves evenly on independent display baselines."""
        channels = self._visible_wave_channels()
        if not channels:
            self.plot_widget.setYRange(-1, 1, padding=0)
            return
        gap = 2.0
        offsets = {ch: (len(channels) - i - 1) * gap for i, ch in enumerate(channels)}
        self._stack_offsets = offsets
        # 曲线已经按各自尺度归一化到约 [-1, 1]，因此堆叠槽位使用
        # 统一的显示高度，不能再使用原始通道幅值作为 Y 轴范围。
        # 堆叠槽位的显示范围保持固定；滚轮增益只改变曲线相对基线的
        # 振幅，不能让整个 Y 轴重新缩放，否则基线会随滚轮移动。
        self.plot_widget.set_stacked_layout(channels, offsets, 1.0)

    def _on_vertical_gain_changed(self, factor: float) -> None:
        self._display_gain = float(np.clip(self._display_gain * factor, 0.2, 8.0))
        self._on_viewport_changed()

    # -------------------------------------------------------------------------
    # 交互现场状态的快照与恢复（active-only 缓存策略的配套）
    # -------------------------------------------------------------------------

    def capture_view_state(self) -> dict:
        """快照当前交互现场：视口、通道顺序、隐藏通道、各行高度。

        viewer 被销毁前调用；下次激活同一数据集时经 init_data
        的 view_state 参数恢复，用户无感。
        """
        channel_order = list(self._channel_order)
        hidden = [ch for ch, cb in self._checkboxes.items() if not cb.isChecked()]
        return {
            "x_view_start": self.x_view_start,
            "x_view_size": self.x_view_size,
            "channel_order": channel_order,
            "hidden_channels": hidden,
        }

    def _restore_viewport_state(self, view_state: dict) -> None:
        # 夹取只用数据边界（滑块 1h 下限是 UI 约束，现场恢复不受其限）
        size = view_state.get("x_view_size")
        if isinstance(size, pd.Timedelta) and pd.Timedelta(0) < size <= self.duration:
            self.x_view_size = size
        start = view_state.get("x_view_start")
        max_start = self.duration - self.x_view_size
        if isinstance(start, pd.Timedelta) and pd.Timedelta(0) <= start <= max_start:
            self.x_view_start = start

    def _restore_channel_order(self, view_state: dict) -> None:
        saved = view_state.get("channel_order") or []
        current = set(self._channel_order)
        ordered = [ch for ch in saved if ch in current]
        seen = set(ordered)
        ordered += [ch for ch in self._channel_order if ch not in seen]
        self._channel_order = ordered

    def _restore_hidden_channels(self, view_state: dict) -> None:
        # 勾选状态变化会触发 _on_channel_toggled → 隐藏对应行
        for ch in view_state.get("hidden_channels") or []:
            cb = self._checkboxes.get(ch)
            if cb is not None and cb.isChecked():
                cb.setChecked(False)

    def _restore_splitter_sizes(self, view_state: dict) -> None:
        # 旧版每通道 splitter 尺寸状态不再适用于堆叠图。
        return

    def _build_channel_order(self) -> None:
        """构建有序通道列表：E 通道 → B/H 通道 → Kp（最后）。"""
        if self.em_data is None:
            self._channel_order = []
            return
        e_channels: list[str] = []
        b_channels: list[str] = []
        other_channels: list[str] = []
        for ch in self.em_data.channel_names():
            if ch.startswith(('Ex', 'Ey')):
                e_channels.append(ch)
            elif ch.startswith(('Hx', 'Hy', 'Hz', 'Bx', 'By', 'Bz')):
                b_channels.append(ch)
            else:
                # 未知通道仍保留在复选框中，但默认不勾选。
                other_channels.append(ch)
        e_channels.sort()
        b_channels.sort()
        other_channels.sort()
        self._channel_order = e_channels + b_channels + other_channels
        # Kp 数据始终排在最后
        if self.em_data.kp_data is not None:
            self._channel_order.append('Kp')

    def _widget_index_in_splitter(self, ch: str) -> int:
        """返回某通道的 row_container 在 splitter 中的索引。"""
        resources = self.plot_rows.get(ch)
        if resources is None or resources.row_container is None:
            return -1
        return self.splitter.indexOf(resources.row_container)

    def _splitter_channel_order(self) -> list[str]:
        """从 splitter 中读取当前 widget 排列顺序。"""
        order: list[str] = []
        for i in range(self.splitter.count()):
            widget = self.splitter.widget(i)
            for ch, resources in self.plot_rows.items():
                if resources.row_container is widget:
                    order.append(ch)
                    break
        return order

    # -------------------------------------------------------------------------
    # CheckBox 组
    # -------------------------------------------------------------------------

    def _rebuild_checkboxes(self) -> None:
        """重建通道可见性 CheckBox 组。

        使用 lambda 闭包捕获通道名 c=ch，避免循环变量延迟绑定导致所有
        checkbox 都关联到最后一个通道名。
        """
        # 清除旧的
        for cb in self._checkboxes.values():
            cb.deleteLater()
        self._checkboxes.clear()
        # 清除 layout 中所有 widget
        while self.checkbox_layout.count():
            item = self.checkbox_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        if not self._channel_order:
            self.checkbox_group.setVisible(False)
            return

        cols = self._checkbox_columns
        # 标准 E/H/B 通道和 Kp 默认显示；Err、Status 等未知辅助列保留
        # 复选框，但默认不勾选。
        default_checked = {
            ch for ch in self._channel_order
            if ch == "Kp" or ch.startswith(_DISPLAY_CHANNEL_PREFIXES)
        }

        for i, ch in enumerate(self._channel_order):
            cb = QCheckBox(ch)
            cb.setChecked(ch in default_checked)
            # lambda 默认参数 c=ch 捕获当前循环值，避免闭包延迟绑定
            cb.stateChanged.connect(lambda state, c=ch: self._on_channel_toggled(c, state))
            self._checkboxes[ch] = cb
            # 网格布局：行列自动换行
            self.checkbox_layout.addWidget(cb, i // cols, i % cols)
            # 同步初始可见性：未勾选的通道对应 BriefPlot 需要隐藏
            if not cb.isChecked():
                resources = self.plot_rows.get(ch)
                if resources is not None and resources.row_container is not None:
                    resources.row_container.setVisible(False)
        self.checkbox_group.setVisible(True)

    def _on_channel_toggled(self, ch: str, state: int) -> None:
        """CheckBox 切换 → 更新堆叠布局或 Kp 右轴图层。"""
        resources = self.plot_rows.get(ch)
        if resources is None or resources.widget is None:
            return
        if ch == "Kp":
            self.plot_widget.set_kp_visible(state == Qt.CheckState.Checked.value)
        elif self.em_data is not None:
            # 通道集合变化后立即按新的可见集合重算偏移、曲线数据和 Y 范围，
            # 避免新通道沿用旧槽位而发生显示错位。
            self._on_viewport_changed()
        self.plot_widget.update()

    def _sync_left_axes(self) -> None:
        """Restore the shared waveform plot axis layout."""
        self.plot_widget.sync_axis_layout()

    # -------------------------------------------------------------------------
    # 通道 / Plot 管理
    # -------------------------------------------------------------------------

    def _required_channels(self) -> set[str]:
        """返回所有需要显示的通道名。"""
        if self.em_data is None:
            return set()
        # 所有原始通道都登记到共享图中；是否默认显示由复选框决定。
        required: set[str] = set(self.em_data.channel_names())
        if self.em_data.kp_data is not None:
            required.add('Kp')
        return required

    def _ensure_plot_row(self, ch: str) -> None:
        """登记通道；所有通道共用一个 BriefPlot。"""
        if ch not in self.plot_rows:
            self.plot_rows[ch] = PlotResources(widget=self.plot_widget)

    def _teardown_plot_row(self, ch: str) -> None:
        """销毁通道的容器和绘图控件。"""
        resources = self.plot_rows.pop(ch, None)
        self._pyramids.pop(ch, None)
        self._channel_centers.pop(ch, None)
        self._channel_scales.pop(ch, None)
        if resources is None:
            return

        # 共享 plot_widget 由 DataViewWidget 统一销毁。

    def _sync_plot_rows(self) -> None:
        """增量同步：为新增通道创建控件，移除已删除通道，并重排序。"""
        required = self._required_channels()
        current = set(self.plot_rows.keys())

        # 集合差集：需要创建的通道
        for ch in required - current:
            self._ensure_plot_row(ch)
        # 集合差集：需要销毁的通道
        for ch in current - required:
            self._teardown_plot_row(ch)

        self._update_stacked_layout()

    def _reorder_splitter(self) -> None:
        """确保 splitter 中 widget 的排列顺序与 _channel_order 一致。

        实现方式：先全部取出（保存各容器原尺寸），按 channel_order 顺序重新插入，恢复尺寸。
        """
        if len(self._channel_order) <= 1:
            return
        if self._splitter_channel_order() == self._channel_order:
            return
        # 先全部取出，再按顺序插入
        containers: list[tuple[str, QWidget]] = []
        for ch in self._channel_order:
            res = self.plot_rows.get(ch)
            if res is not None and res.row_container is not None:
                containers.append((ch, res.row_container))

        if len(containers) <= 1:
            return

        # 记录当前 sizes，便于恢复各通道之前的高度比例
        old_sizes = self.splitter.sizes()
        ch_to_size: dict[str, int] = {}
        for ch, container in containers:
            idx = self.splitter.indexOf(container)
            if 0 <= idx < len(old_sizes):
                ch_to_size[ch] = old_sizes[idx]

        # 全部取出（不显示）
        for _, container in containers:
            container.hide()
            container.setParent(None)

        # 按顺序重新插入
        for ch, container in containers:
            self.splitter.addWidget(container)
            container.show()

        # 恢复 sizes，保持用户手动调整的各通道高度
        # 若旧尺寸全为零（初始化阶段 splitter 尚未布局），则跳过恢复，
        # 交由 Qt 自动均分，避免所有行被设为零高度而不可见。
        if sum(old_sizes) > 0:
            sizes = self.splitter.sizes()
            for i, (ch, _) in enumerate(containers):
                if ch in ch_to_size and i < len(sizes):
                    sizes[i] = ch_to_size[ch]
            if len(sizes) == len(containers):
                self.splitter.setSizes(sizes)

    def _refresh_plots(self, progress_callback: Callable[[str], None] | None = None):
        """同步构建：逐通道建金字塔并安装曲线（测试/兼容路径）。

        主窗口激活路径走 start_display_build() 的异步版本。
        E/B/H 通道绘制为去均值折线图，Kp 绘制为三色柱状图。
        """
        if self.em_data is None:
            return

        x_vals = dti_to_numpy(self.em_data.datetime_index)
        channel_count = self.em_data.NCh

        for idx, (key, y_data) in enumerate(self.em_data.iter_channels()):
            if key not in self.plot_rows:
                continue
            # LOD 保留原始数组；居中在取出可见切片后完成，避免为每个
            # 通道永久保留一份全长 centered 副本。
            values = np.asarray(y_data.values, dtype=np.float64)
            self._pyramids[key] = self._build_channel_pyramid(x_vals, values)
            center, scale = self._display_parameters(values)
            self._channel_centers[key] = center
            self._channel_scales[key] = scale
            if progress_callback is not None:
                progress_callback(f"正在初始化通道 {key} ({idx+1}/{channel_count})")

        self._install_display()

    @staticmethod
    def _build_channel_pyramid(x_vals: np.ndarray, values: np.ndarray) -> LODPyramid:
        """基于原始值构建单通道金字塔（纯 numpy/Rust，可在池线程执行）。"""
        return LODPyramid(x_vals, values, max_finest_points=PYRAMID_MAX_FINEST_POINTS)

    @staticmethod
    def _display_parameters(values: np.ndarray) -> tuple[float, float]:
        """Return nanmean center and robust per-channel display scale."""
        mean = safe_nanmean(values)
        center = float(mean) if np.isfinite(mean) else 0.0
        centered = np.asarray(values - center, dtype=np.float64)
        finite = np.abs(centered[np.isfinite(centered)])
        if finite.size == 0:
            return center, 1.0
        scale = float(np.nanpercentile(finite, 98))
        return center, scale if np.isfinite(scale) and scale > 0 else 1.0

    def _install_display(self) -> None:
        """按已装金字塔从当前视口取数建堆叠曲线（主线程执行）。"""
        if self.em_data is None:
            return
        view_start = self.em_data.start_time + self.x_view_start
        view_end = view_start + self.x_view_size
        self.plot_widget.clear_curves()
        self._update_stacked_layout()
        for key in self._channel_order:
            if key == "Kp":
                continue
            pyramid = self._pyramids.get(key)
            if pyramid is None:
                continue
            xs, ys = pyramid.get(
                view_start.timestamp(), view_end.timestamp(), max(1, self.plot_widget.width()),
            )
            offset = self._stack_offsets.get(key, 0.0)
            center = self._channel_centers.get(key, 0.0)
            scale = self._channel_scales.get(key, 1.0)
            self.plot_widget.add_curve(
                x_data=xs,
                y_data=(ys - center) / max(scale, 1e-12) * self._display_gain + offset,
                label=key, graph_type="plot",
            )

        # add_curve 可能触发 PyQtGraph 自动范围；曲线全部安装后再固定
        # 堆叠显示范围，确保各通道均匀分布在整个绘图区内。
        self._update_stacked_layout()
        self.plot_widget.sync_legend_visibility(self._visible_wave_channels())
        if self.em_data.kp_data is not None:
            self.plot_widget.add_kp_overlay(
                self.em_data.kp_data["Kp_datetime"], self.em_data.kp_data["Kp"]
            )
            self.plot_widget.set_kp_visible(
                self._checkboxes.get("Kp") is None or self._checkboxes["Kp"].isChecked()
            )
        self.plot_widget.update_plot(view_start, view_end)

    def start_display_build(self) -> None:
        """在计算池异步构建逐通道金字塔，完成后主线程安装并发 display_ready。

        金字塔构建是 viewer 创建的重头（5s 采样年度文件约数秒），放后台
        避免主线程冻结；进度框由调用方（MainWindow）管理。
        """
        if self.em_data is None:
            return
        self._build_gen += 1
        self._channel_build_gen.clear()
        gen = self._build_gen
        em_data = self.em_data
        self._pending_gen = gen
        self._pending_data = em_data
        channel_values = [
            (key, y_data.values)
            for key, y_data in em_data.iter_channels()
            if key in self.plot_rows
        ]

        def task():
            x_vals = dti_to_numpy(em_data.datetime_index)
            pyramids = {
                key: (
                    DataViewWidget._build_channel_pyramid(x_vals, values),
                    DataViewWidget._display_parameters(values),
                )
                for key, values in channel_values
            }
            return gen, em_data, pyramids

        # on_done/on_error 是 QObject 绑定方法 → 排队连接回主线程
        # （切勿换成闭包：闭包接收者会退化成 DirectConnection，
        # 在池线程里直接碰控件）
        run_in_pool(
            "compute", task,
            on_done=self._on_display_build_done,
            on_error=self._on_display_build_failed,
            owner=self,
        )

    def _on_display_build_done(self, payload) -> None:
        gen, em_data, pyramids = payload
        if self._build_gen != gen or self.em_data is not em_data:
            return  # 已被更新的 init_data / 新构建取代，丢弃过期结果
        self._pending_gen = -1
        self._pending_data = None
        self._pyramids = {
            k: item[0] for k, item in pyramids.items() if k in self.plot_rows
        }
        self._channel_centers = {
            k: item[1][0] for k, item in pyramids.items() if k in self.plot_rows
        }
        self._channel_scales = {
            k: item[1][1] for k, item in pyramids.items() if k in self.plot_rows
        }
        self._install_display()
        self.display_ready.emit()

    def _on_display_build_failed(self, exc: BaseException) -> None:
        logger.error("显示数据构建失败: %r", exc, exc_info=exc)

    def _start_channel_display_build(self, channel: str) -> None:
        """仅在计算池重建已变更通道，复用现有时间轴和其他通道金字塔。"""
        em_data = self.em_data
        if (
            em_data is None
            or channel not in self.plot_rows
            or not em_data.has_channel(channel)
        ):
            return

        # 全量构建抓取的是更新前的数据快照；通道变化时启动新一代全量
        # 构建，避免旧全量结果随后覆盖新通道结果。
        if self._pending_gen == self._build_gen and self._pending_data is em_data:
            self.start_display_build()
            return

        x_vals = next(
            (
                pyramid.x_raw
                for pyramid in self._pyramids.values()
                if pyramid.x_raw.size == em_data.npts
            ),
            None,
        )
        if x_vals is None:
            self.start_display_build()
            return

        values = np.asarray(
            em_data.get_channel_array(channel).values,
            dtype=np.float64,
        )
        generation = self._channel_build_gen.get(channel, 0) + 1
        self._channel_build_gen[channel] = generation

        def task():
            pyramid = DataViewWidget._build_channel_pyramid(x_vals, values)
            parameters = DataViewWidget._display_parameters(values)
            return channel, generation, em_data, pyramid, parameters

        run_in_pool(
            "compute", task,
            on_done=self._on_channel_display_build_done,
            on_error=self._on_display_build_failed,
            owner=self,
        )

    def _on_channel_display_build_done(self, payload) -> None:
        channel, generation, em_data, pyramid, parameters = payload
        if (
            self.em_data is not em_data
            or self._channel_build_gen.get(channel) != generation
            or channel not in self.plot_rows
        ):
            return
        self._pyramids[channel] = pyramid
        self._channel_centers[channel], self._channel_scales[channel] = parameters
        self._on_viewport_changed()
        self.plot_widget.show()

    # -------------------------------------------------------------------------
    # 视口更新（实现 Mixin 钩子）
    # -------------------------------------------------------------------------

    def _on_viewport_changed(self) -> None:
        """实现 Mixin 钩子：视口变化时更新所有子图。

        折线通道的曲线数据在主线程同步替换为"视口适配像素宽度"的
        LOD 降采样数据（查询是 O(log n) 级，开销可忽略）；X 轴范围
        直接在主线程设置——视口更新已在 Mixin 侧 16ms 防抖，不需要
        再经工作线程中转。
        """
        if self.em_data is None:
            return
        start = self.em_data.start_time + self.x_view_start
        end = start + self.x_view_size
        x_start = start.timestamp()
        x_end = end.timestamp()
        # 先根据当前可见通道计算新槽位，再更新曲线数据；否则切换复选框
        # 时会出现标签/基线已移动、曲线仍使用旧偏移的错位。
        self._update_stacked_layout()
        for ch in self._visible_wave_channels():
            pyramid = self._pyramids.get(ch)
            if pyramid is None:
                continue
            xs, ys = pyramid.get(x_start, x_end, max(1, self.plot_widget.width()))
            if len(xs) > 0:
                center = self._channel_centers.get(ch, 0.0)
                scale = self._channel_scales.get(ch, 1.0)
                offset = self._stack_offsets.get(ch, 0.0)
                self.plot_widget.update_curve(
                    ch, xs,
                    (ys - center) / max(scale, 1e-12) * self._display_gain + offset,
                )
        self.plot_widget.sync_legend_visibility(self._visible_wave_channels())
        self.plot_widget.update_plot(start, end)
        self.set_time(pts_to_qdt(start), pts_to_qdt(end))

    def resizeEvent(self, event) -> None:
        """窗口宽度变化后按新像素宽度重查 LOD（经防抖定时器合并）。"""
        super().resizeEvent(event)
        if self.em_data is not None and self._viewport_debounce is not None:
            self._viewport_debounce.start()

    # -------------------------------------------------------------------------
    # 槽函数
    # -------------------------------------------------------------------------

    def _on_scroll_drag_start(self) -> None:
        """滚动条开始拖动；视口变化已采用直接刷新。"""
        return

    def _on_scroll_drag_end(self) -> None:
        """松开滚动条后做一次最终刷新。"""
        if self._viewport_debounce is not None:
            self._viewport_debounce.stop()
        self._on_viewport_changed()

    def on_x_view_start_changed(self) -> None:
        """横向滚动条变化时立即刷新，不再等待防抖定时器。"""
        self.x_view_start = pd.Timedelta(seconds=self.scroll_x.value())
        if self._viewport_debounce is not None:
            self._viewport_debounce.stop()
        self._on_viewport_changed()

    def on_x_view_size_changed(self) -> None:
        """窗口长度变化时直接刷新，LOD 已足以支撑连续拖动。"""
        self.x_view_size = pd.Timedelta(seconds=self.slide_time.value())
        max_start = self.duration - self.x_view_size
        self.scroll_x.setMaximum(max(0, int(max_start.total_seconds())))
        self.scroll_x.setPageStep(int(self.x_view_size.total_seconds()))
        if self.x_view_start > max_start:
            self.x_view_start = max_start
            self.scroll_x.setValue(int(self.x_view_start.total_seconds()))
        if self._viewport_debounce is not None:
            self._viewport_debounce.stop()
        self._on_viewport_changed()

    def on_btn_plot_clicked(self) -> None:
        """跳转到指定时间范围。"""
        start = qdt_to_pts(self.start_time.dateTime())
        end = qdt_to_pts(self.end_time.dateTime())
        start_duration, end_duration = self._clamp_jump(                    # Mixin
            start, end,
            data_start=self.em_data.start_time,
            data_end=self.em_data.end_time,
            tz=self.em_data.start_time.tz,
        )
        self._apply_jump(start_duration, end_duration)                      # Mixin

    def on_change_data_finished(self, channel: str) -> None:
        """外部通知某通道（或全部通道）数据已更新。"""
        if self.em_data is None:
            return
        if channel and channel not in self.plot_rows:
            self._sync_plot_rows()
        if channel and self.em_data.has_channel(channel):
            self._start_channel_display_build(channel)
        else:
            self.start_display_build()

    def shutdown(self) -> None:
        """安全关闭所有工作线程并释放 UI 资源。"""
        if self._shutdown_done:
            return
        self._shutdown_done = True

        for ch in list(self.plot_rows.keys()):
            self._teardown_plot_row(ch)

    def closeEvent(self, event) -> None:
        """窗口关闭时自动清理所有后台线程。"""
        self.shutdown()
        super().closeEvent(event)
