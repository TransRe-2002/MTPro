from __future__ import annotations

import logging
from dataclasses import dataclass
from collections.abc import Callable
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter,
    QSizePolicy, QSlider, QDateTimeEdit, QScrollBar,
    QLabel, QPushButton, QFrame, QHBoxLayout, QGridLayout, QGroupBox,
    QCheckBox, QApplication,
)
from PySide6.QtCore import Qt, Signal, QPoint, QMimeData
from PySide6.QtGui import QFont, QMouseEvent, QDrag, QPixmap, QPainter, QColor

from ui.brief_plot import BriefPlot
from utils.time_convert import pts_to_qdt, qdt_to_pts
from utils.series import dti_to_numpy
from utils.decimate import LODPyramid, PYRAMID_MAX_FINEST_POINTS
from utils.qt_bridge import run as run_in_pool
from utils.signals import block_signals

logger = logging.getLogger(__name__)
from core.em_data import EMData
from base.time_viewport_mixin import TimeViewportMixin

# 自定义 MIME 类型用于通道拖拽排序
_CHANNEL_DRAG_MIME = "application/x-channel-reorder"


@dataclass
class PlotResources:
    widget: Optional[BriefPlot] = None
    row_container: Optional[QWidget] = None


class PlotRowContainer(QFrame):
    """通道行容器：整体可拖拽重排序，同时接受其他行的 QDrag 释放。"""

    drop_reorder = Signal(str, int)  # channel_name, target_index

    def __init__(self, channel: str, brief_plot: BriefPlot, parent=None):
        super().__init__(parent)
        self._channel = channel
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.NoFrame)

        # 左侧拖拽把手 + 右侧 BriefPlot
        row_layout = QHBoxLayout(self)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)

        # 拖拽把手（左侧竖条）
        handle = QFrame(self)
        handle.setFixedWidth(13)
        handle.setStyleSheet(
            "QFrame { background: palette(button); border-right: 1px solid palette(mid); }"
        )
        handle_layout = QVBoxLayout(handle)
        handle_layout.setContentsMargins(0, 0, 0, 0)
        handle_layout.setSpacing(0)
        handle_label = QLabel(">>")
        handle_label.setFont(QFont("Arial", 9))
        handle_label.setStyleSheet("color: palette(mid);")
        handle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        handle_layout.addWidget(handle_label)
        row_layout.addWidget(handle)

        row_layout.addWidget(brief_plot, 1)

        self._press_pos: Optional[QPoint] = None

    # ── 作为拖拽源：整行可拖 ────────────────────────────────

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        """记录左键按下位置，用于后续拖拽检测。"""
        if ev.button() == Qt.MouseButton.LeftButton:
            self._press_pos = ev.position().toPoint()
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        """鼠标移动超过 10px 曼哈顿距离后启动 QDrag 拖拽重排序。"""
        if self._press_pos is None:
            super().mouseMoveEvent(ev)
            return
        # 防止轻微抖动误触发拖拽
        if (ev.position().toPoint() - self._press_pos).manhattanLength() < 10:
            ev.accept()
            return

        # 构建 QDrag：MIME 数据携带通道名
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(_CHANNEL_DRAG_MIME, self._channel.encode())
        drag.setMimeData(mime)

        # 拖拽预览：整行缩略图
        pixmap = self.grab().scaled(
            self.width() // 2, self.height() // 2,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))
        drag.exec(Qt.DropAction.MoveAction)

        self._press_pos = None
        ev.accept()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        self._press_pos = None
        super().mouseReleaseEvent(ev)

    # ── 作为释放目标 ────────────────────────────────────────

    def _index_in_splitter(self) -> int:
        splitter = self.parent()
        if isinstance(splitter, QSplitter):
            return splitter.indexOf(self)
        return -1

    def dragEnterEvent(self, ev) -> None:
        """拖拽进入时显示蓝色顶部边框作为放置位置指示。"""
        if ev.mimeData().hasFormat(_CHANNEL_DRAG_MIME):
            ev.acceptProposedAction()
            # 蓝色顶部边框指示插入位置
            self.setStyleSheet(
                "PlotRowContainer { border-top: 3px solid #4080c0; }"
                "PlotRowContainer QFrame { border-bottom: 1px solid #c0c0c0; }"
            )
        else:
            super().dragEnterEvent(ev)

    def dragMoveEvent(self, ev) -> None:
        if ev.mimeData().hasFormat(_CHANNEL_DRAG_MIME):
            ev.acceptProposedAction()
        else:
            super().dragMoveEvent(ev)

    def dragLeaveEvent(self, ev) -> None:
        self.setStyleSheet("")
        super().dragLeaveEvent(ev)

    def dropEvent(self, ev) -> None:
        """拖拽释放：解码被拖通道名，查找目标索引，发出重排序信号。"""
        self.setStyleSheet("")
        if not ev.mimeData().hasFormat(_CHANNEL_DRAG_MIME):
            super().dropEvent(ev)
            return

        # 从 MIME 数据中解码出被拖拽的通道名
        dragged_ch = bytes(ev.mimeData().data(_CHANNEL_DRAG_MIME)).decode()
        if dragged_ch == self._channel:
            ev.ignore()
            return

        target_idx = self._index_in_splitter()
        if target_idx >= 0:
            ev.acceptProposedAction()
            self.drop_reorder.emit(dragged_ch, target_idx)
        else:
            ev.ignore()


class DataViewWidget(TimeViewportMixin, QWidget):

    # 异步显示构建完成后发出（主线程），供外部关闭进度框
    display_ready = Signal()

    # -------------------------------------------------------------------------
    # 初始化
    # -------------------------------------------------------------------------

    def __init__(self, parent=None):
        super().__init__(parent)

        self._viewport_debounce_init()  # Mixin: 初始化视口更新防抖

        self.em_data: Optional[EMData] = None
        self.x_view_start: pd.Timedelta = pd.Timedelta(seconds=0)
        self.x_view_size: pd.Timedelta = pd.Timedelta(hours=1)
        self.duration: pd.Timedelta = pd.Timedelta(hours=0)
        self._build_gen: int = 0           # 显示构建代次（防中途切换/重初始化串台）
        self._pending_gen: int = -1
        self._pending_data: Optional[EMData] = None

        # 每个通道一个独立的 BriefPlot（按通道名索引）
        self.plot_rows: Dict[str, PlotResources] = {}
        # 每通道的 LOD 金字塔：显示层只画视口适配分辨率的数据，
        # 全分辨率数据仍留在 EMData 中供处理工具使用
        self._pyramids: Dict[str, LODPyramid] = {}
        # 有序通道列表（控制显示顺序）
        self._channel_order: List[str] = []

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
        self._checkboxes: Dict[str, QCheckBox] = {}

        self.init_ui()
        self.connect_signals()
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
        view_state: Optional[dict] = None,
        build_plots: bool = True,
    ):
        """初始化/更换数据。

        build_plots=True 时同步构建金字塔与曲线（测试/兼容路径）；
        False 时只建控件与状态，金字塔走 start_display_build() 异步
        构建（主窗口激活路径，避免大文件冻住 UI）。
        """
        self._build_gen += 1
        self.em_data = data
        if self.em_data is None:
            self._pyramids.clear()
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

    # -------------------------------------------------------------------------
    # 交互现场状态的快照与恢复（active-only 缓存策略的配套）
    # -------------------------------------------------------------------------

    def capture_view_state(self) -> dict:
        """快照当前交互现场：视口、通道顺序、隐藏通道、各行高度。

        viewer 被销毁前调用；下次激活同一数据集时经 init_data
        的 view_state 参数恢复，用户无感。
        """
        channel_order = self._splitter_channel_order() or list(self._channel_order)
        sizes = self.splitter.sizes()
        hidden = [ch for ch, cb in self._checkboxes.items() if not cb.isChecked()]
        return {
            "x_view_start": self.x_view_start,
            "x_view_size": self.x_view_size,
            "channel_order": channel_order,
            "hidden_channels": hidden,
            "splitter_sizes": dict(zip(channel_order, sizes)),
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
        sizes_map = view_state.get("splitter_sizes") or {}
        order = self._splitter_channel_order()
        sizes = []
        for ch in order:
            s = sizes_map.get(ch)
            if not isinstance(s, int) or s <= 0:
                return  # 记录不全则保持默认布局
            sizes.append(s)
        if len(sizes) == self.splitter.count() > 1:
            self.splitter.setSizes(sizes)

    def _build_channel_order(self) -> None:
        """构建有序通道列表：E 通道 → B/H 通道 → Kp（最后）。"""
        if self.em_data is None:
            self._channel_order = []
            return
        e_channels: List[str] = []
        b_channels: List[str] = []
        # 按首字母分类：E 通道在上，B/H 通道居中
        for ch in self.em_data.channel_names():
            if ch.startswith('E'):
                e_channels.append(ch)
            elif ch.startswith('B') or ch.startswith('H'):
                b_channels.append(ch)
        e_channels.sort()
        b_channels.sort()
        self._channel_order = e_channels + b_channels
        # Kp 数据始终排在最后
        if self.em_data.kp_data is not None:
            self._channel_order.append('Kp')

    def _widget_index_in_splitter(self, ch: str) -> int:
        """返回某通道的 row_container 在 splitter 中的索引。"""
        resources = self.plot_rows.get(ch)
        if resources is None or resources.row_container is None:
            return -1
        return self.splitter.indexOf(resources.row_container)

    def _move_channel(self, ch: str, direction: int) -> None:
        """将通道在 splitter 中上移(-1)或下移(+1)。"""
        idx = self._widget_index_in_splitter(ch)
        if idx < 0:
            return
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= self.splitter.count():
            return
        resources = self.plot_rows[ch]
        container = resources.row_container
        container.hide()
        self.splitter.insertWidget(new_idx, container)
        container.show()
        # 同步 _channel_order
        if ch in self._channel_order:
            old = self._channel_order.index(ch)
            self._channel_order.pop(old)
            insert_at = old + direction
            # 计算新位置在 channel_order 中的正确位置
            other_ch = None
            for c, res in self.plot_rows.items():
                if res.row_container is not None and self._widget_index_in_splitter(c) == idx:
                    other_ch = c
                    break
            # 简化：直接根据 splitter 重建 channel_order
            self._channel_order = self._splitter_channel_order()

    def _splitter_channel_order(self) -> List[str]:
        """从 splitter 中读取当前 widget 排列顺序。"""
        order: List[str] = []
        for i in range(self.splitter.count()):
            widget = self.splitter.widget(i)
            for ch, resources in self.plot_rows.items():
                if resources.row_container is widget:
                    order.append(ch)
                    break
        return order

    def _on_drop_reorder(self, dragged_ch: str, target_idx: int) -> None:
        """QDrag 释放 → 将被拖通道移动到目标位置。

        使用 splitter.insertWidget() 完成 widget 重排，然后从 splitter
        同步 _channel_order 以保持一致性。
        """
        res = self.plot_rows.get(dragged_ch)
        if res is None or res.row_container is None:
            return
        container = res.row_container
        old_idx = self.splitter.indexOf(container)
        if old_idx < 0 or old_idx == target_idx:
            return
        # 在 splitter 中移动到新位置
        container.hide()
        self.splitter.insertWidget(target_idx, container)
        container.show()
        # 从 splitter 重建 channel_order，与 widget 顺序保持一致
        self._channel_order = self._splitter_channel_order()

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
        # 默认勾选：Kp始终勾选 + 其余通道取前7个；总数≤8则全勾
        total = len(self._channel_order)
        if total <= 8:
            default_checked = {ch for ch in self._channel_order}
        else:
            non_kp = [ch for ch in self._channel_order if ch != 'Kp'][:7]
            default_checked = set(non_kp)
            if 'Kp' in self._channel_order:
                default_checked.add('Kp')

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
        """CheckBox 切换 → 显示/隐藏对应 BriefPlot。"""
        resources = self.plot_rows.get(ch)
        if resources is None or resources.row_container is None:
            return
        visible = state == Qt.CheckState.Checked.value
        resources.row_container.setVisible(visible)
        self._sync_left_axes()

    def _sync_left_axes(self) -> None:
        """Restore waveform subplot axis sizing and labels."""
        for resources in self.plot_rows.values():
            widget = resources.widget
            if widget is None:
                continue
            widget.sync_axis_layout()

    # -------------------------------------------------------------------------
    # 通道 / Plot 管理
    # -------------------------------------------------------------------------

    def _required_channels(self) -> set[str]:
        """返回所有需要显示的通道名。"""
        if self.em_data is None:
            return set()
        required: set[str] = set(self.em_data.channel_names())
        if self.em_data.kp_data is not None:
            required.add('Kp')
        return required

    def _ensure_plot_row(self, ch: str) -> None:
        """为通道创建 BriefPlot 和容器（若尚不存在）。"""
        if ch not in self.plot_rows:
            self.plot_rows[ch] = PlotResources()

        resources = self.plot_rows[ch]

        if resources.widget is None:
            widget = BriefPlot(label=ch)

            container = PlotRowContainer(ch, widget)
            container.drop_reorder.connect(self._on_drop_reorder)

            resources.widget = widget
            resources.row_container = container
            self.splitter.addWidget(container)
        else:
            resources.widget.set_label(ch)

    def _teardown_plot_row(self, ch: str) -> None:
        """销毁通道的容器和绘图控件。"""
        resources = self.plot_rows.pop(ch, None)
        self._pyramids.pop(ch, None)
        if resources is None:
            return

        if resources.row_container is not None:
            resources.row_container.hide()
            resources.row_container.setParent(None)
            resources.row_container.deleteLater()

        if resources.widget is not None:
            resources.widget.deleteLater()

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

        # 按 channel_order 排好 splitter 中的顺序
        self._reorder_splitter()
        self._sync_left_axes()

    def _reorder_splitter(self) -> None:
        """确保 splitter 中 widget 的排列顺序与 _channel_order 一致。

        实现方式：先全部取出（保存各容器原尺寸），按 channel_order 顺序重新插入，恢复尺寸。
        """
        if len(self._channel_order) <= 1:
            return
        # 先全部取出，再按顺序插入
        containers: List[tuple[str, QWidget]] = []
        for ch in self._channel_order:
            res = self.plot_rows.get(ch)
            if res is not None and res.row_container is not None:
                containers.append((ch, res.row_container))

        if len(containers) <= 1:
            return

        # 记录当前 sizes，便于恢复各通道之前的高度比例
        old_sizes = self.splitter.sizes()
        ch_to_size: Dict[str, int] = {}
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
            # 构建 LOD 金字塔（Rust 加速 + 截顶）；曲线在 _install_display
            # 里只接收当前视口适配像素宽度的降采样数据
            self._pyramids[key] = self._build_channel_pyramid(x_vals, y_data.values)
            if progress_callback is not None:
                progress_callback(f"正在初始化通道 {key} ({idx+1}/{channel_count})")

        self._install_display()

    @staticmethod
    def _build_channel_pyramid(x_vals: np.ndarray, values: np.ndarray) -> LODPyramid:
        """去均值后构建单通道金字塔（纯 numpy/Rust，可在池线程执行）。"""
        with np.errstate(all='ignore'):
            mean = np.nanmean(values)
        y_vals = np.asarray(values - (mean if np.isfinite(mean) else 0.0), dtype=np.float64)
        return LODPyramid(x_vals, y_vals, max_finest_points=PYRAMID_MAX_FINEST_POINTS)

    def _install_display(self) -> None:
        """按已装金字塔从当前视口取数建曲线（主线程执行）。"""
        if self.em_data is None:
            return
        view_start = self.em_data.start_time + self.x_view_start
        view_end = view_start + self.x_view_size
        for key, resources in self.plot_rows.items():
            widget = resources.widget
            pyramid = self._pyramids.get(key)
            if widget is None or pyramid is None:
                continue
            widget.clear_curves()
            xs, ys = pyramid.get(
                view_start.timestamp(), view_end.timestamp(), max(1, widget.width()),
            )
            widget.add_curve(x_data=xs, y_data=ys, label=key, graph_type='plot')

        # Kp 数据使用柱状图（三色方案：蓝/黄/红），数据量小，全量绘制
        if self.em_data.kp_data is not None and 'Kp' in self.plot_rows:
            kp_widget = self.plot_rows['Kp'].widget
            if kp_widget is not None:
                kp_widget.clear_curves()
                kp_widget.add_curve(
                    x_data=self.em_data.kp_data['Kp_datetime'],
                    y_data=self.em_data.kp_data['Kp'],
                    label='Kp',
                    graph_type='bar',
                )

        for resources in self.plot_rows.values():
            if resources.widget is not None:
                resources.widget.update_plot(view_start, view_end)
        self._sync_left_axes()

    def start_display_build(self) -> None:
        """在 io 池异步构建逐通道金字塔，完成后主线程安装并发 display_ready。

        金字塔构建是 viewer 创建的重头（5s 采样年度文件约数秒），放后台
        避免主线程冻结；进度框由调用方（MainWindow）管理。
        """
        if self.em_data is None:
            return
        self._build_gen += 1
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
            return {
                key: DataViewWidget._build_channel_pyramid(x_vals, values)
                for key, values in channel_values
            }

        # on_done/on_error 是 QObject 绑定方法 → 排队连接回主线程
        # （切勿换成闭包：闭包接收者会退化成 DirectConnection，
        # 在池线程里直接碰控件）
        run_in_pool(
            "io", task,
            on_done=self._on_display_build_done,
            on_error=self._on_display_build_failed,
            owner=self,
        )

    def _on_display_build_done(self, pyramids) -> None:
        if self._build_gen != self._pending_gen or self.em_data is not self._pending_data:
            return  # 已被更新的 init_data 取代，丢弃过期结果
        self._pyramids = {k: p for k, p in pyramids.items() if k in self.plot_rows}
        self._install_display()
        self.display_ready.emit()

    def _on_display_build_failed(self, exc: BaseException) -> None:
        logger.error("显示数据构建失败: %r", exc, exc_info=exc)

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
        for ch, resources in self.plot_rows.items():
            widget = resources.widget
            pyramid = self._pyramids.get(ch)
            if widget is None or pyramid is None:
                continue
            xs, ys = pyramid.get(x_start, x_end, max(1, widget.width()))
            if len(xs) > 0:
                widget.update_curve(ch, xs, ys)
        for resources in self.plot_rows.values():
            if resources.widget is not None:
                resources.widget.update_plot(start, end)
        self._sync_left_axes()
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
        """滚动条开始拖动：切换到更宽松的去抖间隔以降低渲染负载。"""
        if self._viewport_debounce is not None:
            self._viewport_debounce.setInterval(50)

    def _on_scroll_drag_end(self) -> None:
        """滚动条拖动结束：恢复短间隔去抖并立即刷新。"""
        if self._viewport_debounce is not None:
            self._viewport_debounce.setInterval(16)
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
        if channel == "":
            self._refresh_plots()
            return
        x_vals = dti_to_numpy(self.em_data.datetime_index)

        if channel not in self.plot_rows:
            self._sync_plot_rows()

        if channel not in self.plot_rows:
            return

        ch_series = self.em_data.get_channel_series(channel)
        widget = self.plot_rows[channel].widget
        with np.errstate(all='ignore'):
            mean = np.nanmean(ch_series.values)
        y_vals = np.asarray(ch_series.values - (mean if np.isfinite(mean) else 0.0), dtype=np.float64)
        # 通道数据变了：重建该通道的 LOD 金字塔，并以当前视口数据重画
        pyramid = LODPyramid(x_vals, y_vals)
        self._pyramids[channel] = pyramid
        if widget is not None:
            view_start = self.em_data.start_time + self.x_view_start
            view_end = view_start + self.x_view_size
            xs, ys = pyramid.get(
                view_start.timestamp(), view_end.timestamp(), max(1, widget.width()),
            )
            widget.clear_curves()
            widget.add_curve(xs, ys, channel, graph_type='plot')
            widget.show()
        self._sync_left_axes()
        self._on_viewport_changed()

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
