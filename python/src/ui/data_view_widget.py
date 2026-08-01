from __future__ import annotations

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
from PySide6.QtCore import Qt, QThread, Signal, QPoint, QMimeData
from PySide6.QtGui import QFont, QMouseEvent, QDrag, QPixmap, QPainter, QColor

from ui.brief_plot import BriefPlot, BriefWorker
from utils.time_convert import pts_to_qdt, qdt_to_pts
from utils.series import dti_to_numpy
from utils.decimate import LODPyramid
from core.em_data import EMData
from base.time_viewport_mixin import TimeViewportMixin

# 自定义 MIME 类型用于通道拖拽排序
_CHANNEL_DRAG_MIME = "application/x-channel-reorder"


@dataclass
class PlotResources:
    widget: Optional[BriefPlot] = None
    row_container: Optional[QWidget] = None
    thread: Optional[QThread] = None
    worker: Optional[BriefWorker] = None


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

    def init_data(self, data: EMData, progress_callback: Callable[[str], None] | None = None):
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

        self.slide_time.setMinimum(int(pd.Timedelta(hours=1).total_seconds()))
        self.slide_time.setMaximum(int(self.duration.total_seconds()))
        self.slide_time.setValue(int(self.x_view_size.total_seconds()))
        self.slide_time.setEnabled(True)

        self.scroll_x.setMinimum(0)
        self.scroll_x.setMaximum(int((self.duration - self.x_view_size).total_seconds()))
        self.scroll_x.setValue(int(self.x_view_start.total_seconds()))
        self.scroll_x.setPageStep(int(self.x_view_size.total_seconds()))
        self.scroll_x.setEnabled(True)

        self._build_channel_order()
        self._sync_plot_rows()
        self._refresh_plots(progress_callback)
        self._rebuild_checkboxes()
        self._on_viewport_changed()

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
        """为通道创建 BriefPlot、工作线程和容器（若尚不存在）。"""
        if ch not in self.plot_rows:
            self.plot_rows[ch] = PlotResources()

        resources = self.plot_rows[ch]

        if resources.widget is None:
            widget = BriefPlot(label=ch)
            thread = QThread(self)
            worker = BriefWorker(label=ch)
            # 将 worker 移到子线程，update 信号跨线程排队调用 widget.update_plot
            worker.moveToThread(thread)
            worker.update.connect(widget.update_plot)
            thread.start()

            container = PlotRowContainer(ch, widget)
            container.drop_reorder.connect(self._on_drop_reorder)

            resources.widget = widget
            resources.row_container = container
            resources.worker = worker
            resources.thread = thread
            self.splitter.addWidget(container)
        else:
            resources.widget.set_label(ch)

    def _teardown_plot_row(self, ch: str) -> None:
        """安全销毁通道的工作线程、容器和绘图控件。"""
        resources = self.plot_rows.pop(ch, None)
        self._pyramids.pop(ch, None)
        if resources is None:
            return

        # 先退出工作线程，再清理 UI 控件
        if resources.thread is not None:
            resources.thread.quit()
            resources.thread.wait()
            resources.thread.deleteLater()

        if resources.worker is not None:
            resources.worker.deleteLater()

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
        """每个通道一个 BriefPlot，每个 BriefPlot 只画一条曲线。

        E/B/H 通道绘制为去均值折线图，Kp 绘制为三色柱状图。
        """
        if self.em_data is None:
            return

        x_vals = dti_to_numpy(self.em_data.datetime_index)
        channel_count = self.em_data.NCh
        view_start = self.em_data.start_time + self.x_view_start
        view_end = view_start + self.x_view_size

        for idx, (key, y_data) in enumerate(self.em_data.iter_channels()):
            if key not in self.plot_rows:
                continue
            widget = self.plot_rows[key].widget
            if widget is None:
                continue
            widget.clear_curves()
            with np.errstate(all='ignore'):
                mean = np.nanmean(y_data.values)
            y_vals = np.asarray(y_data.values - (mean if np.isfinite(mean) else 0.0), dtype=np.float64)
            # 构建 LOD 金字塔（Rust 加速），曲线只接收当前视口适配像素
            # 宽度的降采样数据；数据总量再大，单次绘制的点数也有上限
            self._pyramids[key] = LODPyramid(x_vals, y_vals)
            xs, ys = self._pyramids[key].get(
                view_start.timestamp(), view_end.timestamp(), max(1, widget.width()),
            )
            widget.add_curve(
                x_data=xs,
                y_data=ys,
                label=key,
                graph_type='plot',
            )
            if progress_callback is not None:
                progress_callback(f"正在初始化通道 {key} ({idx+1}/{channel_count})")

        # Kp 数据使用柱状图（三色方案：蓝/黄/红）
        if self.em_data.kp_data is not None and 'Kp' in self.plot_rows:
            if progress_callback is not None:
                progress_callback("正在初始化 Kp 指数")
            kp_widget = self.plot_rows['Kp'].widget
            if kp_widget is not None:
                kp_widget.clear_curves()
                kp_widget.add_curve(
                    x_data=self.em_data.kp_data['Kp_datetime'],
                    y_data=self.em_data.kp_data['Kp'],
                    label='Kp',
                    graph_type='bar',
                )

        # 初始加载后直接设置 X 轴范围，使曲线立即可见，
        # 避免等待跨线程 worker 信号排队延迟。
        for resources in self.plot_rows.values():
            if resources.widget is not None:
                resources.widget.update_plot(view_start, view_end)
        self._sync_left_axes()

    # -------------------------------------------------------------------------
    # 视口更新（实现 Mixin 钩子）
    # -------------------------------------------------------------------------

    def _on_viewport_changed(self) -> None:
        """实现 Mixin 钩子：视口变化时更新所有子图。

        折线通道的曲线数据在主线程同步替换为"视口适配像素宽度"的
        LOD 降采样数据（查询是 O(log n) 级，开销可忽略）；X 轴范围
        仍经各通道 worker 信号排队更新。
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
            if resources.worker is not None:
                resources.worker.range_change(start, end)
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
