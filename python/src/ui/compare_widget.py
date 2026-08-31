from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDateTimeEdit,
    QLabel,
    QPushButton,
    QScrollBar,
    QSizePolicy,
    QSlider,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from base.time_viewport_mixin import TimeViewportMixin
from core.em_data import EMData
from ui.brief_plot import BriefPlot
from utils.series import dti_to_numpy, safe_nanmean
from utils.decimate import LODPyramid, PYRAMID_MAX_FINEST_POINTS
from utils.qt_bridge import run as run_in_pool
from utils.time_convert import pts_to_qdt, qdt_to_pts

logger = logging.getLogger(__name__)


class CompareWidget(TimeViewportMixin, QWidget):
    """序列对比控件：将多个 EMData 通道按 E/B 分组绘制到上下两个子图中。

    上方子图(row='E')显示 Ex 通道，下方子图(row='B')显示 Bx/Hx 通道。
    """

    # 异步显示构建完成（主线程），供外部关进度框等
    display_ready = Signal()

    def __init__(
        self,
        channel_refs: list[tuple[EMData, str]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._viewport_debounce_init()  # Mixin: 初始化视口更新防抖

        # 用极端值初始化起止时间，便于遍历时取 min/max（内部统一 UTC）
        self.start: pd.Timestamp = pd.Timestamp("2200-01-01 00:00:00", tz="UTC")
        self.end: pd.Timestamp = pd.Timestamp("1900-01-01 00:00:00", tz="UTC")
        self.channel_refs: list[tuple[EMData, str]] = channel_refs

        # 遍历所有参考通道，找到全局时间范围
        for em_data, channel_name in self.channel_refs:
            ch_start, ch_end = em_data.channel_time_range(channel_name)
            if ch_start < self.start:
                self.start = ch_start
            if ch_end > self.end:
                self.end = ch_end

        self.x_view_start: pd.Timedelta = pd.Timedelta(seconds=0)
        self.x_view_size: pd.Timedelta = pd.Timedelta(hours=1)
        self.duration: pd.Timedelta = pd.Timedelta(hours=0)

        # 每条曲线的 LOD 金字塔（label → LODPyramid）与异步构建代次守卫
        self._pyramids: dict[str, LODPyramid] = {}
        self._build_gen: int = 0
        self._pending_gen: int = -1
        self._pending_labels: dict[str, str] = {}

        # 子面板：用 DataFrame 管理 E/B 两行的 widget
        self.widget_data: pd.DataFrame = pd.DataFrame(
            data={
                "widget": [None] * 2,
            },
            index=["E", "B"],
        )

        self.splitter: QSplitter = QSplitter(
            Qt.Orientation.Vertical
        )  # 创建纵向分割器
        self.splitter.setChildrenCollapsible(
            False
        )  # 重要：防止图表被拖拽到完全消失

        self.slide_time: QSlider = QSlider(Qt.Orientation.Horizontal)
        self.slide_time.setFixedWidth(300)
        self.slide_time.setDisabled(True)
        self.start_time: QDateTimeEdit = QDateTimeEdit()
        self.start_time.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.start_time.setCalendarPopup(True)
        self.start_time.setTimeSpec(Qt.TimeSpec.UTC)
        self.end_time: QDateTimeEdit = QDateTimeEdit()
        self.end_time.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.end_time.setCalendarPopup(True)
        self.end_time.setTimeSpec(Qt.TimeSpec.UTC)
        self.btn_plot: QPushButton = QPushButton("时间序列跳转")
        self.btn_plot.setDisabled(True)
        self.scroll_x: QScrollBar = QScrollBar(Qt.Orientation.Horizontal)
        self.scroll_x.setDisabled(True)

        self.init_ui()
        self.connect_signals()
        self.init_data()

    def init_ui(self) -> None:
        """构建工具栏（窗口长度滑块、时间跳转控件）和纵向分割器。"""
        plot_panel = QWidget()
        plot_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setSpacing(1)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QToolBar()
        toolbar.addWidget(QLabel("绘制时间长度："))
        toolbar.addWidget(self.slide_time)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("绘制时间范围: \t"))
        toolbar.addWidget(self.start_time)
        toolbar.addWidget(QLabel("  ~  "))
        toolbar.addWidget(self.end_time)
        toolbar.addSeparator()
        toolbar.addWidget(self.btn_plot)

        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(self.splitter)
        plot_layout.addWidget(self.scroll_x)

        self.setLayout(plot_layout)

    def connect_signals(self) -> None:
        self.slide_time.valueChanged.connect(
            self.on_x_view_size_changed
        )  # Mixin
        self.scroll_x.valueChanged.connect(
            self.on_x_view_start_changed
        )  # Mixin
        self.btn_plot.clicked.connect(self.on_btn_plot_clicked)
        self.start_time.dateTimeChanged.connect(self.on_time_changed)  # Mixin
        self.end_time.dateTimeChanged.connect(self.on_time_changed)  # Mixin

    def init_data(self) -> None:
        """初始化数据范围、滑动控件和子图。"""
        self.duration = self.end - self.start

        # 设置时间控件的可选范围
        self.set_time_range(  # Mixin
            pts_to_qdt(self.start),
            pts_to_qdt(self.end),
        )

        # 窗口长度滑块：最小值 1 小时，最大值等于数据全长
        self.slide_time.setMinimum(
            int(pd.Timedelta(hours=1).total_seconds())
        )
        self.slide_time.setMaximum(int(self.duration.total_seconds()))
        self.slide_time.setValue(int(self.x_view_size.total_seconds()))
        self.slide_time.setEnabled(True)

        # X 轴滚动条：范围 = 总长 - 窗口长
        self.scroll_x.setMinimum(0)
        self.scroll_x.setMaximum(
            int((self.duration - self.x_view_size).total_seconds())
        )
        self.scroll_x.setValue(int(self.x_view_start.total_seconds()))
        self.scroll_x.setPageStep(int(self.x_view_size.total_seconds()))
        self.scroll_x.setEnabled(True)

        self._init_widget()
        self._refresh_plots()
        self._on_viewport_changed()

    def _init_widget(self) -> None:
        """初始化 E/B 两行的 BriefPlot。"""
        for row in self.widget_data.index:
            widget: Optional[BriefPlot] = self.widget_data.loc[row, "widget"]
            if widget is None:
                widget = BriefPlot(label=row)
                self.widget_data.loc[row, "widget"] = widget
                self.splitter.addWidget(widget)
            else:
                widget.set_label(row)

    def _refresh_plots(self) -> None:
        """异步构建：逐参考通道建截顶 LODPyramid，完成后在主线程装曲线。

        曲线只接收当前视口适配像素宽度的降采样数据（min/max 包络，尖峰
        不丢）；全量数据仍留在各 EMData 里，金字塔随对比控件一起释放。
        """
        for widget in self.widget_data["widget"].values:
            if widget is not None:
                widget.clear_curves()

        self._build_gen += 1
        gen = self._build_gen
        self._pending_gen = gen
        self._pending_labels = {}

        entries = []
        for em_data, channel_name in self.channel_refs:
            key: str = channel_name
            # 按首字母确定所属子图
            if key.startswith("E"):
                row: str = "E"
            elif key.startswith("B") or key.startswith("H"):
                row = "B"
            else:
                continue
            label: str = f"{em_data.name} - {key}"
            self._pending_labels[label] = row
            s = em_data.get_channel_series(channel_name)
            entries.append((label, dti_to_numpy(s.index), s.to_numpy(dtype=np.float64, copy=True)))

        def task():
            # 去均值以对齐不同通道的基准线，再建金字塔（纯 numpy/Rust，可在池线程执行）
            pyramids: dict[str, LODPyramid] = {}
            for label, x, values in entries:
                mean = safe_nanmean(values)
                y = np.asarray(values - (mean if np.isfinite(mean) else 0.0), dtype=np.float64)
                pyramids[label] = LODPyramid(x, y, max_finest_points=PYRAMID_MAX_FINEST_POINTS)
            return pyramids

        # on_done/on_error 须是 QObject 绑定方法（排队回主线程）；
        # 闭包接收者会退化为 DirectConnection 在池线程碰控件
        run_in_pool(
            "io", task,
            on_done=self._on_display_build_done,
            on_error=self._on_display_build_failed,
            owner=self,
        )

    def _on_display_build_done(self, pyramids: dict) -> None:
        if self._build_gen != self._pending_gen:
            return  # 已被更新的 _refresh_plots 取代，丢弃过期结果
        self._pyramids = pyramids
        start: pd.Timestamp = self.start + self.x_view_start
        end: pd.Timestamp = start + self.x_view_size
        for label, pyramid in self._pyramids.items():
            row = self._pending_labels.get(label)
            if row is None:
                continue
            widget: Optional[BriefPlot] = self.widget_data.loc[row, "widget"]
            if widget is None:
                continue
            xs, ys = pyramid.get(start.timestamp(), end.timestamp(), max(1, widget.width()))
            widget.add_curve(x_data=xs, y_data=ys, label=label, graph_type="plot")
        for key, widget in self.widget_data["widget"].items():
            widget.setVisible(len(widget.curves) > 0)
        self._on_viewport_changed()
        self.display_ready.emit()

    def _on_display_build_failed(self, exc: BaseException) -> None:
        logger.error("对比视图显示数据构建失败: %r", exc, exc_info=exc)

    def _on_viewport_changed(self) -> None:
        """实现 Mixin 钩子：视口变化时替换各曲线为适配像素宽度的 LOD 数据。

        视口更新已在 Mixin 侧 16ms 防抖，直接在主线程查询金字塔 + setData 即可。
        """
        start: pd.Timestamp = self.start + self.x_view_start
        end: pd.Timestamp = start + self.x_view_size
        x_start = start.timestamp()
        x_end = end.timestamp()
        for widget in self.widget_data["widget"].values:
            if widget is None:
                continue
            for label in list(widget.curves.keys()):
                pyramid = self._pyramids.get(label)
                if pyramid is None:
                    continue
                xs, ys = pyramid.get(x_start, x_end, max(1, widget.width()))
                if len(xs) > 0:
                    widget.update_curve(label, xs, ys)
            widget.update_plot(start, end)
        self.set_time(pts_to_qdt(start), pts_to_qdt(end))

    # -------------------------------------------------------------------------
    # 槽函数
    # -------------------------------------------------------------------------

    def on_btn_plot_clicked(self) -> None:
        """跳转到指定时间范围。"""
        start: pd.Timestamp = qdt_to_pts(self.start_time.dateTime())
        end: pd.Timestamp = qdt_to_pts(self.end_time.dateTime())
        start_duration, end_duration = self._clamp_jump(  # Mixin
            start,
            end,
            data_start=self.start,
            data_end=self.end,
            tz=self.start.tz,
        )
        self._apply_jump(start_duration, end_duration)  # Mixin

