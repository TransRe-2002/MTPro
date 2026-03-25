from PySide6.QtWidgets import (
    QWidget, QGridLayout, QToolBar,
    QScrollBar, QSlider, QSizePolicy,
    QLabel, QDateTimeEdit, QPushButton,
    QApplication
)
from PySide6.QtGui import QAction, QIcon
from PySide6.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph import DateAxisItem
import numpy as np
import pandas as pd
from typing import Optional

from utils.time_convert import pts_to_qdt, qdt_to_pts
from utils.series import dti_to_numpy
from base.time_viewport_mixin import TimeViewportMixin


class PlotPanel(TimeViewportMixin, QWidget):

    # -------------------------------------------------------------------------
    # 初始化
    # -------------------------------------------------------------------------

    def __init__(
        self,
        x_data: Optional[pd.DatetimeIndex] = None,
        y_data: Optional[pd.Series] = None,
        label: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)

        if self.parent() is not None:
            self.setWindowFlag(Qt.WindowType.Window)

        self.label = label
        self.x_data = x_data
        self.y_data = y_data
        self.duration: Optional[pd.Timedelta] = (
            (x_data[-1] - x_data[0]) if x_data is not None else None
        )

        # PyQtGraph相关属性
        self.plot_widget = None
        self.plot_item = None
        self.plot_curve = None

        # 视口参数（Mixin 约定属性名为 x_view_start / x_view_size）
        self.x_view_start: Optional[pd.Timedelta] = None
        self.x_view_size: Optional[pd.Timedelta] = None
        self.y_viewport_start: Optional[float] = None
        self.y_viewport_size: Optional[float] = None

        # 信号增益
        self.zoom_ratio = 1.0

        self.init_ui()
        self.connect_signals()
        self.init_plot()

    def init_ui(self):
        toolbar = QToolBar()
        self.zoom_in_action = QAction("放大", self)
        self.zoom_in_action.setIcon(QIcon.fromTheme("zoom-in"))
        self.zoom_in_action.setIconText("放大")
        toolbar.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction("缩小", self)
        self.zoom_out_action.setIcon(QIcon.fromTheme("zoom-out"))
        self.zoom_out_action.setIconText("缩小")
        toolbar.addAction(self.zoom_out_action)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("绘制时间长度："))
        self.slide_time = QSlider(Qt.Orientation.Horizontal)
        self.slide_time.setFixedWidth(200)
        self.slide_time.setEnabled(False)
        toolbar.addWidget(self.slide_time)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("绘制时间范围: \t"))
        self.start_time = QDateTimeEdit()
        self.start_time.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.start_time.setCalendarPopup(True)
        toolbar.addWidget(self.start_time)
        toolbar.addWidget(QLabel("  ~  "))
        self.end_time = QDateTimeEdit()
        self.end_time.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.end_time.setCalendarPopup(True)
        toolbar.addWidget(self.end_time)
        toolbar.addSeparator()

        self.btn_plot = QPushButton("时间序列跳转")
        self.btn_plot.setDisabled(True)
        toolbar.addWidget(self.btn_plot)

        self.plot_widget = pg.PlotWidget(axisItems={'bottom': DateAxisItem()})
        self.plot_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideButtons()
        self.plot_widget.getAxis('left').setStyle(tickLength=0)
        self.plot_widget.getAxis('bottom').setStyle(tickLength=0)
        self.plot_item = self.plot_widget.getPlotItem()
        if self.label is not None:
            self.plot_item.setLabel('left', self.label)

        self.scroll_x = QScrollBar(Qt.Orientation.Horizontal)
        self.scroll_x.setEnabled(False)
        self.scroll_y = QScrollBar(Qt.Orientation.Vertical)
        self.scroll_y.setEnabled(False)
        self.scroll_y.setInvertedControls(True)
        self.scroll_y.setInvertedAppearance(True)

        glay = QGridLayout()
        glay.addWidget(toolbar, 0, 0, 1, 2)
        glay.addWidget(self.plot_widget, 1, 1)
        glay.addWidget(self.scroll_x, 2, 1)
        glay.addWidget(self.scroll_y, 1, 2)
        self.setLayout(glay)

        screen = QApplication.primaryScreen().geometry()
        self.resize(screen.width() // 2, screen.height() // 2)
        self.move(screen.center() - self.rect().center())

    def connect_signals(self):
        self.scroll_x.valueChanged.connect(self.on_x_view_start_changed)  # Mixin
        self.scroll_y.valueChanged.connect(self.update_y_viewport)
        self.slide_time.valueChanged.connect(self.on_x_view_size_changed)  # Mixin
        self.start_time.dateTimeChanged.connect(self.on_time_changed)      # Mixin
        self.end_time.dateTimeChanged.connect(self.on_time_changed)        # Mixin
        self.zoom_in_action.triggered.connect(self.on_zoom_in_triggered)
        self.zoom_out_action.triggered.connect(self.on_zoom_out_triggered)
        self.btn_plot.clicked.connect(self.on_btn_plot_clicked)

    # -------------------------------------------------------------------------
    # 绘图初始化
    # -------------------------------------------------------------------------

    def init_plot(self):
        if self.x_data is None or self.y_data is None:
            self.add_text_placeholder("请先加载数据")
            return

        self.clear_plot()
        self.set_time_range(pts_to_qdt(self.x_data[0]), pts_to_qdt(self.x_data[-1]))  # Mixin

        x_values = dti_to_numpy(self.x_data)
        y_values = self.y_data.to_numpy()
        self.plot_curve = pg.PlotCurveItem(
            x_values, y_values,
            pen=pg.mkPen(color='b', width=1),
            autoDownsample=True,
            downsample=1000,
            downsampleMethod='subsample',
            clipToView=True,
            useOpenGL=True,
            connect='finite'
        )
        self.plot_item.addItem(self.plot_curve)

        grid_pen = pg.mkPen(color=(100, 100, 100, 180), width=1, style=Qt.PenStyle.DashLine)
        self.plot_item.showGrid(x=True, y=True, alpha=0.7)
        self.plot_item.getAxis('bottom').setPen(grid_pen)
        self.plot_item.getAxis('left').setPen(grid_pen)

        # 初始化 x 视口
        self.x_view_size = self.duration
        self.x_view_start = pd.Timedelta(seconds=0)
        self.scroll_x.setEnabled(True)
        self.scroll_x.setMinimum(0)
        self.scroll_x.setMaximum(0)
        self.scroll_x.setPageStep(int(self.x_view_size.total_seconds()))
        self.scroll_x.setValue(0)

        # 初始化 y 视口
        y_min = float(np.nanmin(self.y_data))
        y_max = float(np.nanmax(self.y_data))
        y_center = (y_max + y_min) / 2
        y_diff = (y_max - y_min) / 2
        ymin = y_center - y_diff * 1.05
        ymax = y_center + y_diff * 1.05
        yrange = ymax - ymin
        self.y_viewport_start = ymin
        self.y_viewport_size = yrange * self.zoom_ratio
        self.scroll_y.setEnabled(True)
        self.scroll_y.setMinimum(int(ymin * 10000))
        self.scroll_y.setMaximum(max(int(ymin * 10000), int((ymax - self.y_viewport_size) * 10000)))
        self.scroll_y.setPageStep(int(self.y_viewport_size * 10000))
        self.scroll_y.setValue(int(self.y_viewport_start * 10000))

        # 初始化时间滑块
        self.slide_time.setEnabled(True)
        self.slide_time.setMinimum(int(pd.Timedelta(minutes=30).total_seconds()))
        self.slide_time.setMaximum(max(1, int(self.duration.total_seconds())))
        self.slide_time.setValue(int(self.x_view_size.total_seconds()))

        self.update_display()

    def clear_plot(self):
        """清除现有绘图"""
        if self.plot_curve:
            self.plot_item.removeItem(self.plot_curve)
            self.plot_curve = None

    def add_text_placeholder(self, text):
        """添加文本占位符"""
        self.plot_item.clear()
        text_item = pg.TextItem(text, color='k', anchor=(0.5, 0.5))
        self.plot_item.addItem(text_item)
        text_item.setPos(0, 0)

    # -------------------------------------------------------------------------
    # 视口更新
    # -------------------------------------------------------------------------

    def _on_viewport_changed(self) -> None:
        """实现 Mixin 钩子：视口变化时刷新绘图显示。"""
        self.update_display()

    def update_display(self):
        """更新绘图显示范围。"""
        if self.x_data is None or self.y_data is None:
            return

        x_start = min(self.x_data[0] + self.x_view_start, self.x_data[-1])
        x_end = min(x_start + self.x_view_size, self.x_data[-1])

        y_min = self.y_viewport_start
        y_max = y_min + self.y_viewport_size
        if self.zoom_ratio > 1.0:
            y_center = (y_min + y_max) / 2
            y_diff = (y_max - y_min) / 2
            y_min = y_center - y_diff * self.zoom_ratio
            y_max = y_center + y_diff * self.zoom_ratio

        self.plot_item.setXRange(x_start.timestamp(), x_end.timestamp(), padding=0)
        self.plot_item.setYRange(y_min, y_max, padding=0)

        if self.x_data[-1] > x_end > x_start > self.x_data[0]:
            self.set_time(pts_to_qdt(x_start), pts_to_qdt(x_end))

    def update_y_viewport(self):
        """纵坐标滚动条变化 → 更新 y 视口起点。"""
        self.y_viewport_start = self.scroll_y.value() / 10000.0
        self.update_display()

    def update_y_viewport_size(self):
        """缩放比例变化 → 更新 y 视口大小。"""
        if self.x_data is None:
            return
        y_min = float(np.nanmin(self.y_data))
        y_max = float(np.nanmax(self.y_data))

        if self.zoom_ratio < 1.0:
            yrange = y_max - y_min
            self.y_viewport_size = yrange * self.zoom_ratio
            max_start = y_max - self.y_viewport_size
            self.y_viewport_start = (y_max + y_min - self.y_viewport_size) / 2
            self.scroll_y.setMinimum(int(y_min * 10000))
            self.scroll_y.setMaximum(int(max_start * 10000))
            self.scroll_y.setPageStep(int(self.y_viewport_size * 10000))
            self.scroll_y.setValue(int(self.y_viewport_start * 10000))
            self.scroll_y.setEnabled(True)
        elif self.zoom_ratio == 1.0:
            self.y_viewport_size = y_max - y_min
            self.y_viewport_start = y_min
            self.scroll_y.setMinimum(int(y_min * 10000))
            self.scroll_y.setMaximum(int((y_max - self.y_viewport_size) * 10000))
            self.scroll_y.setValue(int(self.y_viewport_start * 10000))
            self.scroll_y.setPageStep(int(self.y_viewport_size * 10000))
            self.scroll_y.setEnabled(False)
        else:
            self.scroll_y.setEnabled(False)
        self.update_display()

    # -------------------------------------------------------------------------
    # 槽函数
    # -------------------------------------------------------------------------

    def on_zoom_in_triggered(self):
        if 1 / 64.0 < self.zoom_ratio <= 64.0:
            self.zoom_out_action.setEnabled(True)
            self.zoom_ratio /= 2.0
        self.update_y_viewport_size()
        if self.zoom_ratio <= 1 / 64.0:
            self.zoom_in_action.setDisabled(True)

    def on_zoom_out_triggered(self):
        if 1 / 64.0 <= self.zoom_ratio < 64.0:
            self.zoom_in_action.setEnabled(True)
            self.zoom_ratio *= 2.0
        self.update_y_viewport_size()
        if self.zoom_ratio >= 64.0:
            self.zoom_out_action.setDisabled(True)

    def on_btn_plot_clicked(self):
        """跳转到指定时间范围。"""
        start = qdt_to_pts(self.start_time.dateTime())
        end = qdt_to_pts(self.end_time.dateTime())
        start_duration, end_duration = self._clamp_jump(  # Mixin
            start, end,
            data_start=self.x_data[0],
            data_end=self.x_data[-1],
            tz=self.x_data[0].tz,
        )
        self._apply_jump(start_duration, end_duration)  # Mixin

    def closeEvent(self, event):
        if self.parent() is not None:
            event.ignore()
            self.hide()
        else:
            event.accept()

    # -------------------------------------------------------------------------
    # 公开接口
    # -------------------------------------------------------------------------

    def range_change(self, start: pd.Timestamp, end: pd.Timestamp):
        """外部调用：跳转到指定时间范围。"""
        start = max(start, self.x_data[0])
        end = min(end, self.x_data[-1])
        self.x_view_start = start - self.x_data[0]
        self.x_view_size = end - start
        self.set_time(pts_to_qdt(start), pts_to_qdt(end))
        self._sync_slide_time()   # Mixin
        self._sync_scroll_x()     # Mixin
        self.update_display()

    def set_label(self, xlabel, ylabel):
        """设置坐标轴标签。"""
        self.plot_item.setLabel('bottom', xlabel)
        self.plot_item.setLabel('left', ylabel)
