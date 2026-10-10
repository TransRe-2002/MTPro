from __future__ import annotations


import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import DateAxisItem
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDateTimeEdit,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollBar,
    QSizePolicy,
    QSlider,
    QToolBar,
    QWidget,
)

from ui.widgets.time_viewport import TimeViewportMixin
from utils.decimate import LODPyramid
from utils.series import dti_to_numpy
from utils.time_convert import pts_to_qdt, qdt_to_pts


class PlotPanel(TimeViewportMixin, QWidget):
    """时间序列绘图面板，集成LOD金字塔、视口窗口化和双轴滚动控制。"""

    # -------------------------------------------------------------------------
    # 初始化
    # -------------------------------------------------------------------------

    def __init__(
        self,
        x_data: pd.DatetimeIndex | None = None,
        y_data: pd.Series | None = None,
        label: str | None = None,
        parent=None,
    ):
        """
        初始化绘图面板。

        Args:
            x_data: 时间轴数据（DatetimeIndex）
            y_data: 信号幅值数据
            label: 纵轴标签文本
            parent: 父级 QWidget
        """
        super().__init__(parent)

        self._viewport_debounce_init()  # Mixin: 初始化视口更新防抖

        if self.parent() is not None:
            self.setWindowFlag(Qt.WindowType.Window)

        self.label = label
        self.x_data = x_data
        self.y_data = y_data
        self.duration: pd.Timedelta | None = (
            (x_data[-1] - x_data[0])
            if x_data is not None and len(x_data) > 0
            else None
        )

        # PyQtGraph相关属性
        self.plot_widget = None       # pg.PlotWidget 控件
        self.plot_item = None         # plot 的 PlotItem，管理坐标轴/曲线
        self.plot_curve = None        # 当前绘制的 PlotCurveItem

        # LOD 金字塔（视口窗口化用）
        self._x_raw: np.ndarray | None = None    # 原始时间戳数组（秒级浮点）
        self._y_raw: np.ndarray | None = None    # 原始信号幅值数组
        self._pyramid: LODPyramid | None = None  # LOD 缩放金字塔（按像素数查询）
        self._has_plottable_data = False            # 至少存在一个有限幅值

        # 视口参数（Mixin 约定属性名为 x_view_start / x_view_size）
        self.x_view_start: pd.Timedelta | None = None  # 横轴视口起点（相对数据首点偏移）
        self.x_view_size: pd.Timedelta | None = None   # 横轴视口跨度
        self.y_viewport_start: float | None = None     # 纵轴视口下界（幅值）
        self.y_viewport_size: float | None = None      # 纵轴视口跨度（幅值范围）

        # 信号增益（纵轴缩放倍率）
        self.zoom_ratio = 1.0

        self.init_ui()
        self.connect_signals()
        self.init_plot()

    def init_ui(self):
        """构建工具栏、网格布局和坐标轴配置。"""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        self.zoom_in_action = QAction(QIcon.fromTheme("zoom-in"), "放大(Y)", self)
        self.zoom_in_action.setToolTip("纵轴放大（减小显示幅值范围）")
        toolbar.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction(QIcon.fromTheme("zoom-out"), "缩小(Y)", self)
        self.zoom_out_action.setToolTip("纵轴缩小（增大显示幅值范围）")
        toolbar.addAction(self.zoom_out_action)

        toolbar.addSeparator()

        win_label = QLabel("  窗口:", toolbar)
        win_label.setFont(QFont("Arial", 9))
        toolbar.addWidget(win_label)
        self.slide_time = QSlider(Qt.Orientation.Horizontal)
        self.slide_time.setFixedWidth(160)
        self.slide_time.setEnabled(False)
        self.slide_time.setToolTip("拖动调整时间窗口长度")
        toolbar.addWidget(self.slide_time)

        toolbar.addSeparator()

        start_label = QLabel("  起始:", toolbar)
        start_label.setFont(QFont("Arial", 9))
        toolbar.addWidget(start_label)
        self.start_time = QDateTimeEdit()
        self.start_time.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.start_time.setCalendarPopup(True)
        self.start_time.setTimeSpec(Qt.TimeSpec.UTC)
        toolbar.addWidget(self.start_time)

        end_label = QLabel("  结束:", toolbar)
        end_label.setFont(QFont("Arial", 9))
        toolbar.addWidget(end_label)
        self.end_time = QDateTimeEdit()
        self.end_time.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.end_time.setCalendarPopup(True)
        self.end_time.setTimeSpec(Qt.TimeSpec.UTC)
        toolbar.addWidget(self.end_time)

        toolbar.addSeparator()
        self.btn_plot = QPushButton("跳转")
        self.btn_plot.setToolTip("跳转到指定的起始/结束时间范围")
        self.btn_plot.setFixedHeight(24)
        self.btn_plot.setDisabled(True)
        toolbar.addWidget(self.btn_plot)

        # 创建PlotWidget，横轴使用DateAxisItem支持日期刻度
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': DateAxisItem(utcOffset=0)})
        self.plot_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.plot_widget.setMouseEnabled(x=False, y=False)  # 禁用鼠标拖拽，使用滚动条控制
        self.plot_widget.hideButtons()
        self.plot_widget.getAxis('left').setStyle(tickLength=0)
        self.plot_widget.getAxis('bottom').setStyle(tickLength=0)
        self.plot_item = self.plot_widget.getPlotItem()
        if self.label is not None:
            self.plot_item.setLabel('left', self.label)

        # 横轴滚动条：控制时间窗口起点
        self.scroll_x = QScrollBar(Qt.Orientation.Horizontal)
        self.scroll_x.setEnabled(False)
        # 纵轴滚动条：控制幅值窗口起点
        self.scroll_y = QScrollBar(Qt.Orientation.Vertical)
        self.scroll_y.setEnabled(False)
        self.scroll_y.setInvertedControls(True)   # 向上拖动 = 增大值
        self.scroll_y.setInvertedAppearance(True)  # 滑块视觉上也倒置

        # 四宫格布局：工具栏(0,0跨度2列) | 水平分隔线 | 绘图(2,1) + 横向滚动条(3,1) + 纵向滚动条(2,2)
        glay = QGridLayout()
        glay.setContentsMargins(4, 4, 4, 4)
        glay.setSpacing(2)
        glay.addWidget(toolbar, 0, 0, 1, 2)

        # 水平分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        glay.addWidget(sep, 1, 0, 1, 2)

        glay.addWidget(self.plot_widget, 2, 1)
        glay.addWidget(self.scroll_x, 3, 1)
        glay.addWidget(self.scroll_y, 2, 2)
        self.setLayout(glay)

        # 窗口居中，默认大小为屏幕一半
        screen = QApplication.primaryScreen().geometry()
        self.resize(screen.width() // 2, screen.height() // 2)
        self.move(screen.center() - self.rect().center())

    def connect_signals(self):
        """连接所有控件信号到对应的槽函数。"""
        self.scroll_x.valueChanged.connect(self.on_x_view_start_changed)  # Mixin
        self.scroll_x.sliderPressed.connect(self._on_scroll_drag_start)
        self.scroll_x.sliderReleased.connect(self._on_scroll_drag_end)
        self.scroll_y.valueChanged.connect(self.update_y_viewport)
        self.scroll_y.sliderPressed.connect(self._on_scroll_drag_start)
        self.scroll_y.sliderReleased.connect(self._on_scroll_drag_end)
        self.slide_time.valueChanged.connect(self.on_x_view_size_changed)  # Mixin
        self.slide_time.sliderPressed.connect(self._on_scroll_drag_start)
        self.slide_time.sliderReleased.connect(self._on_scroll_drag_end)
        self.start_time.dateTimeChanged.connect(self.on_time_changed)      # Mixin
        self.end_time.dateTimeChanged.connect(self.on_time_changed)        # Mixin
        self.zoom_in_action.triggered.connect(self.on_zoom_in_triggered)
        self.zoom_out_action.triggered.connect(self.on_zoom_out_triggered)
        self.btn_plot.clicked.connect(self.on_btn_plot_clicked)

    # -------------------------------------------------------------------------
    # 绘图初始化
    # -------------------------------------------------------------------------

    def init_plot(self):
        """创建LOD金字塔、初始化视口、设置滚动条范围并触发首次显示。"""
        if (
            self.x_data is None
            or self.y_data is None
            or len(self.x_data) == 0
            or len(self.y_data) == 0
        ):
            self.add_text_placeholder("请先加载数据")
            return

        self.clear_plot()
        x_values = dti_to_numpy(self.x_data)
        y_values = self.y_data.to_numpy(dtype=np.float64)
        finite_values = y_values[np.isfinite(y_values)]
        if finite_values.size == 0:
            self.add_text_placeholder("该通道没有有效数据")
            self.zoom_in_action.setEnabled(False)
            self.zoom_out_action.setEnabled(False)
            return

        self._has_plottable_data = True
        self.set_time_range(pts_to_qdt(self.x_data[0]), pts_to_qdt(self.x_data[-1]))  # Mixin
        self._x_raw = x_values
        self._y_raw = y_values
        # 构建LOD金字塔：将全量数据预抽稀为多级LOD，后续按像素数查询对应分辨率的数据
        self._pyramid = LODPyramid(x_values, y_values)
        self.plot_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color='b', width=1),
            autoDownsample=True,
            downsampleMethod='peak',
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
        y_min = float(finite_values.min())
        y_max = float(finite_values.max())
        y_center = (y_max + y_min) / 2
        y_diff = (y_max - y_min) / 2
        if y_diff == 0.0:
            # 常量（包括全 0）通道也需要一个非零显示范围。
            y_diff = max(abs(y_center) * 0.05, 1.0)
        ymin = y_center - y_diff * 1.05
        ymax = y_center + y_diff * 1.05
        yrange = ymax - ymin
        self.y_viewport_start = ymin
        self.y_viewport_size = yrange * self.zoom_ratio
        self.scroll_y.setEnabled(True)
        # *10000：将浮点幅值映射为int，保留4位小数精度，满足QScrollBar的int范围限制
        self.scroll_y.setMinimum(int(ymin * 10000))
        self.scroll_y.setMaximum(max(int(ymin * 10000), int((ymax - self.y_viewport_size) * 10000)))
        self.scroll_y.setPageStep(int(self.y_viewport_size * 10000))
        self.scroll_y.setValue(int(self.y_viewport_start * 10000))

        # 初始化时间滑块（窗口宽度控制）
        self.slide_time.setEnabled(True)
        self.slide_time.setMinimum(int(pd.Timedelta(minutes=30).total_seconds()))
        self.slide_time.setMaximum(max(1, int(self.duration.total_seconds())))
        self.slide_time.setValue(int(self.x_view_size.total_seconds()))

        self.update_display()

    def clear_plot(self):
        """清除现有绘图（移除PlotCurveItem）。"""
        if self.plot_curve:
            self.plot_item.removeItem(self.plot_curve)
            self.plot_curve = None

    def add_text_placeholder(self, text):
        """添加文本占位符（无数据时显示提示文字）。"""
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
        """更新绘图显示范围：LOD金字塔查询 + setXRange + setYRange。"""
        if not self._has_plottable_data:
            return

        x_start = min(self.x_data[0] + self.x_view_start, self.x_data[-1])
        x_end = min(x_start + self.x_view_size, self.x_data[-1])
        x_start_ts = x_start.timestamp()
        x_end_ts = x_end.timestamp()

        # 视口窗口化：按当前PlotWidget像素宽度从金字塔取出适配分辨率的LOD数据
        if self._pyramid is not None and self._x_raw is not None:
            pw = max(self.plot_widget.width(), 200)         # 像素宽度，最小200防止零宽
            lx, ly = self._pyramid.get(x_start_ts, x_end_ts, pw)  # LOD查询：返回抽稀后的xy
            self.plot_curve.setData(lx, ly)                 # 仅传入可见范围数据，减少GPU/CPU负载

        y_min = self.y_viewport_start
        y_max = y_min + self.y_viewport_size
        # zoom_ratio > 1 时以视口中心为锚点向外扩展幅值范围
        if self.zoom_ratio > 1.0:
            y_center = (y_min + y_max) / 2
            y_diff = (y_max - y_min) / 2
            y_min = y_center - y_diff * self.zoom_ratio
            y_max = y_center + y_diff * self.zoom_ratio

        self.plot_item.setXRange(x_start_ts, x_end_ts, padding=0)
        self.plot_item.setYRange(y_min, y_max, padding=0)

        if self.x_data[-1] > x_end > x_start > self.x_data[0]:
            self.set_time(pts_to_qdt(x_start), pts_to_qdt(x_end))

    def update_y_viewport(self):
        """纵坐标滚动条变化 → 更新 y 视口起点（经防抖）。"""
        # /10000.0：将scrollbar int值还原为浮点幅值
        self.y_viewport_start = self.scroll_y.value() / 10000.0
        self._viewport_debounce.start()

    def update_y_viewport_size(self):
        """缩放比例变化 → 同步更新 y 视口大小和滚动条参数。"""
        if not self._has_plottable_data:
            return
        finite_values = self._y_raw[np.isfinite(self._y_raw)]
        y_min = float(finite_values.min())
        y_max = float(finite_values.max())
        if y_max == y_min:
            padding = max(abs(y_min) * 0.05, 1.0)
            y_min -= padding
            y_max += padding

        if self.zoom_ratio < 1.0:
            # 放大状态：视口小于数据范围，启用滚动条
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
            # 原始大小：视口恰好覆盖全数据范围，滚动条禁用
            self.y_viewport_size = y_max - y_min
            self.y_viewport_start = y_min
            self.scroll_y.setMinimum(int(y_min * 10000))
            self.scroll_y.setMaximum(int((y_max - self.y_viewport_size) * 10000))
            self.scroll_y.setValue(int(self.y_viewport_start * 10000))
            self.scroll_y.setPageStep(int(self.y_viewport_size * 10000))
            self.scroll_y.setEnabled(False)
        else:
            # 缩小状态：视口大于数据范围，通过update_display以中心锚点扩展
            self.scroll_y.setEnabled(False)
        self.update_display()

    # -------------------------------------------------------------------------
    # 槽函数
    # -------------------------------------------------------------------------

    def _on_scroll_drag_start(self) -> None:
        """滚动条开始拖动：切换去抖间隔为50ms（20fps）以降低渲染负载。"""
        if self._viewport_debounce is not None:
            self._viewport_debounce.setInterval(50)  # 20fps，拖动中降低重绘频率

    def _on_scroll_drag_end(self) -> None:
        """滚动条拖动结束：恢复16ms去抖间隔（60fps）并立即刷新。"""
        if self._viewport_debounce is not None:
            self._viewport_debounce.setInterval(16)  # 恢复 60fps
            self._viewport_debounce.stop()
        self.update_display()

    def on_zoom_in_triggered(self):
        """放大(Y)触发：zoom_ratio减半，钳位在[1/64, 64]范围内。"""
        if 1 / 64.0 < self.zoom_ratio <= 64.0:
            self.zoom_out_action.setEnabled(True)
            self.zoom_ratio /= 2.0
        self.update_y_viewport_size()
        # 到达下限则禁用放大按钮，防止无限缩小视口导致数据不可见
        if self.zoom_ratio <= 1 / 64.0:
            self.zoom_in_action.setDisabled(True)

    def on_zoom_out_triggered(self):
        """缩小(Y)触发：zoom_ratio翻倍，钳位在[1/64, 64]范围内。"""
        if 1 / 64.0 <= self.zoom_ratio < 64.0:
            self.zoom_in_action.setEnabled(True)
            self.zoom_ratio *= 2.0
        self.update_y_viewport_size()
        # 到达上限则禁用缩小按钮
        if self.zoom_ratio >= 64.0:
            self.zoom_out_action.setDisabled(True)

    def on_btn_plot_clicked(self):
        """跳转到指定时间范围：从DateTimeEdit读取起止时间并应用。"""
        start = qdt_to_pts(self.start_time.dateTime())
        end = qdt_to_pts(self.end_time.dateTime())
        start_duration, end_duration = self._clamp_jump(  # Mixin: 钳位到数据范围内
            start, end,
            data_start=self.x_data[0],
            data_end=self.x_data[-1],
            tz=self.x_data[0].tz,
        )
        self._apply_jump(start_duration, end_duration)  # Mixin

    def closeEvent(self, event):
        """关闭事件：有父窗口时隐藏而非销毁，无父窗口时正常关闭。"""
        if self.parent() is not None:
            event.ignore()
            self.hide()
        else:
            event.accept()

    # -------------------------------------------------------------------------
    # 公开接口
    # -------------------------------------------------------------------------

    def range_change(self, start: pd.Timestamp, end: pd.Timestamp):
        """外部调用：跳转到指定时间范围（用于从其他组件联动跳转）。"""
        start = max(start, self.x_data[0])        # 钳位到数据首点
        end = min(end, self.x_data[-1])           # 钳位到数据末点
        self.x_view_start = start - self.x_data[0]
        self.x_view_size = end - start
        self.set_time(pts_to_qdt(start), pts_to_qdt(end))
        self._sync_slide_time()   # Mixin: 同步时间滑块到新视口
        self._sync_scroll_x()     # Mixin: 同步横轴滚动条到新视口
        self.update_display()

    def set_label(self, xlabel, ylabel):
        """设置坐标轴标签。"""
        self.plot_item.setLabel('bottom', xlabel)
        self.plot_item.setLabel('left', ylabel)
