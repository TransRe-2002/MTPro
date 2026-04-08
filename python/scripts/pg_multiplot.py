import os

os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QToolBar,
    QScrollBar, QSlider, QSizePolicy,
    QLabel, QDateTimeEdit, QPushButton,
    QMessageBox
)
from PySide6.QtGui import QAction, QIcon
from PySide6.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph import DateAxisItem
import numpy as np
import pandas as pd

from type_convert import pts_to_qdt, qdt_to_pts

# 设置PyQtGraph样式
pg.setConfigOptions(antialias=True)  # 抗锯齿
pg.setConfigOption('background', 'w')  # 背景白色
pg.setConfigOption('foreground', 'k')  # 前景黑色


class PlotPanel(QWidget):
    def __init__(
            self,
            panel_type: str = 'plot',
            label: str | None = None,
            parent=None,
    ):
        super().__init__(parent)

        if self.parent() is not None:
            self.setWindowFlag(Qt.WindowType.Window)

        self.label = label
        self.panel_type = panel_type \
            if panel_type in ['plot', 'bar'] \
            else 'plot'

        # 多曲线管理相关属性
        self.curves = []  # 存储曲线对象
        self.curve_data = []  # 存储曲线数据
        self.base_x_data = None  # 用于控件的基础x数据
        self.duration: pd.Timedelta | float | int | None = None

        # PyQtGraph相关属性
        self.plot_widget = None
        self.plot_item = None

        # 视口参数
        self.x_viewport_start: pd.Timestamp | float | int | None = None
        self.x_viewport_size: pd.Timedelta | float | int | None = None
        self.y_viewport_start: float | int | None = None
        self.y_viewport_size: float | int | None = None

        # 信号增益
        self.zoom_ratio = 1.0

        # 初始化UI
        self.init_ui()
        self.connect_signals()
        # 不再在init中初始化绘图，等待add_curve添加数据

    def init_ui(self):
        # 工具栏
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

        self.slide_time = QSlider(Qt.Horizontal)
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

        # 创建PyQtGraph绘图部件
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        # 禁用PyQtGraph的默认交互（使用自定义控制）
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideButtons()  # 隐藏默认的缩放按钮
        self.plot_widget.getAxis('left').setStyle(tickLength=0)
        self.plot_widget.getAxis('bottom').setStyle(tickLength=0)
        self.plot_item = self.plot_widget.getPlotItem()

        # 添加图例
        self.plot_item.addLegend()

        if self.label is not None:
            self.plot_item.setLabel('left', self.label)

        # 滚动条
        self.scroll_x = QScrollBar(Qt.Horizontal)
        self.scroll_x.setEnabled(False)
        self.scroll_y = QScrollBar(Qt.Vertical)
        self.scroll_y.setEnabled(False)
        # 垂直滚动条倒转
        self.scroll_y.setInvertedControls(True)
        self.scroll_y.setInvertedAppearance(True)

        # 布局
        glay = QGridLayout()
        glay.addWidget(toolbar, 0, 0, 1, 2)
        glay.addWidget(self.plot_widget, 1, 1)
        glay.addWidget(self.scroll_x, 2, 1)
        glay.addWidget(self.scroll_y, 1, 2)
        self.setLayout(glay)

    def connect_signals(self):
        # 滚动条信号 - 直接连接，不使用节流器
        self.scroll_x.valueChanged.connect(self.update_x_viewport)
        self.scroll_y.valueChanged.connect(self.update_y_viewport)
        self.slide_time.valueChanged.connect(self.update_x_viewport_size)

        # 时间变化信号
        self.start_time.dateTimeChanged.connect(self.on_time_changed)
        self.end_time.dateTimeChanged.connect(self.on_time_changed)

        # 缩放信号
        self.zoom_in_action.triggered.connect(self.on_zoom_in_triggered)
        self.zoom_out_action.triggered.connect(self.on_zoom_out_triggered)

        # 按钮信号
        self.btn_plot.clicked.connect(self.on_btn_plot_clicked)

    def add_curve(self, x_data, y_data, label=None, color=None, pen=None):
        """
        添加一条曲线到图表中

        参数:
            x_data: x轴数据，可以是pd.DatetimeIndex、pd.Series、np.ndarray或list
            y_data: y轴数据，可以是pd.Series、np.ndarray或list
            label: 曲线标签（用于图例显示）
            color: 曲线颜色，如'r'、'g'、'b'或十六进制颜色代码
            pen: 直接指定pen对象，如果提供则忽略color参数
        """
        if x_data is None or y_data is None or len(x_data) == 0 or len(y_data) == 0:
            self.show_warning("数据为空，无法添加曲线")
            return None

        # 检查数据长度是否一致
        if len(x_data) != len(y_data):
            self.show_warning(f"x_data长度({len(x_data)})与y_data长度({len(y_data)})不一致")
            return None

        # 转换数据格式
        if isinstance(x_data, pd.DatetimeIndex) or (len(x_data) > 0 and isinstance(x_data[0], pd.Timestamp)):
            x_values = np.array([x_data[i].timestamp() for i in range(len(x_data))])
            is_time_data = True
        else:
            x_values = np.array(x_data)
            is_time_data = False

        if isinstance(y_data, pd.Series):
            y_values = y_data.to_numpy()
        else:
            y_values = np.array(y_data)

        # 如果是第一条曲线，设置基础数据和初始化控件
        if self.base_x_data is None:
            self.base_x_data = x_data
            self.duration = (x_data[len(x_data) - 1] - x_data[0]) if len(x_data) > 1 else pd.Timedelta(0)
            self._init_controls(is_time_data)

        # 创建曲线
        if self.panel_type == 'plot':
            # 设置曲线样式
            if pen is None:
                if color is None:
                    # 默认颜色循环
                    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                    color_idx = len(self.curves) % len(colors)
                    color = colors[color_idx]
                pen = pg.mkPen(color=color, width=1)

            # 生成默认标签
            if label is None:
                label = f"曲线{len(self.curves) + 1}"

            # 创建曲线项
            curve = pg.PlotCurveItem(
                x_values, y_values,
                pen=pen,
                name=label,  # 用于图例显示
                autoDownsample=True,
                downsample=1000,
                downsampleMethod='subsample',
                clipToView=True,
                useOpenGL=True,
                connect='finite'
            )

            self.plot_item.addItem(curve)
            self.curves.append(curve)

            # 存储曲线数据
            self.curve_data.append({
                'x_data': x_data,
                'y_data': y_data,
                'x_values': x_values,
                'y_values': y_values,
                'label': label,
                'pen': pen
            })

            # 更新y轴范围
            self._update_y_range()

            # 如果是时间数据且需要设置时间轴
            if is_time_data and isinstance(self.base_x_data, (pd.DatetimeIndex, pd.Series)) and \
                    not isinstance(self.plot_widget.getAxis('bottom'), DateAxisItem):
                # 移除现有plot_widget，重新创建带日期轴的
                self.layout().removeWidget(self.plot_widget)
                self.plot_widget.deleteLater()
                self.plot_widget = pg.PlotWidget(axisItems={'bottom': DateAxisItem()})
                self.plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                self.plot_widget.setMouseEnabled(x=False, y=False)
                self.plot_widget.hideButtons()
                self.plot_widget.getAxis('left').setStyle(tickLength=0)
                self.plot_widget.getAxis('bottom').setStyle(tickLength=0)
                self.plot_item = self.plot_widget.getPlotItem()
                self.plot_item.addLegend()

                # 重新添加所有曲线
                for curve_info in self.curve_data:
                    new_curve = pg.PlotCurveItem(
                        curve_info['x_values'], curve_info['y_values'],
                        pen=curve_info['pen'],
                        name=curve_info['label'],
                        autoDownsample=True,
                        downsample=1000,
                        downsampleMethod='subsample',
                        clipToView=True,
                        useOpenGL=True,
                        connect='finite'
                    )
                    self.plot_item.addItem(new_curve)

                # 更新布局
                self.layout().addWidget(self.plot_widget, 1, 1)

                # 添加网格
                grid_pen = pg.mkPen(
                    color=pg.mkColor(100, 100, 100),
                    width=1,
                    alpha=1,
                    antialiased=True,
                    style=Qt.DashLine
                )
                self.plot_item.showGrid(x=True, y=True, alpha=0.7)
                self.plot_item.getAxis('bottom').setPen(grid_pen)
                self.plot_item.getAxis('left').setPen(grid_pen)

            # 更新显示
            self.update_display()

            return curve
        else:
            # 条形图处理（保持原有逻辑，只支持单条曲线）
            if len(self.curves) > 0:
                self.show_warning("条形图模式只支持单条曲线")
                return None

            # 原有条形图逻辑...
            # 这里省略，与原有代码相同
            pass

    def _init_controls(self, is_time_data=False):
        """初始化控件（内部方法）"""
        if self.base_x_data is None:
            return

        # 设置时间控件
        if is_time_data and isinstance(self.base_x_data, (pd.DatetimeIndex, pd.Series)):
            self.set_time_range(
                pts_to_qdt(self.base_x_data[0]),
                pts_to_qdt(self.base_x_data[len(self.base_x_data) - 1])
            )

        # 初始化视口参数
        if is_time_data:
            self.x_viewport_size = self.duration
            self.x_viewport_start = pd.Timedelta(seconds=0)
        else:
            self.x_viewport_size = len(self.base_x_data)
            self.x_viewport_start = 0

        # 设置滚动条和滑块
        self.scroll_x.setEnabled(True)
        self.scroll_x.setMinimum(0)

        if is_time_data:
            self.scroll_x.setMaximum(
                max(0, int((self.duration - self.x_viewport_size).total_seconds()))
            )
            self.scroll_x.setPageStep(int(self.x_viewport_size.total_seconds()))
            self.scroll_x.setValue(int(self.x_viewport_start.total_seconds()))

            self.slide_time.setEnabled(True)
            min_half_hour = pd.Timedelta(minutes=30)
            self.slide_time.setMinimum(int(min_half_hour.total_seconds()))
            self.slide_time.setMaximum(max(1, int(self.duration.total_seconds())))
            self.slide_time.setValue(int(self.x_viewport_size.total_seconds()))
        else:
            self.scroll_x.setMaximum(max(0, len(self.base_x_data) - int(self.x_viewport_size)))
            self.scroll_x.setPageStep(int(self.x_viewport_size))
            self.scroll_x.setValue(int(self.x_viewport_start))

            self.slide_time.setEnabled(True)
            self.slide_time.setMinimum(1)
            self.slide_time.setMaximum(max(1, len(self.base_x_data)))
            self.slide_time.setValue(int(self.x_viewport_size))

    def _update_y_range(self):
        """更新y轴范围（内部方法）"""
        if not self.curve_data:
            return

        # 计算所有曲线的y值范围
        all_y_values = []
        for curve_info in self.curve_data:
            all_y_values.append(curve_info['y_values'])

        y_all = np.concatenate(all_y_values) if len(all_y_values) > 1 else all_y_values[0]
        y_min = np.nanmin(y_all)
        y_max = np.nanmax(y_all)

        # 计算y轴范围
        y_center = (y_max + y_min) / 2
        y_diff = (y_max - y_min) / 2
        ymin = y_center - y_diff * 1.05
        ymax = y_center + y_diff * 1.05
        yrange = ymax - ymin
        self.y_viewport_start = ymin
        self.y_viewport_size = yrange * self.zoom_ratio

        # 更新y轴滚动条
        self.scroll_y.setEnabled(True)
        self.scroll_y.setMinimum(int(ymin * 10000))
        self.scroll_y.setMaximum(max(int(ymin * 10000), int((ymax - self.y_viewport_size) * 10000)))
        self.scroll_y.setPageStep(int(self.y_viewport_size * 10000))
        self.scroll_y.setValue(int(self.y_viewport_start * 10000))

    def clear_plot(self):
        """清除所有曲线"""
        for curve in self.curves:
            self.plot_item.removeItem(curve)
        self.curves.clear()
        self.curve_data.clear()
        self.base_x_data = None

        # 重置控件状态
        self.scroll_x.setEnabled(False)
        self.scroll_y.setEnabled(False)
        self.slide_time.setEnabled(False)

        # 清除时间显示
        self.start_time.clear()
        self.end_time.clear()

        # 添加文本占位符
        self.add_text_placeholder("请先加载数据")

    def add_text_placeholder(self, text):
        """添加文本占位符"""
        self.plot_item.clear()
        text_item = pg.TextItem(text, color='k', anchor=(0.5, 0.5))
        self.plot_item.addItem(text_item)
        text_item.setPos(0, 0)

    def set_time(self, start, end):
        """设置时间显示"""
        self.start_time.blockSignals(True)
        self.end_time.blockSignals(True)
        self.start_time.setDateTime(start)
        self.end_time.setDateTime(end)
        self.start_time.blockSignals(False)
        self.end_time.blockSignals(False)

    def set_time_range(self, start, end):
        """设置时间范围限制"""
        self.start_time.blockSignals(True)
        self.end_time.blockSignals(True)
        self.start_time.setDateTime(start)
        self.start_time.setMinimumDateTime(start)
        self.end_time.setMinimumDateTime(start)
        self.end_time.setMaximumDateTime(end)
        self.start_time.blockSignals(False)
        self.end_time.blockSignals(False)
        self.set_time(start, end)

    def set_label(self, xlabel, ylabel):
        """设置坐标轴标签"""
        self.plot_item.setLabel('bottom', xlabel)
        self.plot_item.setLabel('left', ylabel)

    def on_zoom_in_triggered(self):
        """放大信号"""
        if 1 / 64.0 < self.zoom_ratio <= 64.0:
            self.zoom_out_action.setEnabled(True)
            self.zoom_ratio /= 2.0
        self.update_y_viewport_size()
        if self.zoom_ratio <= 1 / 64.0:
            self.zoom_in_action.setDisabled(True)

    def on_zoom_out_triggered(self):
        """缩小信号"""
        if 1 / 64.0 <= self.zoom_ratio < 64.0:
            self.zoom_in_action.setEnabled(True)
            self.zoom_ratio *= 2.0
        self.update_y_viewport_size()
        if self.zoom_ratio >= 64.0:
            self.zoom_out_action.setDisabled(True)

    def on_time_changed(self):
        """时间控件改变时启用跳转按钮"""
        if self.base_x_data is not None:
            self.btn_plot.setEnabled(True)

    def update_x_viewport(self):
        """更新横坐标视口"""
        if self.base_x_data is None:
            return

        if isinstance(self.base_x_data, (pd.DatetimeIndex, pd.Series)):
            self.x_viewport_start = pd.Timedelta(seconds=self.scroll_x.value())
        else:
            self.x_viewport_start = self.scroll_x.value()

        self.update_display()

    def update_y_viewport(self):
        """更新纵坐标视口"""
        self.y_viewport_start = self.scroll_y.value() / 10000.0
        self.update_display()

    def update_x_viewport_size(self):
        """更新横坐标视口大小"""
        if self.base_x_data is None:
            return

        if isinstance(self.base_x_data, (pd.DatetimeIndex, pd.Series)):
            self.x_viewport_size = pd.Timedelta(seconds=self.slide_time.value())
            max_start = self.duration - self.x_viewport_size

            self.scroll_x.setMaximum(int(max_start.total_seconds()))
            self.scroll_x.setPageStep(int(self.x_viewport_size.total_seconds()))

            if self.x_viewport_start > max_start:
                self.x_viewport_start = max_start
                self.scroll_x.setValue(int(self.x_viewport_start.total_seconds()))
        else:
            self.x_viewport_size = self.slide_time.value()
            max_start = len(self.base_x_data) - self.x_viewport_size

            self.scroll_x.setMaximum(max(0, int(max_start)))
            self.scroll_x.setPageStep(int(self.x_viewport_size))

            if self.x_viewport_start > max_start:
                self.x_viewport_start = max_start
                self.scroll_x.setValue(int(self.x_viewport_start))

        self.update_display()

    def update_y_viewport_size(self):
        """更新纵坐标视口大小"""
        if not self.curve_data:
            return

        # 计算所有曲线的y值范围
        all_y_values = []
        for curve_info in self.curve_data:
            all_y_values.append(curve_info['y_values'])

        y_all = np.concatenate(all_y_values) if len(all_y_values) > 1 else all_y_values[0]
        y_min = np.nanmin(y_all)
        y_max = np.nanmax(y_all)

        if self.zoom_ratio < 1.0:
            yrange = y_max - y_min
            self.y_viewport_size = yrange * self.zoom_ratio
            max_start = y_max - self.y_viewport_size
            center = (y_max + y_min - self.y_viewport_size) / 2
            self.y_viewport_start = center

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

    def update_display(self):
        """更新显示"""
        if not self.curve_data:
            return

        # 计算显示范围
        if isinstance(self.base_x_data, (pd.DatetimeIndex, pd.Series)):
            x_start = min(
                self.base_x_data[0] + self.x_viewport_start,
                self.base_x_data[len(self.base_x_data) - 1]
            )
            x_end = min(
                x_start + self.x_viewport_size,
                self.base_x_data[len(self.base_x_data) - 1]
            )

            # 设置视图范围
            self.plot_item.setXRange(x_start.timestamp(), x_end.timestamp(), padding=0)

            # 同步时间控件
            if self.base_x_data[len(self.base_x_data) - 1] > x_end > x_start > self.base_x_data[0]:
                self.set_time(
                    pts_to_qdt(x_start),
                    pts_to_qdt(x_end)
                )
        else:
            x_start = self.x_viewport_start
            x_end = min(x_start + self.x_viewport_size, len(self.base_x_data) - 1)
            self.plot_item.setXRange(x_start, x_end, padding=0)

        # 设置y轴范围
        y_min = self.y_viewport_start
        y_max = y_min + self.y_viewport_size

        if self.zoom_ratio > 1.0:
            y_center = (y_min + y_max) / 2
            y_diff = (y_max - y_min) / 2
            y_min = y_center - y_diff * self.zoom_ratio
            y_max = y_center + y_diff * self.zoom_ratio

        self.plot_item.setYRange(y_min, y_max, padding=0)

    def show_warning(self, message):
        """显示警告消息"""
        msg = QMessageBox(self)
        msg.setWindowModality(Qt.NonModal)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("警告")
        msg.setText(message)
        msg.show()
        # 使用单次定时器关闭消息框
        from PySide6.QtCore import QTimer
        QTimer.singleShot(3000, msg.close)

    def on_btn_plot_clicked(self):
        """跳转到指定时间范围"""
        if self.base_x_data is None or not isinstance(self.base_x_data, (pd.DatetimeIndex, pd.Series)):
            return

        start = qdt_to_pts(self.start_time.dateTime())
        end = qdt_to_pts(self.end_time.dateTime())
        start = start.tz_convert(self.base_x_data[0].tz)
        end = end.tz_convert(self.base_x_data[0].tz)

        if start > end:
            self.show_warning("起始时间大于结束时间，已自动调换...")
            start, end = end, start

        data_start = self.base_x_data[0]
        data_end = self.base_x_data[len(self.base_x_data) - 1]

        if start < data_start:
            start = data_start
            self.start_time.setDateTime(pts_to_qdt(start))
        if end > data_end:
            end = data_end
            self.end_time.setDateTime(pts_to_qdt(end))

        start_duration = start - data_start
        end_duration = end - data_start

        # 边界限幅
        start_duration = max(
            pd.Timedelta(seconds=0),
            min(start_duration, self.duration)
        )
        end_duration = max(
            pd.Timedelta(seconds=0),
            min(end_duration, self.duration)
        )

        # 确保至少包含一分钟样本（并让 end_idx > start_idx）
        if end_duration <= start_duration:
            end_duration = min(
                start_duration + pd.Timedelta(minutes=1),
                self.duration
            )

        self.x_viewport_start = start_duration
        self.x_viewport_size = end_duration - start_duration

        self.slide_time.blockSignals(True)
        self.slide_time.setValue(int(self.x_viewport_size.total_seconds()))
        self.slide_time.blockSignals(False)

        self.scroll_x.blockSignals(True)
        self.scroll_x.setMaximum(
            max(0, int((self.duration - self.x_viewport_size).total_seconds()))
        )
        self.scroll_x.setValue(int(self.x_viewport_start.total_seconds()))
        self.scroll_x.setPageStep(int(self.x_viewport_size.total_seconds()))
        self.scroll_x.blockSignals(False)

        self.update_display()

        start_dt = self.base_x_data[0] + self.x_viewport_start
        end_dt = self.base_x_data[0] + self.x_viewport_start + self.x_viewport_size
        self.set_time(pts_to_qdt(start_dt), pts_to_qdt(end_dt))

        self.btn_plot.setDisabled(True)

    def closeEvent(self, event):
        if self.parent() is not None:
            event.ignore()
            self.hide()
        else:
            event.accept()

    def range_change(self, start: pd.Timestamp, end: pd.Timestamp):
        if self.base_x_data is None or not isinstance(self.base_x_data, (pd.DatetimeIndex, pd.Series)):
            return

        if start < self.base_x_data[0]:
            start = self.base_x_data[0]
        if end > self.base_x_data[len(self.base_x_data) - 1]:
            end = self.base_x_data[len(self.base_x_data) - 1]

        self.x_viewport_start = start - self.base_x_data[0]
        self.x_viewport_size = end - start
        self.set_time(pts_to_qdt(start), pts_to_qdt(end))

        self.slide_time.blockSignals(True)
        self.slide_time.setValue(int(self.x_viewport_size.total_seconds()))
        self.slide_time.blockSignals(False)

        self.scroll_x.blockSignals(True)
        self.scroll_x.setMaximum(
            max(0, int((self.duration - self.x_viewport_size).total_seconds()))
        )
        self.scroll_x.setValue(int(self.x_viewport_start.total_seconds()))
        self.scroll_x.setPageStep(int(self.x_viewport_size.total_seconds()))
        self.scroll_x.blockSignals(False)

        self.update_display()


# 测试示例
import unittest


class Test(unittest.TestCase):
    def test_multiple_curves(self):
        from PySide6.QtWidgets import QApplication
        import pandas as pd
        import sys

        # 检查是否已存在实例
        app = QApplication.instance()
        if app is None:
            # 不存在则创建新实例
            app = QApplication(sys.argv)
            new_instance = True
        else:
            # 已存在则使用现有实例
            print("使用现有的QApplication实例")
            new_instance = False

        # 创建时间数据
        t_start = pd.Timestamp(year=2025, month=1, day=1, hour=0, minute=0, second=0, tz='UTC+08:00')
        x = np.arange(0, 100000, 100)  # 减少数据点以加快测试
        x_t = t_start + pd.to_timedelta(x, unit='s')

        # 创建多条曲线数据
        y1 = 50 * np.sin(2 * np.pi * 0.001 * x) + 10
        y2 = 30 * np.cos(2 * np.pi * 0.002 * x) + 20
        y3 = 20 * np.sin(2 * np.pi * 0.003 * x) * np.cos(2 * np.pi * 0.001 * x) + 30

        # 创建PlotPanel实例
        window = PlotPanel(label="多曲线示例")

        # 添加多条曲线
        window.add_curve(x_t, y1, label="正弦波", color='b')
        window.add_curve(x_t, y2, label="余弦波", color='r')
        window.add_curve(x_t, y3, label="混合波", color='g')

        window.set_label("时间", "幅值")
        window.show()

        if new_instance:
            app.exec()
        return 0

    def test_bar_panel(self):
        from PySide6.QtWidgets import QApplication
        import sys

        # 模拟Kp数据
        dates = pd.date_range(start='2025-01-01', end='2025-01-10', freq='3H')
        kp_values = np.random.randint(0, 9, size=len(dates))

        app = QApplication(sys.argv)
        window = PlotPanel(panel_type='bar', label="Kp指数")
        window.add_curve(dates, kp_values, label="Kp")
        window.show()
        app.exec()
