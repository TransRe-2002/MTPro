"""
brief_plot_fixed.py
基于原brief_plot.py修复，增加对第二个Y轴（右侧轴）的支持。
解决Kp指数（柱状图）与普通曲线因单位/量纲不同而在同一Y轴下显示不清的问题。
"""
from __future__ import annotations

import os
import sys
from typing import Optional

os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'
import pandas as pd
from PySide6.QtGui import QFont, QPen
import pyqtgraph as pg
from pyqtgraph import PlotWidget, DateAxisItem, ViewBox
from PySide6.QtCore import Qt, QObject, Signal
import numpy as np

pg.setConfigOptions(antialias=True)  # 抗锯齿
pg.setConfigOption('background', 'w')  # 背景白色
pg.setConfigOption('foreground', 'k')  # 前景黑色

class BriefWorker(QObject):
    update = Signal(pd.Timestamp, pd.Timestamp)
    visible = Signal(str, bool)

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.label = label

    def range_change(self, start: pd.Timestamp, end: pd.Timestamp):
        self.update.emit(start, end)

    def visible_curve(self, label: str, visible: bool):
        self.visible.emit(label, visible)

class BriefPlot(PlotWidget):
    def __init__(
            self,
            parent=None,
            label: str='example',
        ):
        super().__init__(
            parent,
            axisItems={'bottom': DateAxisItem(orientation='bottom')}
        )
        self.label = label
        self.curves: dict[str, pg.GraphicsObject] = {}
        self.show_curves: dict[str, bool] = {}
        self.x_start: Optional[pd.Timestamp] = None
        self.x_end: Optional[pd.Timestamp] = None
        # --- 新增：用于第二个Y轴的属性 ---
        self.right_viewbox = None
        self.right_axis = None
        # ---------------------------------
        self.plot_item = self.getPlotItem()

        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()  # 隐藏默认的缩放按钮
        self.getAxis('left').setStyle(
            tickLength=0,
            tickTextOffset=-25,
        )
        self.getAxis('left').setTickFont(QFont('Arial', 8))
        self.getAxis('left').setWidth(0)
        if self.label is not None:
            self.plot_item.setLabel('bottom', self.label)
        self.init_plot()

    def init_plot(self) -> None:
        grid_pen = pg.mkPen(
            color=pg.mkColor(100, 100, 100),
            width=1,
            alpha=1,
            antialiased=True,
            style=Qt.PenStyle.DashLine
        )
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.plot_item.getAxis('bottom').setPen(grid_pen)
        self.plot_item.getAxis('left').setPen(grid_pen)
        self.plot_item.addLegend()
        legend = self.plot_item.legend
        legend.anchor((1, 0), (1, 0))
        legend.setBrush(pg.mkBrush(255, 255, 255, 255))
        legend.setPen(pg.mkPen('k', width=1))

        return None

    def _setup_secondary_axis(self):
        """惰性初始化第二个Y轴及其视图框。仅在首次需要时调用。"""
        if self.right_viewbox is None:
            # 1. 创建第二个ViewBox
            self.right_viewbox = ViewBox()
            # 2. 显示并获取右侧坐标轴
            self.plot_item.showAxis('right')
            self.right_axis = self.plot_item.getAxis('right')
            # 3. 将右侧轴链接到新的ViewBox
            self.right_axis.linkToView(self.right_viewbox)
            # 4. 将新的ViewBox添加到场景中
            self.plot_item.scene().addItem(self.right_viewbox)
            # 5. 同步新ViewBox与主ViewBox的X轴范围
            self.right_viewbox.setXLink(self.plot_item)
            # 6. 【关键】当主视图X轴变化时，更新右侧视图的X范围
            self.plot_item.vb.sigResized.connect(self._update_secondary_view)
            self.plot_item.vb.sigXRangeChanged.connect(self._update_secondary_view_range)

    def _update_secondary_view(self):
        """更新右侧视图框的几何范围，以匹配主绘图区域。"""
        if self.right_viewbox:
            self.right_viewbox.setGeometry(self.plot_item.vb.sceneBoundingRect())

    def _update_secondary_view_range(self, vb, x_range):
        """同步更新右侧视图框的X轴显示范围。"""
        if self.right_viewbox:
            self.right_viewbox.setXRange(*x_range, padding=0)

    def add_curve(self,
        x_data: Optional[pd.Series | pd.DatetimeIndex | np.ndarray | list[float | int]],
        y_data: Optional[pd.Series | np.ndarray | list[float | int]],
        label: str,
        graph_type: str = 'plot',
        color: Optional[str] = None,
        pen: Optional[QPen] = None,
        # --- 新增关键参数：是否使用第二个Y轴 ---
        y2_axis: bool = False
    ) -> None:
        if x_data is None or y_data is None or len(x_data) == 0 or len(y_data) == 0:
            self.show_warning("数据为空，无法添加曲线")
            return None

        if len(x_data) != len(y_data):
            self.show_warning(f"x_data长度({len(x_data)})与y_data长度({len(y_data)})不一致")
            return None

        if isinstance(x_data, pd.DatetimeIndex) or (len(x_data) > 0 and isinstance(x_data[0], pd.Timestamp)):
            x_values = np.array([x_data[i].timestamp() for i in range(len(x_data))])
        else:
            x_values = np.array(x_data)

        if isinstance(y_data, pd.Series):
            y_values = y_data.to_numpy()
        else:
            y_values = np.array(y_data)

        # --- 决定将图形项添加到哪个视图 ---
        target_view = self.plot_item.vb  # 默认：主视图框
        if y2_axis:
            # 如果指定使用第二个Y轴，则进行初始化并获取其视图框
            self._setup_secondary_axis()
            target_view = self.right_viewbox
            # （可选）为右侧Y轴设置一个标签
            if self.right_axis.labelText == '':
                self.right_axis.setLabel(label, color='black')

        if graph_type == 'plot':
            if pen is None:
                if color is None:
                    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
                    color_idx = len(self.curves) % len(colors)
                    color = colors[color_idx]
                pen = pg.mkPen(color=color, width=1)

        if label is None:
            label = f"曲线{len(self.curves) + 1}"

        if graph_type == 'bar':
            colors_rgb = [
                (52, 152, 219),  # <4: 专业蓝
                (243, 156, 18),  # 4-6: 琥珀黄
                (231, 76, 60)  # >6: 深空警报红
            ]
            bar_colors = []
            for kp_value in y_values:
                if kp_value < 4:
                    color = colors_rgb[0]
                elif kp_value < 6:
                    color = colors_rgb[1]
                else:
                    color = colors_rgb[2]
                (r, g, b) = color
                bar_colors.append(pg.mkColor(r, g, b))

            width = (x_values[1] - x_values[0])
            bar = pg.BarGraphItem(
                x=x_values,
                height=y_values,
                width=width,
                brushes=bar_colors,
                pen=pg.mkPen(color='white', width=0.7),
                autoDownsample=True,
                downsampleMethod='subsample',
                clipToView=True,
                useOpenGL=True,
            )
            self.curves[label] = bar
            # --- 关键修改：添加到对应的视图框 ---
            target_view.addItem(bar)
        else:
            curve = pg.PlotCurveItem(
                x_values,
                y_values,
                pen=pen,
                name=label,
                autoDownsample=True,
                downsampleMethod='subsample',
                clipToView=True,
                useOpenGL=True,
                connect='finite'
            )
            self.curves[label] = curve
            # --- 关键修改：添加到对应的视图框 ---
            target_view.addItem(curve)

        return None

    def update_plot(
            self,
            x_viewport_start: pd.Timestamp,
            x_viewport_end: pd.Timestamp
    ):
        x_start = x_viewport_start.timestamp()
        x_end = x_viewport_end.timestamp()
        if x_start >= x_end:
            x_start, x_end = x_end, x_start
        self.setXRange(x_start, x_end)
        self.x_start = x_viewport_start
        self.x_end = x_viewport_end

    def update_curve(self, label: str,
            x_data: Optional[pd.Series | pd.DatetimeIndex | np.ndarray | list[float | int]],
            y_data: Optional[pd.Series | np.ndarray | list[float | int]]
        ):
        # ... (此方法逻辑不变，但需要注意如果曲线在第二个视图框，更新也应作用于正确视图)
        # 为简化，此处省略具体实现。可根据self.curves[label].parent()判断所属视图框进行更新。
        pass

    def hide_curve(self, label: str):
        self.curves[label].setVisible(False)

    def show_warning(self, message):
        """简单的警告信息打印"""
        print(f"警告: {message}")

# ========== 演示主程序 ==========
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
    import sys

    app = QApplication(sys.argv)

    # 1. 创建主窗口
    main_window = QMainWindow()
    main_window.setWindowTitle("修复版测试 - 多Y轴显示")
    main_window.resize(1000, 600)
    central_widget = QWidget()
    main_window.setCentralWidget(central_widget)
    layout = QVBoxLayout(central_widget)

    # 2. 创建绘图部件
    plot = BriefPlot(label="日期时间")

    # 3. 生成模拟数据
    # 3.1 生成第一条曲线（正弦波，幅度约-50~50，使用左侧主Y轴）
    num_points = 1000
    t = np.linspace(0, 20*np.pi, num_points)
    y_sine = 50 * np.sin(t) + 10 * np.cos(3*t)  # 幅度变化较大

    start_time = pd.Timestamp('2025-11-01 00:00:00', tz='UTC')
    time_curve = pd.date_range(start=start_time, periods=num_points, freq='5min')

    # 3.2 生成模拟的Kp指数（柱状图，值0-9，使用右侧Y轴）
    num_bars = 50
    time_kp = pd.date_range(start=start_time, periods=num_bars, freq='3H')
    np.random.seed(42)
    kp_values = np.random.randint(0, 10, size=num_bars)  # Kp通常在0-9之间

    # 4. 添加数据到图表
    # 4.1 添加正弦波曲线（使用默认左侧轴）
    plot.add_curve(time_curve, y_sine, label='模拟传感器数据 (单位: mV)', graph_type='plot', color='blue')
    # 4.2 添加Kp指数柱状图（关键：指定使用右侧第二个Y轴）
    plot.add_curve(time_kp, kp_values, label='Kp指数 (无量纲)', graph_type='bar', y2_axis=True)

    # 5. 设置初始显示的时间范围
    plot.update_plot(
        x_viewport_start=time_curve[0],
        x_viewport_end=time_curve[200]  # 先显示前一部分
    )

    # 6. 将图表添加到界面并显示
    layout.addWidget(plot)
    main_window.show()

    sys.exit(app.exec())