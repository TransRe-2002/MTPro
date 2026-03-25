from __future__ import annotations

import pandas as pd
import colorsys
from PySide6.QtGui import QFont, QPen
import pyqtgraph as pg
from pyqtgraph import PlotWidget, DateAxisItem

from PySide6.QtCore import Qt, QObject, Signal
from typing import Optional

from utils.series import dti_to_numpy

class BriefWorker(QObject):
    update = Signal(pd.Timestamp, pd.Timestamp)
    visible = Signal(str, bool)

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.label = label

    def range_change(self, start: pd.Timestamp, end: pd.Timestamp):
        self.update.emit(start, end)

    # 预留信号：设置曲线可见性
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
            antialiased=True,
            style=Qt.PenStyle.DashLine
        )
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.plot_item.getAxis('bottom').setPen(grid_pen)
        self.plot_item.getAxis('left').setPen(grid_pen)
        self.plot_item.addLegend()
        legend = self.plot_item.legend
        # 锚定在右上角
        legend.anchor((1, 0), (1, 0))
        # 设置背景框（半透明白色填充，黑色边框）
        legend.setBrush(pg.mkBrush(255, 255, 255, 255))
        legend.setPen(pg.mkPen('k', width=1))

        return None

    def add_curve(self,
        x_data: Optional[pd.Series | pd.DatetimeIndex],
        y_data: Optional[pd.Series],
        label: str,
        graph_type: str = 'plot',
        color: Optional[str] = None,
        pen: Optional[QPen] = None,
    ) -> None:
        if x_data is None or y_data is None or len(x_data) == 0 or len(y_data) == 0:
            self.show_warning("数据为空，无法添加曲线")
            return None

        # 检查数据长度是否一致
        if len(x_data) != len(y_data):
            self.show_warning(f"x_data长度({len(x_data)})与y_data长度({len(y_data)})不一致")
            return None

        x_values = dti_to_numpy(x_data)
        y_values = y_data.to_numpy()

        if label is None:
            label = f"曲线{len(self.curves) + 1}"

        # 创建并添加曲线
        if graph_type == 'bar':
            colors_rgb = [
                (52, 152, 219),  # <4: 专业蓝
                (243, 156, 18),  # 4-6: 琥珀黄
                (231, 76, 60)  # >6: 深空警报红
            ]

            # 为每个Kp值分配颜色
            bar_colors = []
            for kp_value in y_values:
                if kp_value < 4:
                    color = colors_rgb[0]
                elif kp_value < 6:
                    color = colors_rgb[1]
                else:
                    color = colors_rgb[2]

                # 转换为PyQtGraph可用的颜色格式（0-255整数）
                (r, g, b) = color
                bar_colors.append(pg.mkColor(r, g, b))

            width = (x_values[1] - x_values[0])
            bar = pg.BarGraphItem(
                x=x_values,
                height=y_values,
                width=width,
                brushes=bar_colors,
                pen=pg.mkPen(color='white', width=0.7),
                autoDownsample=True,  # 启用自动降采样
                downsampleMethod='subsample',
                clipToView=True,
                useOpenGL=True,  # 关键参数：启用OpenGL加速
            )
            self.curves[label] = bar
            self.plot_item.addItem(bar)
            self.plot_item.legend.hide()
        else:
            if pen is None:
                if color is None:
                    # 用黄金角增量保证色相均匀分布，背景白色下颜色饱和且深
                    hue = (len(self.curves) * 137.508) % 360  # 黄金角分割
                    # HSL -> RGB: 饱和度85%, 亮度40% -> 颜色深、鲜艳、易区分
                    h = hue / 360.0
                    r, g, b = colorsys.hls_to_rgb(h, 0.40, 0.85)
                    color = (int(r * 255), int(g * 255), int(b * 255))
                    pen = pg.mkPen(color=color, width=1)
                else:
                    pen = pg.mkPen(color=color, width=1)
            curve = pg.PlotCurveItem(
                x_values,
                y_values,
                pen=pen,
                name=label,
                autoDownsample=True,  # 启用自动降采样
                downsampleMethod='subsample',
                clipToView=True,
                useOpenGL=True,  # 关键参数：启用OpenGL加速
                connect='finite'
            )
            self.curves[label] = curve
            self.plot_item.addItem(curve)
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
            x_data: Optional[pd.Series | pd.DatetimeIndex],
            y_data: Optional[pd.Series]
        ):
        if x_data is None or y_data is None or len(x_data) == 0 or len(y_data) == 0:
            self.show_warning("数据为空，无法更新曲线")
            return None
        # 检查数据长度是否一致
        if len(x_data) != len(y_data):
            self.show_warning(f"x_data长度({len(x_data)})与y_data长度({len(y_data)})不一致")

        # 转换为NumPy数组
        x_values = dti_to_numpy(x_data)
        y_values = y_data.to_numpy()

        self.curves[label].setData(x_values, y_values)
        return None

    def hide_curve(self, label: str):
        self.curves[label].setVisible(False)

    def clear_curves(self):
        self.clear()
        self.curves.clear()
        self.show_curves.clear()
