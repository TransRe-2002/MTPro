from __future__ import annotations

"""BriefPlot --- 基于 pyqtgraph 的时间序列曲线/柱状图封装组件。

每个 BriefPlot 管理多条曲线，支持折线图(plot)和柱状图(bar)两种类型。
通过 BriefWorker 将视口范围变化经 Qt 信号转发给外部数据请求逻辑。
"""

import hashlib
import numpy as np
import pandas as pd
import colorsys
from PySide6.QtGui import QFont, QPen
import pyqtgraph as pg
from pyqtgraph import PlotWidget, DateAxisItem

from PySide6.QtCore import Qt, QObject, Signal
from typing import Optional

from utils.series import dti_to_numpy


class BriefWorker(QObject):
    """工作线程桥接对象：将视图范围变更通过 Qt 信号转发给数据源。"""

    # 视口范围变化时发出 (start, end) 时间戳
    update = Signal(pd.Timestamp, pd.Timestamp)
    # 曲线可见性变化时发出 (label, visible)
    visible = Signal(str, bool)

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.label = label

    def range_change(self, start: pd.Timestamp, end: pd.Timestamp):
        """通知外部视口范围已变更。"""
        self.update.emit(start, end)

    # 预留信号：设置曲线可见性
    def visible_curve(self, label: str, visible: bool):
        """切换指定曲线的可见性。"""
        self.visible.emit(label, visible)


class BriefPlot(PlotWidget):
    """时间序列绘图控件，封装 pyqtgraph PlotWidget。

    支持折线图 (plot) 和柱状图 (bar) 两种类型，
    自动根据 label 哈希生成稳定的曲线颜色。
    """

    def __init__(
            self,
            parent=None,
            label: str='example',
        ):
        """初始化 BriefPlot 控件。

        Args:
            parent: 父级 Qt 控件
            label: X 轴底部标签文本
        """
        super().__init__(
            parent,
            axisItems={'bottom': DateAxisItem(orientation='bottom', utcOffset=0)}
        )
        self.label = label
        self.curves: dict[str, pg.GraphicsObject] = {}
        self.show_curves: dict[str, bool] = {}
        self.x_start: Optional[pd.Timestamp] = None
        self.x_end: Optional[pd.Timestamp] = None

        self.plot_item = self.getPlotItem()

        # 禁用鼠标拖拽/缩放交互，由外部 ViewBox 统一控制
        self.setMouseEnabled(x=False, y=False)

    def show_warning(self, msg: str) -> None:
        """非模态警告提示，2 秒后自动消失。"""
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QLabel, QGraphicsProxyWidget
        label = QLabel(msg)
        label.setStyleSheet(
            "background: #fff3cd; color: #856404; padding: 4px 8px; "
            "border: 1px solid #ffc107; border-radius: 4px; font-size: 12px;"
        )
        proxy = QGraphicsProxyWidget()
        proxy.setWidget(label)
        proxy.setZValue(100)
        self.plot_item.addItem(proxy)
        proxy.setPos(10, 10)
        QTimer.singleShot(2000, lambda: self.plot_item.removeItem(proxy) if proxy.scene() else None)
        self.hideButtons()  # 隐藏默认的缩放按钮
        # 自定义 Y 轴样式：无刻度线，向左偏移标签文字
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
        """初始化绘图区域：设置网格线、图例样式及其锚定位置。"""
        grid_pen = pg.mkPen(
            color=pg.mkColor(100, 100, 100),
            width=1,
            antialiased=True,
            style=Qt.PenStyle.DashLine
        )
        # 开启 X/Y 方向网格，透明度 0.3
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
        """向绘图区域添加一条曲线或柱状图。

        折线图模式：若未指定 color 和 pen，则根据 label 的 MD5 哈希值
        计算 HSL 色相，保证同一 label 总是得到相同颜色。

        Args:
            x_data: X 轴时间序列数据
            y_data: Y 轴数值数据
            label: 曲线唯一标识，同时作为图例名称
            graph_type: 'plot' 折线图或 'bar' 柱状图
            color: 可选，指定折线颜色 (RGB 字符串)
            pen: 可选，自定义 QPen 画笔
        """
        if x_data is None or y_data is None or len(x_data) == 0 or len(y_data) == 0:
            self.show_warning("数据为空，无法添加曲线")
            return None

        # 检查数据长度是否一致
        if len(x_data) != len(y_data):
            self.show_warning(f"x_data长度({len(x_data)})与y_data长度({len(y_data)})不一致")
            return None

        x_values = dti_to_numpy(x_data)
        y_values = np.asarray(y_data)

        if label is None:
            label = f"曲线{len(self.curves) + 1}"

        # 创建并添加曲线
        if graph_type == 'bar':
            # 柱状图三色方案：Kp < 4 蓝，4 ≤ Kp < 6 黄，Kp ≥ 6 红
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
            # Kp 每个值代表前3小时区间，如 00:00 的 Kp 对应 21:00–00:00
            bar = pg.BarGraphItem(
                x0=x_values - width,
                height=y_values,
                width=width,
                brushes=bar_colors,
                pen=pg.mkPen(color='white', width=0.7),
                autoDownsample=True,  # 启用自动降采样
                downsampleMethod='peak',
                clipToView=True,
            )
            self.curves[label] = bar
            self.plot_item.addItem(bar)
            # 柱状图模式下隐藏图例，避免颜色映射误导
            if self.plot_item.legend is not None:
                self.plot_item.legend.hide()
        else:
            if pen is None:
                if color is None:
                    # 用 label 哈希生成稳定且互异的色相
                    # MD5 摘要 → 取 360 整数模 → 映射到 HSL 色相 (0~1)
                    # S=0.40, L=0.85 保证颜色偏亮、饱和度适中，易于区分
                    h = (int(hashlib.md5(label.encode()).hexdigest(), 16) % 360) / 360.0
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
                clipToView=True,
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
        """根据外部给定的视口起止时间戳，更新 X 轴显示范围。

        Args:
            x_viewport_start: 视口起始时间戳
            x_viewport_end: 视口结束时间戳
        """
        x_start = x_viewport_start.timestamp()
        x_end = x_viewport_end.timestamp()
        # 确保 start < end
        if x_start >= x_end:
            x_start, x_end = x_end, x_start
        self.setXRange(x_start, x_end)
        self.x_start = x_viewport_start
        self.x_end = x_viewport_end

    def update_curve(self, label: str,
            x_data: Optional[pd.Series | pd.DatetimeIndex],
            y_data: Optional[pd.Series]
        ):
        """更新指定 label 曲线的数据，刷新显示。

        Args:
            label: 曲线标识名称
            x_data: 新的 X 轴时间序列
            y_data: 新的 Y 轴数值序列
        """
        if x_data is None or y_data is None or len(x_data) == 0 or len(y_data) == 0:
            self.show_warning("数据为空，无法更新曲线")
            return None
        # 检查数据长度是否一致
        if len(x_data) != len(y_data):
            self.show_warning(f"x_data长度({len(x_data)})与y_data长度({len(y_data)})不一致")

        # 转换为NumPy数组
        x_values = dti_to_numpy(x_data)
        y_values = np.asarray(y_data)

        self.curves[label].setData(x_values, y_values)
        return None

    def hide_curve(self, label: str):
        """隐藏指定 label 的曲线（不删除数据，仅设置不可见）。"""
        self.curves[label].setVisible(False)

    def set_label(self, label: str):
        """修改底部 X 轴标签文字。"""
        self.label = label
        self.plot_item.setLabel('bottom', self.label)

    def clear_curves(self):
        """清除所有曲线数据及可见性记录，恢复空白绘图区。"""
        self.clear()
        self.curves.clear()
        self.show_curves.clear()
