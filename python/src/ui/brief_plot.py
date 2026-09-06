from __future__ import annotations

"""BriefPlot --- 基于 pyqtgraph 的时间序列曲线/柱状图封装组件。

每个 BriefPlot 管理多条曲线，支持折线图(plot)和柱状图(bar)两种类型。
视口范围由外部（DataViewWidget/CompareWidget）在主线程防抖后直接
调用 update_plot 设置。
"""

import hashlib
import numpy as np
import pandas as pd
import colorsys
from PySide6.QtGui import QFont, QPen
import pyqtgraph as pg
from pyqtgraph import PlotWidget, DateAxisItem

from PySide6.QtCore import Qt, Signal
from typing import Optional

from utils.series import dti_to_numpy
from utils.theme import (
    configure_pyqtgraph_theme,
    pg_background,
    pg_foreground,
    subtle_pen,
    transparent_legend_brush,
)

LEFT_AXIS_WIDTH = 62
BOTTOM_AXIS_HEIGHT = 34


class PassiveLegendSample(pg.ItemSample):
    """Legend sample that displays a curve but never toggles it."""

    def mouseClickEvent(self, event) -> None:
        event.accept()

    def wheelEvent(self, event) -> None:
        event.accept()


class BriefPlot(PlotWidget):
    """时间序列绘图控件，封装 pyqtgraph PlotWidget。

    支持折线图 (plot) 和柱状图 (bar) 两种类型，
    自动根据 label 哈希生成稳定的曲线颜色。
    """

    vertical_gain_changed = Signal(float)

    def __init__(
            self,
            parent=None,
            label: str = "example",
        ) -> None:
        """初始化 BriefPlot 控件。

        Args:
            parent: 父级 Qt 控件
            label: X 轴底部标签文本
        """
        super().__init__(
            parent,
            axisItems={'bottom': DateAxisItem(orientation='bottom', utcOffset=0)}
        )
        configure_pyqtgraph_theme(self)
        self.setBackground(pg_background(self))
        self.label = label
        self.curves: dict[str, pg.GraphicsObject] = {}
        self._stack_labels: dict[str, pg.TextItem] = {}
        self._kp_bar: Optional[pg.BarGraphItem] = None
        self._kp_view: Optional[pg.ViewBox] = None
        self._kp_axis: Optional[pg.AxisItem] = None
        self.show_curves: dict[str, bool] = {}
        self.x_start: Optional[pd.Timestamp] = None
        self.x_end: Optional[pd.Timestamp] = None

        self.plot_item = self.getPlotItem()

        # 禁用鼠标拖拽/缩放交互，由外部 ViewBox 统一控制
        self.setMouseEnabled(x=False, y=False)
        self.sync_axis_layout()
        self.hideButtons()
        self.init_plot()

    def wheelEvent(self, event) -> None:
        """Use the mouse wheel for vertical waveform gain only."""
        delta = event.angleDelta().y()
        if delta:
            self.vertical_gain_changed.emit(1.15 ** (delta / 120.0))
            event.accept()
            return
        super().wheelEvent(event)

    def init_kp_axis(self) -> None:
        """Add the secondary right-axis ViewBox used by the Kp overlay."""
        if self._kp_view is not None:
            return
        self._kp_view = pg.ViewBox()
        self._kp_view.setZValue(-10)
        self._kp_axis = pg.AxisItem("right")
        self.plot_item.layout.addItem(self._kp_axis, 2, 3)
        self.plot_item.scene().addItem(self._kp_view)
        self._kp_axis.linkToView(self._kp_view)
        self._kp_view.setXLink(self.plot_item.vb)
        self._kp_view.setMouseEnabled(x=False, y=False)

        def sync_kp_view():
            self._kp_view.setGeometry(self.plot_item.vb.sceneBoundingRect())
            self._kp_view.linkedViewChanged(self.plot_item.vb, self._kp_view.XAxis)

        self.plot_item.vb.sigResized.connect(sync_kp_view)
        sync_kp_view()
        self._kp_axis.setLabel("Kp")
        self._kp_axis.setRange(0, 9)
        self._kp_axis.setPen(subtle_pen(self))
        self._kp_axis.setStyle(tickLength=4)
        self._kp_axis.setTickFont(QFont("Arial", 8))
        self._kp_axis.setWidth(48)

    def set_stacked_layout(
        self,
        visible_channels: list[str],
        offsets: dict[str, float],
        scale_height: float,
    ) -> None:
        """Arrange visible channels on evenly spaced display baselines."""
        visible = set(visible_channels)
        for ch, curve in self.curves.items():
            curve.setVisible(ch in visible)
        for ch in list(self._stack_labels):
            if ch not in visible:
                self.plot_item.removeItem(self._stack_labels.pop(ch))
        for ch in visible_channels:
            offset = offsets[ch]
            if ch not in self._stack_labels:
                label = pg.TextItem(ch, anchor=(1, 1), color=(0, 0, 0))
                self._stack_labels[ch] = label
                self.plot_item.addItem(label)
            marker_x = self.x_start.timestamp() if self.x_start else 0
            self._stack_labels[ch].setPos(marker_x, offset + scale_height * 0.85)
        if visible_channels:
            low = min(offsets[ch] for ch in visible_channels) - scale_height
            high = max(offsets[ch] for ch in visible_channels) + scale_height
            self.setYRange(low, high, padding=0.02)
        else:
            self.setYRange(-1, 1, padding=0)

    def sync_legend_visibility(self, visible_channels: list[str]) -> None:
        """Make the legend reflect checkbox state without making it interactive."""
        if self.plot_item.legend is None:
            return
        visible = set(visible_channels)
        legend = self.plot_item.legend
        for _, label_item in list(legend.items):
            if label_item.text not in visible:
                legend.removeItem(label_item.text)
        existing = {label_item.text for _, label_item in legend.items}
        for channel in visible_channels:
            curve = self.curves.get(channel)
            if curve is not None and channel not in existing:
                legend.addItem(curve, channel)

    def set_channel_offset(self, channel: str, offset: float, scale: float) -> None:
        """Apply a channel's display scale and vertical offset to its curve."""
        curve = self.curves.get(channel)
        if curve is None or not isinstance(curve, pg.PlotCurveItem):
            return
        data = getattr(curve, "_mtpro_data", None)
        if data is not None:
            x_values, y_values = data
            curve.setData(x_values, np.asarray(y_values) / max(scale, 1e-12) + offset)

    def add_kp_overlay(self, x_data, y_data) -> None:
        """Draw semi-transparent Kp bars in the secondary right-axis ViewBox."""
        self.init_kp_axis()
        if self._kp_bar is not None:
            self._kp_view.removeItem(self._kp_bar)
            self._kp_bar = None
        x_values = dti_to_numpy(x_data)
        y_values = np.asarray(y_data, dtype=float)
        if len(x_values) == 0:
            return
        width = float(np.nanmedian(np.diff(x_values))) if len(x_values) > 1 else 3600.0
        brushes = [pg.mkBrush(52, 152, 219, 35) if v < 4 else
                   pg.mkBrush(243, 156, 18, 35) if v < 6 else
                   pg.mkBrush(231, 76, 60, 35) for v in y_values]
        self._kp_bar = pg.BarGraphItem(
            x0=x_values - width, height=y_values, width=width,
            brushes=brushes, pen=None,
        )
        self._kp_view.addItem(self._kp_bar)
        self._kp_view.setYRange(0, 9, padding=0)

    def set_kp_visible(self, visible: bool) -> None:
        if self._kp_bar is not None:
            self._kp_bar.setVisible(visible)
        if self._kp_axis is not None:
            self._kp_axis.show()
            self._kp_axis.setLabel("Kp" if visible else "")
            self._kp_axis.setStyle(
                tickLength=4 if visible else 0,
                showValues=visible,
            )

    def sync_axis_layout(self) -> None:
        """Restore axis sizing and labels after pyqtgraph item/layout updates."""
        self.plot_item.showAxis('left', True)
        self.plot_item.showAxis('bottom', True)
        left_axis = self.getAxis('left')
        left_axis.show()
        left_axis.setStyle(tickLength=0, showValues=False)
        left_axis.setTickFont(QFont('Arial', 8))
        left_axis.setWidth(LEFT_AXIS_WIDTH)
        bottom_axis = self.getAxis('bottom')
        bottom_axis.show()
        bottom_axis.setStyle(tickLength=4)
        bottom_axis.setTickFont(QFont('Arial', 8))
        bottom_axis.setHeight(BOTTOM_AXIS_HEIGHT)
        bottom_axis.setTextPen(pg.mkPen(pg_foreground(self)))
        bottom_axis.setPen(pg.mkPen(pg_foreground(self), width=1))
        bottom_axis.setLabel('Time')
        if self.label is not None:
            self.plot_item.setLabel('bottom', self.label)
        self.plot_item.updateGeometry()
        self.updateGeometry()

    def show_warning(self, msg: str) -> None:
        """非模态警告提示，2 秒后自动消失。"""
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QLabel, QGraphicsProxyWidget
        label = QLabel(msg)
        label.setStyleSheet(
            "background: palette(base); color: palette(text); padding: 4px 8px; "
            "border: 1px solid palette(mid); border-radius: 4px; font-size: 12px;"
        )
        proxy = QGraphicsProxyWidget()
        proxy.setWidget(label)
        proxy.setZValue(100)
        self.plot_item.addItem(proxy)
        proxy.setPos(10, 10)
        QTimer.singleShot(2000, lambda: self.plot_item.removeItem(proxy) if proxy.scene() else None)

    def init_plot(self) -> None:
        """初始化绘图区域：设置网格线、图例样式及其锚定位置。"""
        self.hideButtons()  # 隐藏默认的缩放按钮
        grid_pen = subtle_pen(self, alpha=180, width=1)
        grid_pen.setStyle(Qt.PenStyle.SolidLine)
        # 开启 X/Y 方向网格，透明度 0.3
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.plot_item.getAxis('bottom').setPen(grid_pen)
        self.plot_item.getAxis('left').setPen(grid_pen)
        self.plot_item.addLegend(sampleType=PassiveLegendSample)
        legend = self.plot_item.legend
        # 锚定在右上角
        legend.anchor((1, 0), (1, 0))
        legend.setBrush(transparent_legend_brush(self))
        legend.setPen(subtle_pen(self))

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
        self.sync_axis_layout()
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

    def update_curve(
        self,
        label: str,
        x_data,
        y_data,
    ) -> bool:
        """原地更新已有折线曲线的数据（用于视口级 LOD 数据替换）。

        不重建 GraphicsItem，只调用 ``setData``，避免频繁视口更新时
        反复 addItem/removeItem 带来的开销。

        Args:
            label: 曲线标识。
            x_data: X 轴数据（epoch 秒 float 数组或 DatetimeIndex）。
            y_data: Y 轴数据。

        Returns:
            更新成功返回 True；曲线不存在或不是折线类型返回 False。
        """
        curve = self.curves.get(label)
        if curve is None or not isinstance(curve, pg.PlotCurveItem):
            return False
        if len(x_data) != len(y_data):
            return False
        x_values = dti_to_numpy(x_data)
        y_values = np.asarray(y_data, dtype=np.float64)
        curve.setData(x_values, y_values)
        return True

    def update_plot(
            self,
            x_viewport_start: pd.Timestamp,
            x_viewport_end: pd.Timestamp,
    ) -> None:
        """根据外部给定的视口起止时间戳，更新 X 轴显示范围。

        Args:
            x_viewport_start: 视口起始时间戳
            x_viewport_end: 视口结束时间戳
        """
        self.sync_axis_layout()
        x_start = x_viewport_start.timestamp()
        x_end = x_viewport_end.timestamp()
        # 确保 start < end
        if x_start >= x_end:
            x_start, x_end = x_end, x_start
        self.setXRange(x_start, x_end)
        self.x_start = x_viewport_start
        self.x_end = x_viewport_end
        label_x = x_start + (x_end - x_start) * 0.005
        for ch, label in self._stack_labels.items():
            label.setPos(label_x, label.pos().y())

    def hide_curve(self, label: str) -> None:
        """隐藏指定 label 的曲线（不删除数据，仅设置不可见）。"""
        self.curves[label].setVisible(False)

    def set_label(self, label: str) -> None:
        """修改底部 X 轴标签文字。"""
        self.label = label
        self.sync_axis_layout()

    def clear_curves(self) -> None:
        """清除所有曲线数据及可见性记录，恢复空白绘图区。"""
        self.clear()
        self.curves.clear()
        self.show_curves.clear()
        self._stack_labels.clear()
        if self._kp_view is not None and self._kp_bar is not None:
            self._kp_view.removeItem(self._kp_bar)
            self._kp_bar = None
        self.set_kp_visible(False)
        self.sync_axis_layout()
