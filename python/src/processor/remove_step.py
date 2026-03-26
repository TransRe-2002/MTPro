from typing import Any, Optional

from PySide6.QtGui import QIntValidator, QAction, QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QLineEdit,
    QSizePolicy, QSlider, QDateTimeEdit, QScrollBar,
    QToolBar, QLabel, QPushButton, QMessageBox, QHBoxLayout,
    QGridLayout, QTextEdit
)
from PySide6.QtCore import Qt, Signal, QRegularExpression
import numpy as np
import pyqtgraph as pg

from core.em_data import Channel
from utils.series import dti_to_numpy


class RemoveStep(QWidget):
    result_signal = Signal(str, np.ndarray)

    def __init__(self,
                 channel: Channel,
                 parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot_widget = None
        self.plot_item = None

        self.ch: Channel = channel
        x = channel.datetime_index()
        self.x_data = dti_to_numpy(x)
        y = channel.cts
        self.nan_index = y[y.isna()].index.tolist()
        print("nan index:", self.nan_index)
        y = y.interpolate(method='linear')
        self.y_data = y
        self.diff_y_data = self.y_data.diff()
        print("diff y:", self.diff_y_data)
        self.duration = len(self.x_data)

        self.de_step_btn = QPushButton("阈值差分去除")
        self.de_step_threshold = QLineEdit()
        float_validator = QRegularExpressionValidator(
            QRegularExpression(r"^\d*\.?\d+$")
        )
        self.de_step_threshold.setValidator(float_validator)
        int_validator = QRegularExpressionValidator(QRegularExpression(r"^\d+$"))
        self.one_step_btn = QPushButton("多步差分去除")
        self.one_step_times = QLineEdit()
        self.one_step_times.setValidator(int_validator)
        self.confirm_btn = QPushButton("确认")

        self.x_viewport_start: Optional[int] = None
        self.x_viewport_size: Optional[int] = None
        self.y_viewport_start: Optional[int] = None
        self.y_viewport_size: Optional[int] = None

        self.zoom_ratio = 1.0

        self._init_ui()
        self._connect_signal()
        self._init_plot()

    def _init_ui(self):
        toolbar1 = QToolBar()
        self.zoom_in_action = QAction("放大", self)
        self.zoom_in_action.setIcon(QIcon.fromTheme("zoom-in"))
        self.zoom_in_action.setIconText("放大")
        toolbar1.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction("缩小", self)
        self.zoom_out_action.setIcon(QIcon.fromTheme("zoom-out"))
        self.zoom_out_action.setIconText("缩小")
        toolbar1.addAction(self.zoom_out_action)

        toolbar1.addSeparator()
        toolbar1.addWidget(QLabel("绘制时间长度："))

        self.slide_region = QSlider(Qt.Orientation.Horizontal)
        self.slide_region.setFixedWidth(400)
        self.slide_region.setEnabled(False)
        toolbar1.addWidget(self.slide_region)

        self.instruction_label = QLabel(
            "操作说明: 首先计算差分，然后根据差值选择去除方式，最后点击确认"
        )
        self.instruction_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )

        self.plot_widget = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem()})
        self.plot_widget.getAxis('left').setStyle(tickLength=0)
        self.plot_widget.getAxis('bottom').setStyle(tickLength=0)
        self.plot_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.plot_item = self.plot_widget.getPlotItem()

        self.scroll_x = QScrollBar(Qt.Orientation.Horizontal)
        self.scroll_x.setEnabled(False)
        self.scroll_y = QScrollBar(Qt.Orientation.Vertical)
        self.scroll_y.setEnabled(False)
        self.scroll_y.setInvertedControls(True)
        self.scroll_y.setInvertedAppearance(True)

        self.plain_text = QTextEdit()
        self.plain_text.setReadOnly(True)
        self.plain_text.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Expanding
        )

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(QLabel("差分阈值："))
        button_layout.addWidget(self.de_step_threshold)
        button_layout.addWidget(self.de_step_btn)
        button_layout.addStretch(1)
        button_layout.addWidget(QLabel("多步差分次数："))
        button_layout.addWidget(self.one_step_times)
        button_layout.addWidget(self.one_step_btn)
        button_layout.addStretch(1)
        button_layout.addWidget(self.confirm_btn)
        button_layout.addStretch(1)

        glay = QGridLayout()
        glay.addWidget(toolbar1, 0, 0, 1, 5)
        glay.addWidget(self.instruction_label, 1, 0, 1, 5)
        glay.addWidget(self.plain_text, 2, 0, 2, 1)
        glay.addWidget(self.plot_widget, 2, 1, 2, 4)
        glay.addWidget(self.scroll_x, 4, 1, 1, 4)
        glay.addWidget(self.scroll_y, 2, 5, 2, 1)
        glay.addLayout(button_layout, 5, 0, 1, 5)
        self.setLayout(glay)

    def _connect_signal(self):
        self.scroll_x.valueChanged.connect(self._on_scroll_x_changed)
        self.scroll_y.valueChanged.connect(self._on_scroll_y_changed)
        self.slide_region.valueChanged.connect(self._on_slide_region_changed)

        self.zoom_in_action.triggered.connect(self._on_zoom_in_triggered)
        self.zoom_out_action.triggered.connect(self._on_zoom_out_triggered)

    def _init_plot(self):
        if self.x_data is None and self.y_data is None:
            return

        y_values = (self.y_data - self.y_data.mean()).to_numpy()
        y_diff_values = self.diff_y_data.to_numpy()
        y_line = pg.PlotCurveItem(
            self.x_data,
            y_values,
            pen=pg.mkPen(255, 0, 0, width=1),
            name="序列数值",
        )
        dy_line = pg.PlotCurveItem(
            self.x_data,
            y_diff_values,
            pen=pg.mkPen(0, 0, 255, width=1),
            name="序列一阶差分",
        )
        legend = self.plot_item.addLegend()
        legend.anchor((1, 0), (1, 0))
        legend.setBrush(pg.mkBrush(255, 255, 255, 255))
        legend.setPen(pg.mkPen('k', width=1))
        legend.show()

        grid_pen = pg.mkPen(
            color=pg.mkColor(100, 100, 100),
            width=1,
            alpha=1,
            antialiased=True,
            style=Qt.PenStyle.DashLine
        )
        self.plot_item.showGrid(x=True, y=True, alpha=0.7)
        self.plot_item.getAxis('bottom').setPen(grid_pen)
        self.plot_item.getAxis('left').setPen(grid_pen)
        self.plot_item.addItem(y_line)
        self.plot_item.addItem(dy_line)

        self._setup_viewport_parameters()

    def _setup_viewport_parameters(self):
        """设置视口参数"""
        self.x_viewport_size = self.duration
        self.x_viewport_start = 0

        self.scroll_x.setEnabled(True)
        self.scroll_x.setMinimum(0)
        self.scroll_x.setMaximum(max(0, self.duration - self.x_viewport_size))
        self.scroll_x.setPageStep(self.x_viewport_size)
        self.scroll_x.setValue(self.x_viewport_start)

        y_min = self.y_data.min()
        y_max = self.y_data.max()
        y_diff = (y_max - y_min) / 2

        ymin =  -y_diff * 1.05
        ymax =  y_diff * 1.05
        yrange = ymax - ymin
        self.y_viewport_start = ymin
        self.y_viewport_size = yrange * self.zoom_ratio

        self.scroll_y.setEnabled(True)
        self.scroll_y.setMinimum(int(ymin * 10000))
        self.scroll_y.setMaximum(max(int(ymin * 10000), int((ymax - self.y_viewport_size) * 10000)))
        self.scroll_y.setPageStep(int(self.y_viewport_size * 10000))
        self.scroll_y.setValue(int(self.y_viewport_start * 10000))

        self.slide_region.setEnabled(True)
        self.slide_region.blockSignals(True)
        self.slide_region.setMinimum(min(2000, self.duration))
        self.slide_region.setMaximum(max(1, self.duration))
        self.slide_region.setValue(self.x_viewport_size)
        self.slide_region.blockSignals(False)

        self._update_display()

    # 视图控制函数
    def _on_scroll_x_changed(self):
        """滚动条值改变时更新绘图"""
        self._update_x_viewport()

    def _on_scroll_y_changed(self):
        """滚动条值改变时更新绘图"""
        self._update_y_viewport()

    def _on_slide_region_changed(self):
        """滑动条时间长度改变时更新绘图"""
        self._update_x_viewport_size()

    def _on_zoom_in_triggered(self):
        """放大信号"""
        if 1 / 64.0 < self.zoom_ratio <= 64.0:
            self.zoom_out_action.setEnabled(True)
            self.zoom_ratio /= 2.0
        self._update_y_viewport_size()
        if self.zoom_ratio <= 1 / 64.0:
            self.zoom_in_action.setDisabled(True)

    def _on_zoom_out_triggered(self):
        """缩小信号"""
        if 1 / 64.0 <= self.zoom_ratio < 64.0:
            self.zoom_in_action.setEnabled(True)
            self.zoom_ratio *= 2.0
        self._update_y_viewport_size()
        if self.zoom_ratio >= 64.0:
            self.zoom_out_action.setDisabled(True)

    def _update_x_viewport(self):
        """更新横坐标视口"""
        if self.x_data is None:
            return
        self.x_viewport_start = self.scroll_x.value()
        self._update_display()

    def _update_y_viewport(self):
        """更新纵坐标视口"""
        self.y_viewport_start = self.scroll_y.value() / 10000.0
        self._update_display()

    def _update_x_viewport_size(self):
        """更新横坐标视口大小"""
        if self.x_data is None:
            return

        self.x_viewport_size = self.slide_region.value()
        max_start = self.duration - self.x_viewport_size

        self.scroll_x.setMaximum(max(0, max_start))
        self.scroll_x.setPageStep(self.x_viewport_size)

        if self.x_viewport_start > max_start:
            self.x_viewport_start = max_start
            self.scroll_x.setValue(self.x_viewport_start)

        self._update_display()

    def _update_y_viewport_size(self):
        """更新纵坐标视口大小"""
        if self.x_data is None:
            return

        y_min = self.y_data.min()
        y_max = self.y_data.max()

        if self.zoom_ratio < 1.0:
            yrange = y_max - y_min
            self.y_viewport_size = yrange * self.zoom_ratio
            max_start = y_max - self.y_viewport_size
            self.y_viewport_start =  -self.y_viewport_size / 2

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

        self._update_display()

    def _update_display(self):
        """更新显示"""
        if self.x_data is None or self.y_data is None:
            return

        x_start = max(0, self.x_viewport_start)
        x_end = min(x_start + self.x_viewport_size, len(self.x_data) - 1)

        y_min = self.y_viewport_start
        y_max = y_min + self.y_viewport_size

        if self.zoom_ratio > 1.0:
            y_center = (y_min + y_max) / 2
            y_diff = (y_max - y_min) / 2
            y_min = y_center - y_diff * self.zoom_ratio
            y_max = y_center + y_diff * self.zoom_ratio

        self.plot_item.setXRange(self.x_data[x_start], self.x_data[x_end], padding=0)
        self.plot_item.setYRange(y_min, y_max, padding=0)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    from io_utils.mat_io import MatLoader

    em_data = MatLoader().load("/home/transen5/Project/atm_rpc/039BE-20240501-20240515-dt5_struct.mat")
    app = QApplication([])
    window = RemoveStep(channel=em_data.data['Ex1'])
    window.setWindowFlag(Qt.WindowType.Window)
    window.show()
    app.exec()
