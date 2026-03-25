from typing import Optional, List
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout,
    QSizePolicy, QSlider, QDateTimeEdit, QScrollBar,
    QToolBar, QLabel, QPushButton, QSplitter
)
from PySide6.QtCore import Qt, QThread
import pytz

from ui.brief_plot import BriefPlot, BriefWorker
from utils.time_convert import pts_to_qdt, qdt_to_pts
from core.em_data import Channel
from base.time_viewport_mixin import TimeViewportMixin


class CompareWidget(TimeViewportMixin, QWidget):
    def __init__(self, list_ch: List[Channel], parent=None):
        super().__init__(parent)
        self.start: pd.Timestamp = pd.Timestamp('2200-01-01 00:00:00', tz=pytz.timezone('Asia/Shanghai'))
        self.end: pd.Timestamp = pd.Timestamp('1900-01-01 00:00:00', tz=pytz.timezone('Asia/Shanghai'))
        self.list_ch = list_ch

        for ch in self.list_ch:
            if ch.start() < self.start:
                self.start = ch.start()
            if ch.end() > self.end:
                self.end = ch.end()

        self.x_view_start: pd.Timedelta = pd.Timedelta(seconds=0)
        self.x_view_size: pd.Timedelta = pd.Timedelta(hours=1)
        self.duration: pd.Timedelta = pd.Timedelta(hours=0)

        # 子面板及其相关组件
        self.widget_data = pd.DataFrame(
            data={
                'widget': [None] * 2,
                'thread': [None] * 2,
                'worker': [None] * 2,
            },
            index=['E', 'B']
        )

        self.splitter = QSplitter(Qt.Orientation.Vertical)  # 创建纵向分割器
        self.splitter.setChildrenCollapsible(False)  # 重要：防止图表被拖拽到完全消失

        self.slide_time = QSlider(Qt.Orientation.Horizontal)
        self.slide_time.setFixedWidth(300)
        self.slide_time.setDisabled(True)
        self.start_time = QDateTimeEdit()
        self.start_time.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.start_time.setCalendarPopup(True)
        self.end_time = QDateTimeEdit()
        self.end_time.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.end_time.setCalendarPopup(True)
        self.btn_plot = QPushButton("时间序列跳转")
        self.btn_plot.setDisabled(True)
        self.scroll_x = QScrollBar(Qt.Orientation.Horizontal)
        self.scroll_x.setDisabled(True)

        self.init_ui()
        self.connect_signals()
        self.init_data()

    def init_ui(self):
        plot_panel = QWidget()
        plot_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
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

    def connect_signals(self):
        self.slide_time.valueChanged.connect(self.on_x_view_size_changed)  # Mixin
        self.scroll_x.valueChanged.connect(self.on_x_view_start_changed)  # Mixin
        self.btn_plot.clicked.connect(self.on_btn_plot_clicked)
        self.start_time.dateTimeChanged.connect(self.on_time_changed)  # Mixin
        self.end_time.dateTimeChanged.connect(self.on_time_changed)  # Mixin

    def init_data(self):
        self.duration = self.end - self.start

        self.set_time_range(  # Mixin
            pts_to_qdt(self.start),
            pts_to_qdt(self.end),
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

        self._init_widget()
        self._refresh_plots()
        self._on_viewport_changed()

    def _init_widget(self):
        """ 初始化子图。"""
        for row in self.widget_data.index:
            widget: Optional[BriefPlot] = self.widget_data.loc[row, 'widget']
            if widget is None:
                widget = BriefPlot(label=row)
                worker = BriefWorker(label=row)
                thread = QThread()
                worker.moveToThread(thread)
                worker.update.connect(widget.update_plot)
                thread.start()
                self.widget_data.loc[row, 'widget'] = widget
                self.widget_data.loc[row, 'thread'] = thread
                self.widget_data.loc[row, 'worker'] = worker
                self.splitter.addWidget(widget)
            else:
                widget.set_label(row)

    def _refresh_plots(self):
        """ 将通道数据绘制到子图中。"""

        for widget in self.widget_data['widget'].values:
            if widget is not None:
                widget.clear_curves()

        for ch in self.list_ch:
            key = ch.name
            if key.startswith('E'):
                row = 'E'
            elif key.startswith('B') or key.startswith('H'):
                row = 'B'
            else:
                continue

            widget: BriefPlot = self.widget_data.loc[row, 'widget']
            x_data = ch.datetime_index()
            y_data = ch.cts - ch.cts.mean()
            label = f'{ch.parent().name} - {key}'
            widget.add_curve(
                x_data=x_data,
                y_data=y_data,
                label=label,
                graph_type='plot'
            )

        for key, widget in self.widget_data['widget'].items():
            if widget is not None and len(widget.curves) > 0:
                widget.show()
            else:
                if widget.isVisible():
                    widget.hide()

    def _on_viewport_changed(self) -> None:
        """实现 Mixin 钩子：视口变化时通知所有子图 worker 更新范围。"""
        start = self.start + self.x_view_start
        end = start + self.x_view_size
        for worker in self.widget_data['worker'].values:
            if worker is not None:
                worker.range_change(start, end)
        self.set_time(pts_to_qdt(start), pts_to_qdt(end))

    # -------------------------------------------------------------------------
    # 槽函数
    # -------------------------------------------------------------------------

    def on_btn_plot_clicked(self):
        """跳转到指定时间范围。"""
        start = qdt_to_pts(self.start_time.dateTime())
        end = qdt_to_pts(self.end_time.dateTime())
        start_duration, end_duration = self._clamp_jump(  # Mixin
            start, end,
            data_start=self.start,
            data_end=self.end,
            tz=self.start.tz,
        )
        self._apply_jump(start_duration, end_duration)  # Mixin

    def closeEvent(self, event):
        """确保窗口关闭时，所有工作线程被安全终止。"""
        for thread in self.widget_data['thread'].values:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait(500)  # 等待线程结束，最多500毫秒
        super().closeEvent(event)
