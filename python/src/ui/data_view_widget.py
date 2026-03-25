from typing import Optional

import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter,
    QSizePolicy, QSlider, QDateTimeEdit, QScrollBar,
    QToolBar, QLabel, QPushButton, QMessageBox
)
from PySide6.QtCore import Qt, QThread

from ui.brief_plot import BriefPlot, BriefWorker
from utils.time_convert import pts_to_qdt, qdt_to_pts
from core.em_data import EMData
from base.time_viewport_mixin import TimeViewportMixin

class DataViewWidget(TimeViewportMixin, QWidget):

    # -------------------------------------------------------------------------
    # 初始化
    # -------------------------------------------------------------------------

    def __init__(self, parent=None):
        super().__init__(parent)

        self.em_data: Optional[EMData] = None
        self.x_view_start: pd.Timedelta = pd.Timedelta(seconds=0)
        self.x_view_size: pd.Timedelta = pd.Timedelta(hours=1)
        self.duration: pd.Timedelta = pd.Timedelta(hours=0)

        # 子面板及其相关组件
        self.widget_data = pd.DataFrame(
            data={
                'widget': [None] * 3,
                'thread': [None] * 3,
                'worker': [None] * 3,
            },
            index=['E', 'B', 'Kp']
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
        self.slide_time.valueChanged.connect(self.on_x_view_size_changed)   # Mixin
        self.scroll_x.valueChanged.connect(self.on_x_view_start_changed)   # Mixin
        self.btn_plot.clicked.connect(self.on_btn_plot_clicked)
        self.start_time.dateTimeChanged.connect(self.on_time_changed)       # Mixin
        self.end_time.dateTimeChanged.connect(self.on_time_changed)         # Mixin

    # -------------------------------------------------------------------------
    # 数据加载
    # -------------------------------------------------------------------------

    def init_data(self, data: EMData):
        self.em_data = data
        if self.em_data is None:
            for row in self.widget_data.index:
                widget: Optional[BriefPlot] = self.widget_data.loc[row, 'widget']
                if widget is None:
                    continue
                widget.clear_curves()
                if widget.isVisible():
                    widget.hide()
            return

        self.duration = self.em_data.end_time - self.em_data.start_time

        self.set_time_range(                                                 # Mixin
            pts_to_qdt(self.em_data.start_time),
            pts_to_qdt(self.em_data.end_time),
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
        """初始化所有子图。"""
        for row in self.widget_data.index:
            label = row
            if row == 'E' and self.em_data.e_units is not None:
                label = row + f' / {self.em_data.e_units}'
            if row == 'B' and self.em_data.m_units is not None:
                label = row + f' / {self.em_data.m_units}'

            widget: Optional[BriefPlot] = self.widget_data.loc[row, 'widget']
            if widget is None:
                widget = BriefPlot(label=label)
                thread = QThread()
                worker = BriefWorker(label=row)
                worker.moveToThread(thread)
                worker.update.connect(widget.update_plot)
                thread.start()
                self.widget_data.loc[row, 'widget'] = widget
                self.widget_data.loc[row, 'worker'] = worker
                self.widget_data.loc[row, 'thread'] = thread
                self.splitter.addWidget(widget)
            else:
                widget.set_label(label)

    def _refresh_plots(self):
        """根据当前 data 更新各子图。"""
        if self.em_data is None:
            return

        for widget in self.widget_data['widget'].values:
            if widget is not None:
                widget.clear_curves()

        x_data = self.em_data.datetime_index

        # 2.1 处理 E, B(H) 通道
        for key, ch in self.em_data.data.items():
            if ch is not None and key.startswith('E'):
                y_data = ch.cts - ch.cts.mean()
                widget: BriefPlot = self.widget_data.loc['E', 'widget']
                widget.add_curve(
                    x_data=x_data,
                    y_data=y_data,
                    label=key,
                    graph_type='plot'
                )
            if ch is not None and (key.startswith('B') or key.startswith('H')):
                y_data = ch.cts - ch.cts.mean()
                b_widget: BriefPlot = self.widget_data.loc['B', 'widget']
                b_widget.add_curve(
                    x_data=x_data,
                    y_data=y_data,
                    label=key,
                    graph_type='plot'
                )
        # 2.2 处理 Kp 数据
        if self.em_data.kp_data is not None:
            kp_widget: BriefPlot = self.widget_data.loc['Kp', 'widget']
            kp_widget.add_curve(
                x_data=self.em_data.kp_data['Kp_datetime'],
                y_data=self.em_data.kp_data['Kp'],
                label='Kp',
                graph_type='bar'
            )

        # 按顺序添加并设置拉伸因子
        for row_key, widget in self.widget_data['widget'].items():
            if widget is not None and len(widget.curves) > 0:
                widget.show()
            else:
                # 如果该行没有数据，确保其控件在splitter中被隐藏
                if widget.isVisible():
                    widget.hide()

    # -------------------------------------------------------------------------
    # 视口更新（实现 Mixin 钩子）
    # -------------------------------------------------------------------------

    def _on_viewport_changed(self) -> None:
        """实现 Mixin 钩子：视口变化时通知所有子图 worker 更新范围。"""
        if self.em_data is None:
            return
        start = self.em_data.start_time + self.x_view_start
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
        start_duration, end_duration = self._clamp_jump(                    # Mixin
            start, end,
            data_start=self.em_data.start_time,
            data_end=self.em_data.end_time,
            tz=self.em_data.start_time.tz,
        )
        self._apply_jump(start_duration, end_duration)                      # Mixin

    def on_change_data_finished(self, channel: str) -> None:
        """外部通知某通道数据已更新，刷新对应子图曲线。"""
        if self.em_data is None:
            return
        x_data = self.em_data.datetime_index
        if channel.startswith('E'):
            row = 'E'
        elif channel.startswith('B') or channel.startswith('H'):
            row = 'B'
        else:
            return

        ch = self.em_data.data[channel]
        widget = self.widget_data.loc[row, 'widget']
        y_data = ch.cts - ch.cts.mean()
        if widget is not None:
            widget.update_curve(channel, x_data, y_data)
        self._on_viewport_changed()

    def close(self):
        for thread in self.widget_data['thread']:
            if thread is not None:
                thread.quit()
                thread.wait()
                thread.deleteLater()
