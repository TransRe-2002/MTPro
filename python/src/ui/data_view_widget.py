from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter,
    QSizePolicy, QSlider, QDateTimeEdit, QScrollBar,
    QToolBar, QLabel, QPushButton
)
from PySide6.QtCore import Qt, QThread

from ui.brief_plot import BriefPlot, BriefWorker
from utils.time_convert import pts_to_qdt, qdt_to_pts
from core.em_data import EMData
from base.time_viewport_mixin import TimeViewportMixin


@dataclass
class PlotResources:
    widget: Optional[BriefPlot] = None
    thread: Optional[QThread] = None
    worker: Optional[BriefWorker] = None

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

        # 子面板及其相关组件，按需创建
        self.plot_rows: Dict[str, PlotResources] = {
            'E': PlotResources(),
            'B': PlotResources(),
            'Kp': PlotResources(),
        }

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
        self._shutdown_done = False

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
            for resources in self.plot_rows.values():
                if resources.widget is not None:
                    resources.widget.clear_curves()
                    resources.widget.hide()
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

        self._sync_plot_rows()
        self._refresh_plots()
        self._on_viewport_changed()

    def _required_rows(self) -> set[str]:
        if self.em_data is None:
            return set()

        required: set[str] = set()
        has_e = any(key.startswith('E') for key in self.em_data.data)
        has_b = any(key.startswith('B') or key.startswith('H') for key in self.em_data.data)

        if has_e:
            required.add('E')
        if has_b:
            required.add('B')
        if self.em_data.kp_data is not None:
            required.add('Kp')
        return required

    def _label_for_row(self, row: str) -> str:
        if self.em_data is None:
            return row
        if row == 'E' and self.em_data.e_units is not None:
            return row + f' / {self.em_data.e_units}'
        if row == 'B' and self.em_data.m_units is not None:
            return row + f' / {self.em_data.m_units}'
        return row

    def _ensure_plot_row(self, row: str):
        resources = self.plot_rows[row]
        label = self._label_for_row(row)

        if resources.widget is None:
            widget = BriefPlot(label=label)
            thread = QThread(self)
            worker = BriefWorker(label=row)
            worker.moveToThread(thread)
            worker.update.connect(widget.update_plot)
            thread.start()

            resources.widget = widget
            resources.worker = worker
            resources.thread = thread
            self.splitter.addWidget(widget)
        else:
            resources.widget.set_label(label)

    def _teardown_plot_row(self, row: str):
        resources = self.plot_rows[row]

        if resources.thread is not None:
            resources.thread.quit()
            resources.thread.wait()
            resources.thread.deleteLater()

        if resources.worker is not None:
            resources.worker.deleteLater()

        if resources.widget is not None:
            resources.widget.hide()
            resources.widget.setParent(None)
            resources.widget.deleteLater()

        self.plot_rows[row] = PlotResources()

    def _sync_plot_rows(self):
        required_rows = self._required_rows()
        for row in self.plot_rows:
            if row in required_rows:
                self._ensure_plot_row(row)
            elif self.plot_rows[row].widget is not None:
                self._teardown_plot_row(row)

    def _refresh_plots(self):
        """根据当前 data 更新各子图。"""
        if self.em_data is None:
            return

        for resources in self.plot_rows.values():
            if resources.widget is not None:
                resources.widget.clear_curves()

        x_data = self.em_data.datetime_index

        # 2.1 处理 E, B(H) 通道
        for key, ch in self.em_data.data.items():
            if ch is not None and key.startswith('E'):
                y_data = ch.cts - ch.cts.mean()
                widget = self.plot_rows['E'].widget
                if widget is None:
                    continue
                widget.add_curve(
                    x_data=x_data,
                    y_data=y_data,
                    label=key,
                    graph_type='plot'
                )
            if ch is not None and (key.startswith('B') or key.startswith('H')):
                y_data = ch.cts - ch.cts.mean()
                b_widget = self.plot_rows['B'].widget
                if b_widget is None:
                    continue
                b_widget.add_curve(
                    x_data=x_data,
                    y_data=y_data,
                    label=key,
                    graph_type='plot'
                )
        # 2.2 处理 Kp 数据
        if self.em_data.kp_data is not None:
            kp_widget = self.plot_rows['Kp'].widget
            if kp_widget is not None:
                kp_widget.add_curve(
                    x_data=self.em_data.kp_data['Kp_datetime'],
                    y_data=self.em_data.kp_data['Kp'],
                    label='Kp',
                    graph_type='bar'
                )

        # 3. 显示所有有数据的 widget
        for resources in self.plot_rows.values():
            if resources.widget is not None:
                resources.widget.setVisible(len(resources.widget.curves) > 0)

    # -------------------------------------------------------------------------
    # 视口更新（实现 Mixin 钩子）
    # -------------------------------------------------------------------------

    def _on_viewport_changed(self) -> None:
        """实现 Mixin 钩子：视口变化时通知所有子图 worker 更新范围。"""
        if self.em_data is None:
            return
        start = self.em_data.start_time + self.x_view_start
        end = start + self.x_view_size
        for resources in self.plot_rows.values():
            if resources.worker is not None:
                resources.worker.range_change(start, end)
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

        self._sync_plot_rows()
        ch = self.em_data.data[channel]
        widget = self.plot_rows[row].widget
        y_data = ch.cts - ch.cts.mean()
        if widget is not None:
            if channel in widget.curves:
                widget.update_curve(channel, x_data, y_data)
            else:
                widget.add_curve(x_data, y_data, channel, graph_type='plot')
            widget.show()
        self._on_viewport_changed()

    def shutdown(self):
        if self._shutdown_done:
            return
        self._shutdown_done = True

        for row in list(self.plot_rows.keys()):
            if (
                self.plot_rows[row].widget is not None
                or self.plot_rows[row].thread is not None
                or self.plot_rows[row].worker is not None
            ):
                self._teardown_plot_row(row)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
