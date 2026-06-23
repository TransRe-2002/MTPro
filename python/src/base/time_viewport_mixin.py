from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMessageBox, QDateTimeEdit, QScrollBar,
    QSlider, QPushButton
)

from utils.signals import block_signals
from utils.time_convert import pts_to_qdt


class TimeViewportMixin:
    """视口状态管理混入类。

    提供 x_view_start / x_view_size 两个核心状态与 Qt 控件的双向同步，
    包括防抖视口更新、时间跳转限幅、以及滑块/滚动条的信号屏蔽同步。

    子类必须实现 _on_viewport_changed() 钩子来响应视口变更。
    """

    # -------------------------------------------------------------------------
    # 时间控件操作
    # -------------------------------------------------------------------------
    start_time: QDateTimeEdit
    end_time: QDateTimeEdit
    scroll_x: QScrollBar
    slide_time: QSlider
    duration: pd.Timedelta
    x_view_start: pd.Timedelta
    x_view_size: pd.Timedelta
    btn_plot: QPushButton

    _viewport_debounce: QTimer | None = None

    def _viewport_debounce_init(self) -> None:
        """在子类 __init__ 中调用，初始化视口更新防抖定时器。

        16ms 间隔对应约 60fps 刷新率，setSingleShot(True) 使得每次
        变化只重启定时器而非累积触发，避免连续滑动造成大量重绘。
        """
        if self._viewport_debounce is None:
            self._viewport_debounce = QTimer()
            self._viewport_debounce.setSingleShot(True)
            self._viewport_debounce.setInterval(16)  # 1000ms / 60 ≈ 16ms → 60fps
            self._viewport_debounce.timeout.connect(self._on_viewport_changed)

    def set_time(self, start, end) -> None:
        """设置时间控件显示值，屏蔽信号避免触发跳转逻辑。"""
        with block_signals(self.start_time, self.end_time):
            self.start_time.setDateTime(start)
            self.end_time.setDateTime(end)

    def set_time_range(self, start, end) -> None:
        """设置时间控件的可选范围并初始化显示值。"""
        with block_signals(self.start_time, self.end_time):
            self.start_time.setDateTime(start)
            self.start_time.setMinimumDateTime(start)
            self.end_time.setMinimumDateTime(start)
            self.end_time.setMaximumDateTime(end)
        self.set_time(start, end)

    # -------------------------------------------------------------------------
    # x 轴视口与控件同步
    # -------------------------------------------------------------------------

    def _sync_scroll_x(self) -> None:
        """将 scroll_x 同步到当前 x_view_start / x_view_size，屏蔽信号。"""
        with block_signals(self.scroll_x):
            self.scroll_x.setMaximum(
                max(0, int((self.duration - self.x_view_size).total_seconds()))
            )
            self.scroll_x.setValue(int(self.x_view_start.total_seconds()))
            self.scroll_x.setPageStep(int(self.x_view_size.total_seconds()))

    def _sync_slide_time(self) -> None:
        """将 slide_time 同步到当前 x_view_size，屏蔽信号。"""
        with block_signals(self.slide_time):
            self.slide_time.setValue(int(self.x_view_size.total_seconds()))

    def on_x_view_size_changed(self) -> None:
        """slide_time 变化槽函数：更新视口大小并同步滚动条。"""
        self.x_view_size = pd.Timedelta(seconds=self.slide_time.value())
        max_start = self.duration - self.x_view_size
        self.scroll_x.setMaximum(int(max_start.total_seconds()))
        self.scroll_x.setPageStep(int(self.x_view_size.total_seconds()))
        if self.x_view_start > max_start:
            self.x_view_start = max_start
            self.scroll_x.setValue(int(self.x_view_start.total_seconds()))
        self._viewport_debounce.start()

    def on_x_view_start_changed(self) -> None:
        """scroll_x 变化槽函数：更新视口起点。"""
        self.x_view_start = pd.Timedelta(seconds=self.scroll_x.value())
        self._viewport_debounce.start()

    def on_time_changed(self) -> None:
        """时间控件变化槽函数：启用跳转按钮。"""
        self.btn_plot.setEnabled(True)

    # -------------------------------------------------------------------------
    # 时间跳转（含边界限幅）
    # -------------------------------------------------------------------------

    def _clamp_jump(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        data_start: pd.Timestamp,
        data_end: pd.Timestamp,
        tz=None,
    ) -> tuple[pd.Timedelta, pd.Timedelta]:
        """
        对跳转目标时间做边界限幅，返回 (start_duration, end_duration)。
        start / end 为用户输入的目标时间，会自动 tz_convert。
        """
        if tz is not None:
            start = start.tz_convert(tz)
            end = end.tz_convert(tz)

        # 确保时间顺序：若起止时间颠倒则自动交换
        if start > end:
            self.show_warning("起始时间大于结束时间，已自动调换...")
            start, end = end, start

        # 限幅至数据边界，防止越界访问
        if start < data_start:
            start = data_start
            self.start_time.setDateTime(pts_to_qdt(start))
        if end > data_end:
            end = data_end
            self.end_time.setDateTime(pts_to_qdt(end))

        # 将绝对时间戳转换为相对于数据起点的持续时长，并限幅在 [0, duration]
        zero = pd.Timedelta(seconds=0)
        start_duration = max(zero, min(start - data_start, self.duration))
        end_duration = max(zero, min(end - data_start, self.duration))

        # 确保最小视口宽度为1分钟，避免零宽度视口
        if end_duration <= start_duration:
            end_duration = min(start_duration + pd.Timedelta(minutes=1), self.duration)

        return start_duration, end_duration

    def _apply_jump(self, start_duration: pd.Timedelta, end_duration: pd.Timedelta) -> None:
        """将限幅结果写入视口状态并同步控件。"""
        self.x_view_start = start_duration
        self.x_view_size = end_duration - start_duration
        self._sync_slide_time()
        self._sync_scroll_x()
        self._on_viewport_changed()
        self.btn_plot.setDisabled(True)

    # -------------------------------------------------------------------------
    # 警告消息框
    # -------------------------------------------------------------------------

    def show_warning(self, message: str) -> None:
        """显示 3 秒后自动关闭的非模态警告消息框。

        NonModal 模式让用户无需关闭警告框即可继续操作；QTimer.singleShot
        实现3秒后自动关闭，避免消息框堆积。
        """
        msg = QMessageBox(self)
        msg.setWindowModality(Qt.WindowModality.NonModal)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("警告")
        msg.setText(message)
        msg.show()
        QTimer.singleShot(3000, msg.close)

    # -------------------------------------------------------------------------
    # 留给子类实现的钩子
    # -------------------------------------------------------------------------

    def _on_viewport_changed(self) -> None:
        """
        视口参数（x_view_start / x_view_size）发生变化后的回调。
        子类必须重写此方法以驱动实际的绘图刷新。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 必须实现 _on_viewport_changed()"
        )
