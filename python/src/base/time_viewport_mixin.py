from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMessageBox, QDateTimeEdit, QScrollBar,
    QSlider, QPushButton
)

from utils.time_convert import pts_to_qdt


class TimeViewportMixin:

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

    def set_time(self, start, end) -> None:
        """设置时间控件显示值，屏蔽信号避免触发跳转逻辑。"""
        self.start_time.blockSignals(True)
        self.end_time.blockSignals(True)
        self.start_time.setDateTime(start)
        self.end_time.setDateTime(end)
        self.start_time.blockSignals(False)
        self.end_time.blockSignals(False)

    def set_time_range(self, start, end) -> None:
        """设置时间控件的可选范围并初始化显示值。"""
        self.start_time.blockSignals(True)
        self.end_time.blockSignals(True)
        self.start_time.setDateTime(start)
        self.start_time.setMinimumDateTime(start)
        self.end_time.setMinimumDateTime(start)
        self.end_time.setMaximumDateTime(end)
        self.start_time.blockSignals(False)
        self.end_time.blockSignals(False)
        self.set_time(start, end)

    # -------------------------------------------------------------------------
    # x 轴视口与控件同步
    # -------------------------------------------------------------------------

    def _sync_scroll_x(self) -> None:
        """将 scroll_x 同步到当前 x_view_start / x_view_size，屏蔽信号。"""
        self.scroll_x.blockSignals(True)
        self.scroll_x.setMaximum(
            max(0, int((self.duration - self.x_view_size).total_seconds()))
        )
        self.scroll_x.setValue(int(self.x_view_start.total_seconds()))
        self.scroll_x.setPageStep(int(self.x_view_size.total_seconds()))
        self.scroll_x.blockSignals(False)

    def _sync_slide_time(self) -> None:
        """将 slide_time 同步到当前 x_view_size，屏蔽信号。"""
        self.slide_time.blockSignals(True)
        self.slide_time.setValue(int(self.x_view_size.total_seconds()))
        self.slide_time.blockSignals(False)

    def on_x_view_size_changed(self) -> None:
        """slide_time 变化槽函数：更新视口大小并同步滚动条。"""
        self.x_view_size = pd.Timedelta(seconds=self.slide_time.value())
        max_start = self.duration - self.x_view_size
        self.scroll_x.setMaximum(int(max_start.total_seconds()))
        self.scroll_x.setPageStep(int(self.x_view_size.total_seconds()))
        if self.x_view_start > max_start:
            self.x_view_start = max_start
            self.scroll_x.setValue(int(self.x_view_start.total_seconds()))
        self._on_viewport_changed()

    def on_x_view_start_changed(self) -> None:
        """scroll_x 变化槽函数：更新视口起点。"""
        self.x_view_start = pd.Timedelta(seconds=self.scroll_x.value())
        self._on_viewport_changed()

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

        if start > end:
            self.show_warning("起始时间大于结束时间，已自动调换...")
            start, end = end, start

        if start < data_start:
            start = data_start
            self.start_time.setDateTime(pts_to_qdt(start))
        if end > data_end:
            end = data_end
            self.end_time.setDateTime(pts_to_qdt(end))

        zero = pd.Timedelta(seconds=0)
        start_duration = max(zero, min(start - data_start, self.duration))
        end_duration = max(zero, min(end - data_start, self.duration))

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
        """显示 3 秒后自动关闭的非模态警告消息框。"""
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
