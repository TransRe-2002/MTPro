from __future__ import annotations

"""处理器窗口共用的时序绘图控件。

这个类是对 pyqtgraph.PlotWidget 的一层轻包装，统一提供：
- 时间轴显示
- Shift/Alt 限制缩放轴
- 双击回调
- 右键回调

这样各个处理器窗口可以共用一致的交互体验。
"""

import os
from typing import Any

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from PySide6 import QtCore, QtGui
import pyqtgraph as pg

from processor._modifier_constraint import apply_modifier_axis_constraint

pg.setConfigOptions(antialias=True)
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


class StepPlotWidget(pg.PlotWidget):
    """带统一交互行为的时间序列绘图控件。"""
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs, axisItems={"bottom": pg.DateAxisItem()})
        self.double_click_callback = None
        self.right_click_callback = None
        self.zoom_callback = None
        self._setup_view()

    def _setup_view(self) -> None:
        """设置初始视图状态。"""
        self.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.getViewBox().setMouseEnabled(x=True, y=True)
        self.setBackground("w")
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        """滚轮缩放时临时应用轴向约束，结束后恢复。"""
        self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        view_box = self.getViewBox()
        apply_modifier_axis_constraint(view_box, ev.modifiers())
        old_yrange = view_box.viewRange()[1]
        super().wheelEvent(ev)
        view_box.setMouseEnabled(x=True, y=True)
        if self.zoom_callback is not None:
            new_yrange = view_box.viewRange()[1]
            self.zoom_callback(old_yrange, new_yrange)

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        """处理右键回调和中键缩放约束。"""
        self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        if (
            ev.button() == QtCore.Qt.MouseButton.RightButton
            and self.right_click_callback is not None
        ):
            scene_pos = self.mapToScene(int(ev.position().x()), int(ev.position().y()))
            view_pos = self.getViewBox().mapSceneToView(scene_pos)
            if bool(self.right_click_callback(float(view_pos.x()), float(view_pos.y()))):
                ev.accept()
                return
        if ev.button() == QtCore.Qt.MouseButton.MiddleButton:
            apply_modifier_axis_constraint(self.getViewBox(), ev.modifiers())
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        """中键拖动时根据修饰键动态更新轴向约束。"""
        if ev.buttons() & QtCore.Qt.MouseButton.MiddleButton:
            apply_modifier_axis_constraint(self.getViewBox(), ev.modifiers())
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        """中键释放后恢复双轴操作。"""
        if ev.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.getViewBox().setMouseEnabled(x=True, y=True)
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:
        """把双击位置转换到数据坐标，并交给外部回调。"""
        self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        if (
            ev.button() == QtCore.Qt.MouseButton.LeftButton
            and self.double_click_callback is not None
        ):
            scene_pos = self.mapToScene(int(ev.position().x()), int(ev.position().y()))
            view_pos = self.getViewBox().mapSceneToView(scene_pos)
            self.double_click_callback(float(view_pos.x()))
            ev.accept()
            return
        super().mouseDoubleClickEvent(ev)
