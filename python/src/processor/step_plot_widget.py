from __future__ import annotations

from typing import Any

from PySide6 import QtCore, QtGui
import pyqtgraph as pg

pg.setConfigOptions(antialias=True)
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


class StepPlotWidget(pg.PlotWidget):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs, axisItems={"bottom": pg.DateAxisItem()})
        self._setup_view()

    def _setup_view(self) -> None:
        self.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.getViewBox().setMouseEnabled(x=True, y=True)
        self.setBackground("w")
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def _apply_modifier_axis_constraint(self, modifiers) -> None:
        view_box = self.getViewBox()
        if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            view_box.setMouseEnabled(x=True, y=False)
        elif modifiers & QtCore.Qt.KeyboardModifier.AltModifier:
            view_box.setMouseEnabled(x=False, y=True)
        else:
            view_box.setMouseEnabled(x=True, y=True)

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        view_box = self.getViewBox()
        self._apply_modifier_axis_constraint(ev.modifiers())
        super().wheelEvent(ev)
        view_box.setMouseEnabled(x=True, y=True)

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        if ev.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._apply_modifier_axis_constraint(ev.modifiers())
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.buttons() & QtCore.Qt.MouseButton.MiddleButton:
            self._apply_modifier_axis_constraint(ev.modifiers())
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.getViewBox().setMouseEnabled(x=True, y=True)
        super().mouseReleaseEvent(ev)
