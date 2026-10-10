"""Reusable pan, zoom, and reset controls for Matplotlib QtAgg canvases."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
from PySide6.QtCore import QEvent, QObject, QTimer, Qt
from PySide6.QtWidgets import QApplication, QPushButton


_ZOOM_STEP = 1.2
_REDRAW_INTERVAL_MS = 16


class MatplotlibInteractionController(QObject):
    """Add left-drag pan, wheel zoom, and a compact reset button to a canvas."""

    def __init__(self, canvas, figure) -> None:
        super().__init__(canvas)
        self.canvas = canvas
        self.figure = figure
        self._home_limits: dict[Hashable, tuple[tuple[float, float], tuple[float, float]]] = {}
        self._axes_signature: tuple[Hashable, ...] = ()
        self._drag_state: dict[str, Any] | None = None

        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(_REDRAW_INTERVAL_MS)
        self._redraw_timer.timeout.connect(self._flush_draw)

        self._reset_button_overlay = True
        self.reset_button = QPushButton("复原", canvas)
        self.reset_button.setToolTip("复原当前图像（也可在子图中双击）")
        self.reset_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_button.clicked.connect(self.reset)
        self.reset_button.adjustSize()

        canvas.installEventFilter(self)
        self._scroll_cid = canvas.mpl_connect("scroll_event", self._on_scroll)
        self._press_cid = canvas.mpl_connect("button_press_event", self._on_button_press)
        self._motion_cid = canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._release_cid = canvas.mpl_connect("button_release_event", self._on_button_release)
        self._draw_cid = canvas.mpl_connect("draw_event", self._on_draw)
        canvas.setCursor(Qt.CursorShape.OpenHandCursor)
        self._position_button()
        self.capture_home()

    def eventFilter(self, watched, event) -> bool:
        if watched is self.canvas and event.type() in {
            QEvent.Type.Resize,
            QEvent.Type.Show,
        }:
            self._position_button()
        return super().eventFilter(watched, event)

    def capture_home(self) -> None:
        """Record the current rendered limits as the next reset destination."""
        self._home_limits = {
            axis: (tuple(axis.get_xlim()), tuple(axis.get_ylim()))
            for axis in self.figure.axes
        }
        self._axes_signature = tuple(self.figure.axes)
        self.reset_button.setEnabled(bool(self._home_limits))
        self._position_button()

    def attach_reset_button(self, layout) -> None:
        """Move the reset control into a Qt layout outside the plot canvas."""
        self._reset_button_overlay = False
        self.reset_button.setParent(layout.parentWidget())
        layout.addWidget(self.reset_button)

    def hide_reset_button(self) -> None:
        """Keep pan and zoom interactions but remove the canvas reset overlay."""
        self._reset_button_overlay = False
        self.reset_button.hide()

    def reset(self) -> None:
        """Restore every subplot in this figure to its last rendered limits."""
        restored = False
        for axis in self.figure.axes:
            limits = self._home_limits.get(axis)
            if limits is None:
                continue
            axis.set_xlim(*limits[0])
            axis.set_ylim(*limits[1])
            restored = True
        if restored:
            self._request_draw()

    def _request_draw(self) -> None:
        """Coalesce dense mouse events into at most one repaint per frame."""
        if not self._redraw_timer.isActive():
            self._redraw_timer.start()

    def _flush_draw(self) -> None:
        self.canvas.draw_idle()

    def _position_button(self) -> None:
        if not self._reset_button_overlay:
            return
        self.reset_button.adjustSize()
        x = max(4, self.canvas.width() - self.reset_button.width() - 8)
        y = max(4, self.canvas.height() - self.reset_button.height() - 8)
        self.reset_button.move(x, y)
        self.reset_button.raise_()

    @staticmethod
    def _axis_mode(event: Any) -> tuple[bool, bool]:
        key = str(getattr(event, "key", "") or "").lower()
        control = "ctrl" in key or "control" in key
        shift = "shift" in key
        if not control and not shift:
            modifiers = QApplication.keyboardModifiers()
            control = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
            shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        if control and not shift:
            return True, False
        if shift and not control:
            return False, True
        return True, True

    @staticmethod
    def _zoom_limits(
        limits: tuple[float, float],
        center: float | None,
        factor: float,
        scale: str,
    ) -> tuple[float, float] | None:
        if center is None or not np.isfinite(center):
            return None
        start, end = limits
        if scale == "log":
            if start <= 0 or end <= 0 or center <= 0:
                return None
            start_log, end_log, center_log = np.log10([start, end, center])
            return (
                float(10 ** (center_log + (start_log - center_log) * factor)),
                float(10 ** (center_log + (end_log - center_log) * factor)),
            )
        return (
            float(center + (start - center) * factor),
            float(center + (end - center) * factor),
        )

    @staticmethod
    def _pan_limits(
        limits: tuple[float, float],
        pixel_delta: float,
        pixel_span: float,
        scale: str,
    ) -> tuple[float, float] | None:
        if not np.isfinite(pixel_delta) or pixel_span <= 0:
            return None
        start, end = limits
        if scale == "log":
            if start <= 0 or end <= 0:
                return None
            start_log, end_log = np.log10([start, end])
            shift = (end_log - start_log) * pixel_delta / pixel_span
            return (
                float(10 ** (start_log - shift)),
                float(10 ** (end_log - shift)),
            )
        shift = (end - start) * pixel_delta / pixel_span
        return float(start - shift), float(end - shift)

    def _on_scroll(self, event: Any) -> None:
        axis = getattr(event, "inaxes", None)
        if axis is None:
            return
        direction = getattr(event, "button", None)
        step = float(getattr(event, "step", 0.0) or 0.0)
        if direction == "up" or step > 0:
            factor = 1.0 / _ZOOM_STEP
        elif direction == "down" or step < 0:
            factor = _ZOOM_STEP
        else:
            return

        zoom_x, zoom_y = self._axis_mode(event)
        changed = False
        if zoom_x:
            x_limits = self._zoom_limits(
                tuple(axis.get_xlim()),
                getattr(event, "xdata", None),
                factor,
                axis.get_xscale(),
            )
            if x_limits is not None:
                axis.set_xlim(*x_limits)
                changed = True
        if zoom_y:
            y_limits = self._zoom_limits(
                tuple(axis.get_ylim()),
                getattr(event, "ydata", None),
                factor,
                axis.get_yscale(),
            )
            if y_limits is not None:
                axis.set_ylim(*y_limits)
                changed = True
        if changed:
            self._request_draw()

    def _on_button_press(self, event: Any) -> None:
        button = getattr(event, "button", None)
        button_value = getattr(button, "value", button)
        if (
            bool(getattr(event, "dblclick", False))
            and button_value == 1
            and getattr(event, "inaxes", None) is not None
        ):
            self.reset()
            return
        axis = getattr(event, "inaxes", None)
        x = getattr(event, "x", None)
        y = getattr(event, "y", None)
        if button_value != 1 or axis is None or x is None or y is None:
            return
        pan_x, pan_y = self._axis_mode(event)
        self._drag_state = {
            "axis": axis,
            "x": float(x),
            "y": float(y),
            "x_limits": tuple(axis.get_xlim()),
            "y_limits": tuple(axis.get_ylim()),
            "pan_x": pan_x,
            "pan_y": pan_y,
        }
        self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _on_motion(self, event: Any) -> None:
        state = self._drag_state
        if state is None:
            return
        x = getattr(event, "x", None)
        y = getattr(event, "y", None)
        if x is None or y is None:
            return

        axis = state["axis"]
        changed = False
        if state["pan_x"]:
            x_limits = self._pan_limits(
                state["x_limits"],
                float(x) - state["x"],
                float(axis.bbox.width),
                axis.get_xscale(),
            )
            if x_limits is not None:
                axis.set_xlim(*x_limits)
                changed = True
        if state["pan_y"]:
            y_limits = self._pan_limits(
                state["y_limits"],
                float(y) - state["y"],
                float(axis.bbox.height),
                axis.get_yscale(),
            )
            if y_limits is not None:
                axis.set_ylim(*y_limits)
                changed = True
        if changed:
            self._request_draw()

    def _on_button_release(self, event: Any) -> None:
        button = getattr(event, "button", None)
        button_value = getattr(button, "value", button)
        if self._drag_state is None or button_value != 1:
            return
        self._drag_state = None
        self.canvas.setCursor(Qt.CursorShape.OpenHandCursor)

    def _on_draw(self, _event: Any) -> None:
        if tuple(self.figure.axes) != self._axes_signature:
            self.capture_home()


def configure_matplotlib_interactions(canvas, figure) -> MatplotlibInteractionController:
    """Install the controller once and refresh its reset baseline."""
    controller = getattr(canvas, "_mtpro_interaction_controller", None)
    if controller is None:
        controller = MatplotlibInteractionController(canvas, figure)
        canvas._mtpro_interaction_controller = controller
    else:
        controller.figure = figure
        controller.capture_home()
    return controller
