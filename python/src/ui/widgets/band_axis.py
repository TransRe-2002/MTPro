from __future__ import annotations

import math
from collections.abc import Sequence

import pyqtgraph as pg
from shiboken6 import isValid
from PySide6.QtCore import QEvent, QPointF
from PySide6.QtWidgets import QToolTip


def band_tooltip(name: str, low_hz: float, high_hz: float) -> str:
    """Describe a frequency band and its reciprocal period range."""
    low, high = float(low_hz), float(high_hz)
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high):
        return str(name)
    minimum_period = 1.0 / high
    maximum_period = f"{1.0 / low:.4g}" if low > 0 else "∞"
    return (f"{name}\n频率：{low:.4g}–{high:.4g} Hz\n"
            f"周期：{minimum_period:.4g}–{maximum_period} s")


class BandAxisItem(pg.AxisItem):
    """A compact axis with individual tooltips for its tick labels."""

    def __init__(self):
        super().__init__(orientation="bottom")
        self._band_ticks: list[tuple[float, str, str]] = []
        self._plot_widget = None

    def bind_plot(self, plot_widget) -> None:
        self._plot_widget = plot_widget
        plot_widget.viewport().installEventFilter(self)

    def set_band_ticks(self, ticks: Sequence[tuple[float, str, str]]) -> None:
        self._band_ticks = [(float(x), str(name), str(tip)) for x, name, tip in ticks]
        self.setTicks([[(x, name) for x, name, _tip in self._band_ticks]])

    def tooltip_at(self, axis_x: float) -> str | None:
        view = self.linkedView()
        if view is None:
            return None
        metrics = self.style.get('tickFont') or self.font()
        from PySide6.QtGui import QFontMetricsF
        font_metrics = QFontMetricsF(metrics)
        for position, name, tooltip in self._band_ticks:
            scene_point = view.mapViewToScene(QPointF(position, 0))
            tick_x = self.mapFromScene(scene_point).x()
            if abs(axis_x - tick_x) <= font_metrics.horizontalAdvance(name) / 2 + 4:
                return tooltip
        return None

    def eventFilter(self, watched, event):
        if (event.type() == QEvent.Type.ToolTip and self._plot_widget is not None
                and isValid(self._plot_widget) and watched is self._plot_widget.viewport()):
            axis_pos = self.mapFromScene(self._plot_widget.mapToScene(event.pos()))
            if self.boundingRect().contains(axis_pos):
                tooltip = self.tooltip_at(axis_pos.x())
                if tooltip:
                    QToolTip.showText(event.globalPos(), tooltip, watched)
                    return True
                QToolTip.hideText()
                return True
        return super().eventFilter(watched, event)
