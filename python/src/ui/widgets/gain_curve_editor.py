from __future__ import annotations

"""Interactive per-band gain curve used by multiscale signal editors."""

from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg
from scipy.interpolate import PchipInterpolator
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from utils.theme import configure_pyqtgraph_theme, pg_background
from ui.widgets.band_axis import BandAxisItem


class GainCurveEditor(QWidget):
    """EQ-style curve whose fixed-frequency nodes can be dragged vertically."""

    gain_changed = Signal(int, float)
    editing_finished = Signal()

    def __init__(self, parent=None, *, minimum: float = 0.0, maximum: float = 4.0):
        super().__init__(parent)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self._targets: list[pg.TargetItem] = []
        self._logical_indices: list[int] = []
        self._band_names: list[str] = []
        self._band_tooltips: list[str] = []
        self._updating = False

        self.band_axis = BandAxisItem()
        self.plot_widget = pg.PlotWidget(self, axisItems={"bottom": self.band_axis})
        self.band_axis.bind_plot(self.plot_widget)
        configure_pyqtgraph_theme(self.plot_widget)
        self.plot_widget.setBackground(pg_background(self.plot_widget))
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.setLabel("left", "增益")
        self.plot_item.setLabel("bottom", "低频  ←  频带  →  高频")
        self.plot_item.showGrid(x=True, y=True, alpha=0.25)
        self.plot_item.setYRange(self.minimum, self.maximum, padding=0)
        self.plot_item.getViewBox().setMouseEnabled(x=False, y=False)
        self.plot_item.hideButtons()
        self.plot_item.setMenuEnabled(False)
        self.reference_line = pg.InfiniteLine(
            pos=1.0,
            angle=0,
            movable=False,
            pen=pg.mkPen(130, 130, 130, width=1, style=Qt.PenStyle.DashLine),
        )
        self.curve = pg.PlotCurveItem(
            pen=pg.mkPen(55, 135, 235, width=2),
            fillLevel=0.0,
            brush=pg.mkBrush(55, 135, 235, 35),
        )
        self.plot_item.addItem(self.reference_line)
        self.plot_item.addItem(self.curve)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)
        self.setMinimumHeight(150)

    @property
    def display_indices(self) -> tuple[int, ...]:
        return tuple(self._logical_indices)

    def set_bands(
        self,
        labels: Sequence[str],
        gains: Sequence[float],
        logical_indices: Sequence[int] | None = None,
        tooltips: Sequence[str] | None = None,
    ) -> None:
        for target in self._targets:
            self.plot_item.removeItem(target)
        self._targets.clear()

        count = min(len(labels), len(gains))
        if logical_indices is None:
            logical_indices = list(range(count))
        self._logical_indices = [int(value) for value in logical_indices[:count]]
        self._band_names = [str(value) for value in labels[:count]]
        self._band_tooltips = [str(tooltips[index]) if tooltips is not None and index < len(tooltips)
                               else self._band_names[index] for index in range(count)]
        show_target_labels = count <= 12

        for display_index, (name, gain) in enumerate(
            zip(self._band_names, gains[:count])
        ):
            label = (
                (lambda _x, y, band_name=name: self._target_label(band_name, y))
                if show_target_labels else None
            )
            target = pg.TargetItem(
                pos=(float(display_index), self._clamp(gain)),
                size=12 if count <= 12 else 7,
                symbol="o",
                pen=pg.mkPen(35, 95, 185, width=2),
                hoverPen=pg.mkPen(255, 180, 45, width=2),
                brush=pg.mkBrush(85, 155, 245, 210),
                hoverBrush=pg.mkBrush(255, 190, 70, 230),
                movable=True,
                label=label,
                labelOpts={"offset": (0, -20)},
            )
            target.sigPositionChanged.connect(
                lambda item=target, index=display_index:
                self._target_moved(index, item)
            )
            target.sigPositionChangeFinished.connect(
                lambda _item=target: self.editing_finished.emit()
            )
            self._targets.append(target)
            target.setToolTip(self._band_tooltips[display_index])
            self.plot_item.addItem(target)

        self._update_curve()
        self._update_ticks()
        self.plot_item.setXRange(-0.65, max(0.65, count - 0.35), padding=0.0)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_ticks()

    def _update_ticks(self) -> None:
        count = len(self._band_names)
        if not count:
            return
        metrics = self.plot_widget.fontMetrics()
        label_width = max(
            metrics.horizontalAdvance(line)
            for name in self._band_names for line in name.split("\n")
        ) + 16
        max_labels = max(1, min(12, (self.plot_widget.width() - 60) // max(1, label_width)))
        tick_step = max(1, int(np.ceil(count / max_labels)))
        ticks = [
            (float(index), self._band_names[index], self._band_tooltips[index])
            for index in range(0, count, tick_step)
        ]
        self.band_axis.set_band_ticks(ticks)

    def set_gain(self, logical_index: int, value: float) -> None:
        try:
            display_index = self._logical_indices.index(int(logical_index))
        except ValueError:
            return
        target = self._targets[display_index]
        self._updating = True
        try:
            target.setPos(float(display_index), self._clamp(value))
        finally:
            self._updating = False
        self._update_curve()

    def gains(self) -> dict[int, float]:
        return {
            logical_index: float(target.pos().y())
            for logical_index, target in zip(self._logical_indices, self._targets)
        }

    def _target_moved(self, display_index: int, target: pg.TargetItem) -> None:
        if self._updating:
            return
        gain = self._clamp(target.pos().y())
        self._updating = True
        try:
            target.setPos(float(display_index), gain)
        finally:
            self._updating = False
        self._update_curve()
        self.gain_changed.emit(self._logical_indices[display_index], gain)

    def _update_curve(self) -> None:
        if not self._targets:
            self.curve.setData([], [])
            return
        x_values = np.arange(len(self._targets), dtype=np.float64)
        y_values = np.asarray(
            [target.pos().y() for target in self._targets],
            dtype=np.float64,
        )
        if len(x_values) > 1:
            smooth_x = np.linspace(0, len(x_values) - 1, (len(x_values) - 1) * 24 + 1)
            smooth_y = PchipInterpolator(x_values, y_values)(smooth_x)
            self.curve.setData(smooth_x, smooth_y)
        else:
            self.curve.setData(x_values, y_values)

    def _clamp(self, value: float) -> float:
        return min(self.maximum, max(self.minimum, float(value)))

    @staticmethod
    def _target_label(name: str, value: float) -> str:
        short_name = name.split("\n", 1)[0]
        return f"{short_name}  {float(value):.2f}"


__all__ = ["GainCurveEditor"]
