from __future__ import annotations

"""Pipeline 时间轴的轻量 LOD 概览和刻度控件。"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QPoint, QTimer
from PySide6.QtGui import QFont, QPainter, QPalette, QPen
from PySide6.QtWidgets import QAbstractScrollArea, QWidget

from core.pipeline_layout import px_per_point, ruler_tick_multipliers
from core.pipeline_model import SnapEngine
from utils.decimate import LODPyramid

LAYER_H = 34
CHANNEL_HEADER_H = LAYER_H * 2
RULER_H = 24
PIPELINE_OVERVIEW_MAX_FINEST_POINTS = 250_000


class ChannelOverviewWidget(QWidget):
    """Viewport-aware LOD envelope for one Pipeline channel.

    Pipeline lanes can be much wider than their visible scroll-area viewport.
    Rendering only the visible LOD section keeps long channels responsive while
    preserving extrema through the min/max pyramid.
    """

    def __init__(self, samples: np.ndarray, data_len: int, parent=None):
        super().__init__(parent)
        self._data_len = max(1, data_len)
        self._zoom = 6.0
        self._pixels_per_point = px_per_point(self._zoom)
        self._samples = np.empty(0, dtype=np.float64)
        self._gap_starts = np.empty(0, dtype=np.int64)
        self._gap_ends = np.empty(0, dtype=np.int64)
        self._pyramid: LODPyramid | None = None
        self._y_min = -0.5
        self._y_max = 0.5
        self._show_baseline = False
        self._scroll_area = None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._refresh_plot)
        self.plot = pg.PlotWidget(
            self, background=self.palette().color(QPalette.ColorRole.Base)
        )
        self.plot.setFrameShape(self.plot.Shape.NoFrame)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.getPlotItem().layout.setContentsMargins(0, 0, 0, 0)
        self.curve = self.plot.plot(
            pen=pg.mkPen(self.palette().color(QPalette.ColorRole.Text), width=1),
            connect="finite",
        )
        self.isolated_points = pg.ScatterPlotItem(
            size=2, pen=None, brush=pg.mkBrush(self.palette().color(QPalette.ColorRole.Text))
        )
        self.plot.addItem(self.isolated_points)
        self.baseline = pg.InfiniteLine(
            pos=0, angle=0, movable=False,
            pen=pg.mkPen(self.palette().color(QPalette.ColorRole.Mid)),
        )
        self.plot.addItem(self.baseline, ignoreBounds=True)
        self.baseline.hide()
        self.setFixedHeight(CHANNEL_HEADER_H)
        self.setMinimumWidth(600)
        self.set_samples(samples)

    def set_zoom(self, zoom: float, pixels_per_point: float | None = None):
        self._zoom = zoom
        self._pixels_per_point = (
            px_per_point(zoom) if pixels_per_point is None else pixels_per_point
        )
        self.setMinimumWidth(max(1, int(self._data_len * self._pixels_per_point)))
        self._schedule_refresh()

    def set_samples(self, samples: np.ndarray):
        normalized = np.asarray(samples, dtype=np.float64)
        if normalized is self._samples:
            return
        self._samples = normalized
        if self._samples.size:
            gap_edges = np.flatnonzero(
                np.diff(np.r_[False, ~np.isfinite(self._samples), False])
            )
            self._gap_starts = gap_edges[::2]
            self._gap_ends = gap_edges[1::2]
            x_data = np.arange(self._samples.size, dtype=np.float64)
            self._pyramid = LODPyramid(
                x_data,
                self._samples,
                max_finest_points=PIPELINE_OVERVIEW_MAX_FINEST_POINTS,
            )
            finite = self._samples[np.isfinite(self._samples)]
            if finite.size:
                self._y_min = float(finite.min())
                self._y_max = float(finite.max())
            else:
                self._y_min, self._y_max = -0.5, 0.5
        else:
            self._gap_starts = np.empty(0, dtype=np.int64)
            self._gap_ends = np.empty(0, dtype=np.int64)
            self._pyramid = None
            self._y_min, self._y_max = -0.5, 0.5
        self._schedule_refresh()

    def set_baseline_visible(self, visible: bool) -> None:
        self._show_baseline = bool(visible)
        self.baseline.setVisible(self._show_baseline)

    def _visible_lod(self, left_px: int, right_px: int) -> tuple[np.ndarray, np.ndarray]:
        if self._pyramid is None or self._samples.size == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        scale = self._pixels_per_point
        start = max(0.0, float(left_px) / scale)
        end = min(float(self._samples.size - 1), float(right_px + 1) / scale)
        x, y = self._pyramid.get(start, end, max(1, right_px - left_px + 1))
        first = np.searchsorted(self._gap_ends, start, side="right")
        last = np.searchsorted(self._gap_starts, end, side="right")
        if first < last:
            # A min/max bucket can contain both valid and missing samples.
            # Keep one separator per pixel so the curve cannot bridge gaps.
            gap_positions = np.maximum(self._gap_starts[first:last], start)
            _, indices = np.unique(np.floor(gap_positions * scale), return_index=True)
            gap_x = gap_positions[indices]
            x = np.concatenate((x, gap_x))
            y = np.concatenate((y, np.full(gap_x.size, np.nan)))
            order = np.argsort(x, kind="stable")
            x, y = x[order], y[order]
        return x, y

    def _schedule_refresh(self, *_args) -> None:
        self._refresh_timer.start(0)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._scroll_area is None:
            parent = self.parentWidget()
            while parent is not None and not isinstance(parent, QAbstractScrollArea):
                parent = parent.parentWidget()
            if parent is not None:
                self._scroll_area = parent
                parent.viewport().installEventFilter(self)
                parent.horizontalScrollBar().valueChanged.connect(self._schedule_refresh)
                parent.verticalScrollBar().valueChanged.connect(self._schedule_refresh)
        self._schedule_refresh()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._schedule_refresh()

    def eventFilter(self, watched, event) -> bool:
        if self._scroll_area is not None and watched is self._scroll_area.viewport():
            if event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
                self._schedule_refresh()
        return super().eventFilter(watched, event)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            palette = self.palette()
            self.plot.setBackground(palette.color(QPalette.ColorRole.Base))
            self.curve.setPen(pg.mkPen(palette.color(QPalette.ColorRole.Text), width=1))
            self.isolated_points.setBrush(pg.mkBrush(palette.color(QPalette.ColorRole.Text)))
            self.baseline.setPen(pg.mkPen(palette.color(QPalette.ColorRole.Mid)))

    def _refresh_plot(self) -> None:
        visible = self.rect()
        if self._scroll_area is not None:
            viewport = self._scroll_area.viewport()
            visible = visible.intersected(
                viewport.rect().translated(self.mapFrom(viewport, QPoint(0, 0)))
            )
        if visible.isEmpty():
            self.plot.hide()
            return
        # The plot stays viewport-sized even when the timeline is millions of
        # pixels wide; its x coordinates remain aligned with the tracks.
        self.plot.setGeometry(visible.left(), 0, visible.width(), self.height())
        self.plot.show()
        x, y = self._visible_lod(visible.left(), visible.right())
        self.curve.setData(x, y, connect="finite")
        finite = np.isfinite(y)
        previous = np.r_[False, finite[:-1]] if finite.size else finite
        following = np.r_[finite[1:], False] if finite.size else finite
        isolated = finite & ~previous & ~following
        self.isolated_points.setData(x=x[isolated], y=y[isolated])
        scale = self._pixels_per_point
        self.plot.setXRange(visible.left() / scale, (visible.right() + 1) / scale, padding=0)
        y_min, y_max = self._y_min, self._y_max
        if y_max - y_min < 1e-12:
            y_min -= 0.5
            y_max += 0.5
        self.plot.setYRange(y_min, y_max, padding=3 / max(1, self.height() - 6))


class RulerWidget(QWidget):
    def __init__(self, snap: SnapEngine, data_len: int, parent=None) -> None:
        super().__init__(parent)
        self.snap = snap
        self.data_len = max(1, data_len)
        self.zoom = 6.0
        self._pixels_per_point = px_per_point(self.zoom)
        self.setFixedHeight(RULER_H)
        self.setMinimumWidth(600)

    def set_data_len(self, data_len: int):
        self.data_len = max(1, data_len)
        self.set_zoom(self.zoom, self._pixels_per_point)

    def set_zoom(self, zoom: float, pixels_per_point: float | None = None):
        self.zoom = zoom
        self._pixels_per_point = (
            px_per_point(zoom) if pixels_per_point is None else pixels_per_point
        )
        self.setMinimumWidth(max(1, int(self.data_len * self._pixels_per_point)))
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.color(QPalette.ColorRole.Base))
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 0.5))
        ruler_font = QFont("Courier New", 9)
        painter.setFont(ruler_font)

        step = max(50, self.snap.grid_pts)
        label_width = painter.fontMetrics().horizontalAdvance(str(self.data_len))
        tick_multiple, label_multiple = ruler_tick_multipliers(
            step,
            self.zoom,
            min_label_pixels=max(48.0, label_width + 12.0),
            pixels_per_point=self._pixels_per_point,
        )
        tick_step = step * tick_multiple
        label_step = step * label_multiple
        pts = 0
        while pts <= self.data_len:
            x_pos = int(pts * self._pixels_per_point)
            is_labeled_tick = pts % label_step == 0
            painter.drawLine(x_pos, 10 if is_labeled_tick else 16, x_pos, self.height())
            if is_labeled_tick:
                painter.setPen(palette.color(QPalette.ColorRole.Text))
                painter.drawText(x_pos + 2, 13, str(pts))
                painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 0.5))
            pts += tick_step
        painter.end()
