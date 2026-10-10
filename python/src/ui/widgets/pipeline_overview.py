from __future__ import annotations

"""Pipeline 时间轴的轻量 LOD 概览和刻度控件。"""

import numpy as np
from PySide6.QtGui import QFont, QPainter, QPalette, QPen
from PySide6.QtWidgets import QWidget

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
        self._pyramid: LODPyramid | None = None
        self._y_min = -0.5
        self._y_max = 0.5
        self._show_baseline = False
        self.setFixedHeight(CHANNEL_HEADER_H)
        self.setMinimumWidth(600)
        self.set_samples(samples)

    def set_zoom(self, zoom: float, pixels_per_point: float | None = None):
        self._zoom = zoom
        self._pixels_per_point = (
            px_per_point(zoom) if pixels_per_point is None else pixels_per_point
        )
        self.setMinimumWidth(max(1, int(self._data_len * self._pixels_per_point)))
        self.update()

    def set_samples(self, samples: np.ndarray):
        normalized = np.asarray(samples, dtype=np.float64)
        if normalized is self._samples:
            return
        self._samples = normalized
        if self._samples.size:
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
            self._pyramid = None
            self._y_min, self._y_max = -0.5, 0.5
        self.update()

    def set_baseline_visible(self, visible: bool) -> None:
        self._show_baseline = bool(visible)
        self.update()

    def _visible_lod(self, left_px: int, right_px: int) -> tuple[np.ndarray, np.ndarray]:
        if self._pyramid is None or self._samples.size == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        scale = self._pixels_per_point
        start = max(0.0, float(left_px) / scale)
        end = min(float(self._samples.size - 1), float(right_px + 1) / scale)
        return self._pyramid.get(start, end, max(1, right_px - left_px + 1))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.color(QPalette.ColorRole.Base))

        rect = event.rect()
        if rect.width() <= 0:
            painter.end()
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 1))

        x_lod, y_lod = self._visible_lod(rect.left(), rect.right())
        finite = np.isfinite(y_lod)
        if not finite.any():
            painter.end()
            return

        y_min, y_max = self._y_min, self._y_max
        span = y_max - y_min
        if span < 1e-12:
            span = 1.0
            y_min -= 0.5
            y_max += 0.5

        top = 3
        bottom = self.height() - 3
        height = max(1, bottom - top)

        def to_y(value: float) -> int:
            ratio = (value - y_min) / span
            return int(bottom - ratio * height)

        scale = self._pixels_per_point
        x_pixels = np.rint(x_lod[finite] * scale).astype(np.int64)
        values = y_lod[finite]
        order = np.argsort(x_pixels, kind="stable")
        x_pixels = x_pixels[order]
        values = values[order]
        starts = np.r_[0, np.flatnonzero(np.diff(x_pixels)) + 1]
        ends = np.r_[starts[1:], x_pixels.size]
        for start, end in zip(starts, ends):
            x_pos = int(x_pixels[start])
            if x_pos < rect.left() or x_pos > rect.right():
                continue
            bucket = values[start:end]
            painter.drawLine(x_pos, to_y(float(bucket.min())), x_pos, to_y(float(bucket.max())))

        if self._show_baseline and y_min <= 0.0 <= y_max:
            painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 1))
            baseline_y = to_y(0.0)
            painter.drawLine(rect.left(), baseline_y, rect.right(), baseline_y)
        painter.end()


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
