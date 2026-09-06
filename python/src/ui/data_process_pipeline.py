from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from PySide6.QtCore import (
    QPoint,
    QRect,
    Qt,
    QObject,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPalette,
    QPen,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from base.data_manager import DataManager
from core.em_data import EMData
from core.pipeline_model import (
    ChannelPipeline,
    PipelineBlock,
    PipelineLayer,
    SnapEngine,
    uid,
)
from core.recipe import recipe_from_pipeline, save_recipe
from plugin_api import ChannelBlock, ChannelMeta
from ui.plugin_adapter import PluginParamDialog, run_plugin_process
from ui.data_process_shared import PROCESSING_TOOL_MIME, PIPELINE_TOOL_MIME
from ui.tools import registry as tool_registry
from utils.decimate import LODPyramid, PYRAMID_MAX_FINEST_POINTS
from utils.qt_bridge import run as run_in_pool


LAYER_H = 34
HEADER_W = 120
CHANNEL_HEADER_H = LAYER_H * 2
ADD_ROW_H = 24
RULER_H = 24
HANDLE_W = 7
MIN_BLOCK_SPAN = 100


def px_per_point(zoom: float) -> float:
    return max(0.02, zoom * 8 / 1000)


def points_to_pixels(points: int, zoom: float) -> float:
    return points * px_per_point(zoom)


def _is_pipeline_capable(spec) -> bool:
    # Pipeline block 必须产出与输入同长的单通道序列。robust estimate 与
    # remote reference 属于站级 output 工具，虽然有参数表单，却产出阻抗等
    # 新物理量，不能伪装成单通道变换 block。
    return tool_registry.get_plugin(spec.tool_id) is not None


def _build_tool_meta() -> Dict[str, Dict[str, object]]:
    meta: Dict[str, Dict[str, object]] = {}
    for spec in tool_registry.all_specs():
        if not _is_pipeline_capable(spec):
            continue
        entry: Dict[str, object] = {"name": spec.display_name, "kind": spec.kind}
        if spec.fields:
            entry["fields"] = list(spec.fields)
        meta[spec.tool_id] = entry
    return meta


TOOL_META: Dict[str, Dict[str, object]] = _build_tool_meta()


def refresh_tool_meta() -> None:
    """Refresh the pipeline catalogue after startup plugin discovery."""

    TOOL_META.clear()
    TOOL_META.update(_build_tool_meta())


BLOCK_STATUS_STYLE: Dict[str, Dict[str, str]] = {
    "empty": {"accent": "#8b8f97", "label": "待处理"},
    "configured": {"accent": "#7e8b99", "label": "已配置"},
    "ready": {"accent": "#778f7c", "label": "就绪"},
    "stale": {"accent": "#9a7b7b", "label": "需刷新"},
    "error": {"accent": "#b34b4b", "label": "执行失败"},
}

class PipelineParamDialog(QDialog):
    def __init__(
        self,
        channel_name: str,
        layer_label: str,
        block: PipelineBlock,
        parent=None,
    ):
        super().__init__(parent)
        meta = TOOL_META.get(block.tool_id, {})
        self.params: dict = {}
        self._fields: dict[str, tuple[QLineEdit, type]] = {}
        self.setWindowTitle(meta.get("name", block.tool_id))

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("channel", QLabel(channel_name))
        form.addRow("layer", QLabel(layer_label))
        form.addRow("tool", QLabel(str(meta.get("name", block.tool_id))))
        form.addRow("range", QLabel(f"{block.start} ~ {block.end}"))

        for key, label, parser, default in meta.get("fields", []):
            edit = QLineEdit(str(block.params.get(key, default)), self)
            form.addRow(label, edit)
            self._fields[key] = (edit, parser)

        layout.addLayout(form)

        tip = QLabel(
            "这类 block 先保存参数，不直接执行。\n"
            "后续接入真实 processor 后再参与 preview/commit。"
        )
        tip.setWordWrap(True)
        layout.addWidget(tip)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self) -> None:
        values: dict[str, object] = {}
        for key, (edit, parser) in self._fields.items():
            text = edit.text().strip()
            if not text:
                QMessageBox.warning(self, "参数错误", f"{key} 不能为空")
                return
            try:
                values[key] = parser(text)
            except ValueError:
                QMessageBox.warning(self, "参数错误", f"{key} 的值无效: {text}")
                return

        self.params = values
        super().accept()


@dataclass
class PreviewChannelSlice:
    name: str
    cts: pd.Series

    def datetime_index(self):
        return self.cts.index


class _PluginTaskBridge(QObject):
    """Relay a worker result through a QObject-bound queued signal."""

    done = Signal(object)
    failed = Signal(object)

    def __init__(self, metadata, parent=None):
        super().__init__(parent)
        self.metadata = metadata

    def relay_done(self, payload):
        self.done.emit((self.metadata, payload))

    def relay_failed(self, exc):
        self.failed.emit((self.metadata, exc))


def pipeline_tool_id_from_mime(mime_data) -> Optional[str]:
    """Read a tool id from the shared processing or Pipeline drag MIME."""
    for mime_type in (PIPELINE_TOOL_MIME, PROCESSING_TOOL_MIME):
        if not mime_data.hasFormat(mime_type):
            continue
        try:
            tool_id = bytes(mime_data.data(mime_type)).decode("utf-8")
        except UnicodeDecodeError:
            continue
        if tool_id:
            return tool_id
    return None


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
        self._samples = np.empty(0, dtype=np.float64)
        self._pyramid: LODPyramid | None = None
        self._y_min = -0.5
        self._y_max = 0.5
        self._show_baseline = False
        self.setFixedHeight(CHANNEL_HEADER_H)
        self.setMinimumWidth(600)
        self.set_samples(samples)

    def set_zoom(self, zoom: float):
        self._zoom = zoom
        self.setMinimumWidth(int(points_to_pixels(self._data_len, zoom)) + 40)
        self.update()

    def set_samples(self, samples: np.ndarray):
        self._samples = np.asarray(samples, dtype=np.float64)
        if self._samples.size:
            x_data = np.arange(self._samples.size, dtype=np.float64)
            self._pyramid = LODPyramid(
                x_data,
                self._samples,
                max_finest_points=PYRAMID_MAX_FINEST_POINTS,
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
        scale = px_per_point(self._zoom)
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

        scale = px_per_point(self._zoom)
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
        self.setFixedHeight(RULER_H)
        self.setMinimumWidth(600)

    def set_data_len(self, data_len: int):
        self.data_len = max(1, data_len)
        self.set_zoom(self.zoom)

    def set_zoom(self, zoom: float):
        self.zoom = zoom
        self.setMinimumWidth(int(points_to_pixels(self.data_len, zoom)) + 40)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.color(QPalette.ColorRole.Base))
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 0.5))
        painter.setFont(QFont("Courier New", 9))

        step = max(50, self.snap.grid_pts)
        pts = 0
        while pts <= self.data_len:
            x_pos = int(points_to_pixels(pts, self.zoom))
            painter.drawLine(x_pos, 16, x_pos, self.height())
            if pts % (step * 2) == 0 or step >= 2000:
                painter.setPen(palette.color(QPalette.ColorRole.Text))
                painter.drawText(x_pos + 2, 13, str(pts))
                painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 0.5))
            pts += step
        painter.end()


class PipelineTrackWidget(QWidget):
    add_block_requested = Signal(int, str, str, int)
    tool_dropped = Signal(int, str, str, str, int)
    block_selected = Signal(str)
    block_double_clicked = Signal(str)
    block_removed = Signal(str)
    guide_changed = Signal(list)
    layout_changed = Signal()
    block_geometry_changed = Signal(str)

    def __init__(
        self,
        pipeline: ChannelPipeline,
        layer: PipelineLayer,
        data_len: int,
        snap: SnapEngine,
        parent=None,
    ):
        super().__init__(parent)
        self.pipeline = pipeline
        self.layer = layer
        self.data_len = max(1, data_len)
        self.snap = snap
        self.zoom = 6.0
        self.selected_block_id = ""
        self._guides: list[int] = []
        self._drag_block: Optional[PipelineBlock] = None
        self._drag_mode = ""
        self._drag_origin_start = 0
        self._drag_origin_end = 0
        self._drag_origin_layer_id = ""
        self._drag_origin_channel = ""
        self._drag_pointer_offset_pts = 0
        self._drag_target_layer: Optional[PipelineLayer] = None
        self._drag_target_pipeline: Optional[ChannelPipeline] = None
        self.setFixedHeight(LAYER_H)
        self.setMinimumWidth(600)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)

    @property
    def _px_per_pt(self) -> float:
        return px_per_point(self.zoom)

    def _pts_to_px(self, pts: int) -> float:
        return points_to_pixels(pts, self.zoom)

    def _px_to_pts(self, px: int) -> int:
        return int(px / self._px_per_pt)

    def set_zoom(self, zoom: float):
        self.zoom = zoom
        self.setMinimumWidth(int(self._pts_to_px(self.data_len)) + 40)
        self.update()

    def set_selected(self, block_id: str):
        self.selected_block_id = block_id
        self.update()

    def set_guides(self, guides: list[int]):
        self._guides = list(guides)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.color(QPalette.ColorRole.Base))

        if self.snap.enabled:
            grid_pen = QPen(palette.color(QPalette.ColorRole.Mid), 0.5, Qt.PenStyle.DotLine)
            painter.setPen(grid_pen)
            pts = 0
            while pts <= self.data_len:
                x_pos = int(self._pts_to_px(pts))
                painter.drawLine(x_pos, 0, x_pos, self.height())
                pts += self.snap.grid_pts

        for block in self.layer.sorted_blocks():
            self._draw_block(painter, block)

        if self._guides:
            painter.setPen(QPen(palette.color(QPalette.ColorRole.Highlight), 1, Qt.PenStyle.DashLine))
            for guide in self._guides:
                x_pos = int(self._pts_to_px(guide))
                painter.drawLine(x_pos, 0, x_pos, self.height())

        painter.end()

    def _draw_block(self, painter: QPainter, block: PipelineBlock):
        x1 = int(self._pts_to_px(block.start))
        x2 = int(self._pts_to_px(block.end))
        width = max(x2 - x1, 12)
        height = self.height() - 6
        rect = QRect(x1, 3, width, height)

        palette = self.palette()
        color = palette.color(QPalette.ColorRole.Button)
        if self._drag_block is not None and self._drag_block.block_id == block.block_id:
            color.setAlpha(160)
        painter.fillRect(rect, color)

        status_meta = BLOCK_STATUS_STYLE.get(block.status, BLOCK_STATUS_STYLE["empty"])
        handle_color = palette.color(QPalette.ColorRole.Light)
        handle_color.setAlpha(24)
        painter.fillRect(x1, 3, HANDLE_W, height, handle_color)
        painter.fillRect(x2 - HANDLE_W, 3, HANDLE_W, height, handle_color)

        border_color = palette.color(QPalette.ColorRole.Mid)
        if block.block_id == self.selected_block_id:
            border_color = palette.color(QPalette.ColorRole.Highlight)
        elif block.status == "stale":
            border_color = QColor(status_meta["accent"])
        painter.setPen(QPen(border_color, 1))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

        painter.setPen(palette.color(QPalette.ColorRole.ButtonText))
        painter.setFont(QFont("Courier New", 9, QFont.Weight.Bold))
        tool_text = str(TOOL_META.get(block.tool_id, {}).get("name", block.tool_id))
        status_text = str(status_meta["label"])
        painter.drawText(
            rect.adjusted(HANDLE_W + 4, 0, -(HANDLE_W + 4), -2),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"{tool_text} [{status_text}]",
        )

    def dragEnterEvent(self, event) -> None:
        if pipeline_tool_id_from_mime(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if pipeline_tool_id_from_mime(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        tool_id = pipeline_tool_id_from_mime(event.mimeData())
        if tool_id is None:
            event.ignore()
            return

        start = self._px_to_pts(event.position().toPoint().x())
        self.tool_dropped.emit(self.pipeline.data_id, self.pipeline.channel, self.layer.layer_id, tool_id, start)
        event.acceptProposedAction()

    def _hit_test(self, x_pos: int) -> tuple[Optional[PipelineBlock], str]:
        for block in reversed(self.layer.sorted_blocks()):
            x1 = int(self._pts_to_px(block.start))
            x2 = max(int(self._pts_to_px(block.end)), x1 + 12)
            if x1 <= x_pos <= x1 + HANDLE_W:
                return block, "resize_left"
            if x2 - HANDLE_W <= x_pos <= x2:
                return block, "resize_right"
            if x1 <= x_pos <= x2:
                return block, "move"
        return None, ""

    def _block_intervals(self, layer: PipelineLayer, exclude_id: str) -> list[tuple[int, int]]:
        intervals: list[tuple[int, int]] = []
        cursor = 0
        for block in layer.sorted_blocks():
            if block.block_id == exclude_id:
                continue
            if block.start > cursor:
                intervals.append((cursor, block.start))
            cursor = max(cursor, block.end)
        if cursor < self.data_len:
            intervals.append((cursor, self.data_len))
        return intervals

    def _fit_start_in_layer(self, layer: PipelineLayer, exclude_id: str, span: int, raw_start: int) -> Optional[int]:
        valid_intervals = []
        for gap_start, gap_end in self._block_intervals(layer, exclude_id):
            if gap_end - gap_start >= span:
                valid_intervals.append((gap_start, gap_end - span))
        if not valid_intervals:
            return None

        best_start = valid_intervals[0][0]
        best_distance = None
        for start_min, start_max in valid_intervals:
            candidate = min(max(raw_start, start_min), start_max)
            distance = abs(candidate - raw_start)
            if best_distance is None or distance < best_distance:
                best_start = candidate
                best_distance = distance
        return best_start

    def _adjacent_bounds(self, layer: PipelineLayer, block_id: str) -> tuple[int, int]:
        blocks = layer.sorted_blocks()
        prev_end = 0
        next_start = self.data_len
        for index, block in enumerate(blocks):
            if block.block_id != block_id:
                continue
            if index > 0:
                prev_end = blocks[index - 1].end
            if index + 1 < len(blocks):
                next_start = blocks[index + 1].start
            break
        return prev_end, next_start

    def _move_block_to_layer(self, target_track: PipelineTrackWidget) -> bool:
        if self._drag_block is None:
            return False
        if self._drag_block in target_track.layer.blocks:
            self._drag_target_layer = target_track.layer
            self._drag_target_pipeline = target_track.pipeline
            return False
        source_layer = self._drag_target_layer or self.layer
        if self._drag_block in source_layer.blocks:
            source_layer.blocks.remove(self._drag_block)
        else:
            for candidate in self._iter_tracks():
                if self._drag_block in candidate.layer.blocks:
                    candidate.layer.blocks.remove(self._drag_block)
                    break
        target_track.layer.blocks.append(self._drag_block)
        self._drag_target_layer = target_track.layer
        self._drag_target_pipeline = target_track.pipeline
        return True

    def _owner(self) -> Optional[DataProcessPipelineWidget]:
        widget = self.parentWidget()
        while widget is not None and not isinstance(widget, DataProcessPipelineWidget):
            widget = widget.parentWidget()
        return widget

    def _iter_tracks(self):
        owner = self._owner()
        if owner is None:
            return []
        return [track for lane in owner._lane_widgets for track in lane._tracks]

    def _find_track(self, data_id: int, channel_name: str, layer_id: str) -> Optional[PipelineTrackWidget]:
        for track in self._iter_tracks():
            if (
                track.pipeline.data_id == data_id
                and track.pipeline.channel == channel_name
                and track.layer.layer_id == layer_id
            ):
                return track
        return None

    def _resolve_target_track(self, global_pos: QPoint) -> Optional[PipelineTrackWidget]:
        widget = QApplication.widgetAt(global_pos)
        while widget is not None and not isinstance(widget, PipelineTrackWidget):
            widget = widget.parentWidget()
        return widget

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return

        x_pos = event.position().toPoint().x()
        block, mode = self._hit_test(x_pos)
        if block is not None:
            self.selected_block_id = block.block_id
            self.block_selected.emit(block.block_id)
            self._drag_block = block
            self._drag_mode = mode
            self._drag_origin_start = block.start
            self._drag_origin_end = block.end
            self._drag_origin_layer_id = self.layer.layer_id
            self._drag_origin_channel = f"{self.pipeline.data_id}:{self.pipeline.channel}"
            self._drag_pointer_offset_pts = self._px_to_pts(x_pos) - block.start
            self._drag_target_layer = self.layer
            self._drag_target_pipeline = self.pipeline
            owner = self._owner()
            pipelines = [] if owner is None else owner.pipelines
            self.snap.update_edges(pipelines, exclude_id=block.block_id)
            self.update()
            event.accept()
            return

        self.selected_block_id = ""
        self.block_selected.emit("")
        self.add_block_requested.emit(
            self.pipeline.data_id,
            self.pipeline.channel,
            self.layer.layer_id,
            self._px_to_pts(x_pos),
        )
        self.update()

    def mouseMoveEvent(self, event) -> None:
        x_pos = event.position().toPoint().x()
        if self._drag_block is None:
            _block, mode = self._hit_test(x_pos)
            if mode in ("resize_left", "resize_right"):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif mode == "move":
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        guides: list[int] = []
        target_track = self
        if self._drag_mode == "move":
            current_pipeline = self._drag_target_pipeline or self.pipeline
            current_layer = self._drag_target_layer or self.layer
            current_track = self._find_track(current_pipeline.data_id, current_pipeline.channel, current_layer.layer_id)
            if current_track is None:
                current_track = self
            resolved = self._resolve_target_track(event.globalPosition().toPoint())
            if resolved is not None:
                target_track = resolved
            else:
                target_track = current_track
            local_x = target_track.mapFromGlobal(event.globalPosition().toPoint()).x()
            raw_start = target_track._px_to_pts(local_x) - self._drag_pointer_offset_pts
            span = self._drag_block.end - self._drag_block.start
            snapped_start, guides = self.snap.snap_interval(
                raw_start,
                raw_start + span,
                target_track._px_per_pt,
                target_track.data_len,
            )
            fitted_start = target_track._fit_start_in_layer(
                target_track.layer,
                self._drag_block.block_id,
                span,
                snapped_start,
            )
            if fitted_start is None and current_track is not None:
                target_track = current_track
                local_x = target_track.mapFromGlobal(event.globalPosition().toPoint()).x()
                raw_start = target_track._px_to_pts(local_x) - self._drag_pointer_offset_pts
                snapped_start, guides = self.snap.snap_interval(
                    raw_start,
                    raw_start + span,
                    target_track._px_per_pt,
                    target_track.data_len,
                )
                fitted_start = target_track._fit_start_in_layer(
                    target_track.layer,
                    self._drag_block.block_id,
                    span,
                    snapped_start,
                )
            if fitted_start is not None:
                self._move_block_to_layer(target_track)
                self._drag_block.start = fitted_start
                self._drag_block.end = fitted_start + span
        else:
            prev_end, next_start = self._adjacent_bounds(self.layer, self._drag_block.block_id)
            if self._drag_mode == "resize_left":
                snapped, guides = self.snap.snap(
                    self._px_to_pts(x_pos),
                    self._px_per_pt,
                    self.data_len,
                )
                self._drag_block.start = min(
                    max(snapped, prev_end),
                    self._drag_block.end - MIN_BLOCK_SPAN,
                )
            elif self._drag_mode == "resize_right":
                snapped, guides = self.snap.snap(
                    self._px_to_pts(x_pos),
                    self._px_per_pt,
                    self.data_len,
                )
                self._drag_block.end = max(
                    min(snapped, next_start),
                    self._drag_block.start + MIN_BLOCK_SPAN,
                )

        self._guides = guides
        self.guide_changed.emit(guides)
        self.layout_changed.emit()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        del event
        if self._drag_block is None:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        changed = (
            (self._drag_target_pipeline or self.pipeline).channel != self._drag_origin_channel
            or (self._drag_target_layer or self.layer).layer_id != self._drag_origin_layer_id
            or self._drag_block.start != self._drag_origin_start
            or self._drag_block.end != self._drag_origin_end
        )
        block_id = self._drag_block.block_id
        self._drag_block = None
        self._drag_mode = ""
        self._drag_target_layer = None
        self._drag_target_pipeline = None
        self._guides = []
        self.guide_changed.emit([])
        self.layout_changed.emit()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        if changed:
            self.block_geometry_changed.emit(block_id)

    def mouseDoubleClickEvent(self, event) -> None:
        block, _mode = self._hit_test(event.position().toPoint().x())
        if block is not None:
            self.block_double_clicked.emit(block.block_id)

    def contextMenuEvent(self, event) -> None:
        block, _mode = self._hit_test(event.pos().x())
        if block is None:
            return

        if self.selected_block_id == block.block_id:
            self.selected_block_id = ""
        self.block_removed.emit(block.block_id)
        self.update()


class ChannelPipelineWidget(QWidget):
    add_layer_requested = Signal(int, str)
    add_block_requested = Signal(int, str, str, int)
    tool_dropped = Signal(int, str, str, str, int)
    block_selected = Signal(str)
    block_double_clicked = Signal(str)
    block_removed = Signal(str)
    guide_changed = Signal(list)
    layout_changed = Signal()
    block_geometry_changed = Signal(str)

    def __init__(
        self,
        pipeline: ChannelPipeline,
        samples: np.ndarray,
        data_len: int,
        snap: SnapEngine,
        show_baseline: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.pipeline = pipeline
        self.samples = np.asarray(samples, dtype=np.float64)
        self.data_len = data_len
        self.snap = snap
        self.zoom = 6.0
        self._show_baseline = bool(show_baseline)
        self._tracks: list[PipelineTrackWidget] = []
        self.overview_widget: Optional[ChannelOverviewWidget] = None
        self._build()

    def _clear_layout(self, layout):
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)
        QWidget().setLayout(layout)

    def _build(self):
        self._clear_layout(self.layout())

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        left = QWidget(self)
        left.setFixedWidth(HEADER_W)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        header_left = QLabel(f"#{self.pipeline.data_id} {self.pipeline.channel}")
        header_left.setFixedHeight(CHANNEL_HEADER_H)
        header_left.setIndent(8)
        header_left.setFrameShape(QFrame.Shape.StyledPanel)
        left_layout.addWidget(header_left)

        self.overview_widget = ChannelOverviewWidget(self.samples, self.data_len, right)
        self.overview_widget.set_baseline_visible(self._show_baseline)
        self.overview_widget.set_zoom(self.zoom)
        right_layout.addWidget(self.overview_widget)

        self._tracks.clear()
        for layer in self.pipeline.layers:
            left_label = QLabel(layer.label)
            left_label.setFixedHeight(LAYER_H)
            left_label.setIndent(18)
            left_label.setFrameShape(QFrame.Shape.StyledPanel)
            left_layout.addWidget(left_label)

            track = PipelineTrackWidget(self.pipeline, layer, self.data_len, self.snap, self)
            track.set_zoom(self.zoom)
            track.add_block_requested.connect(self.add_block_requested)
            track.tool_dropped.connect(self.tool_dropped)
            track.block_selected.connect(self.block_selected)
            track.block_double_clicked.connect(self.block_double_clicked)
            track.block_removed.connect(self.block_removed)
            track.guide_changed.connect(self.guide_changed)
            track.layout_changed.connect(self.layout_changed)
            track.block_geometry_changed.connect(self.block_geometry_changed)
            self._tracks.append(track)
            right_layout.addWidget(track)

        add_left = QPushButton("+ add step", left)
        add_left.setFixedHeight(ADD_ROW_H)
        add_left.clicked.connect(lambda: self.add_layer_requested.emit(self.pipeline.data_id, self.pipeline.channel))
        left_layout.addWidget(add_left)

        add_right = QFrame(right)
        add_right.setFixedHeight(ADD_ROW_H)
        add_right.setFrameShape(QFrame.Shape.StyledPanel)
        right_layout.addWidget(add_right)

        layout.addWidget(left)
        layout.addWidget(right, 1)

    def add_layer(self):
        self.pipeline.add_layer()
        self._build()
        self.set_zoom(self.zoom)

    def set_zoom(self, zoom: float):
        self.zoom = zoom
        if self.overview_widget is not None:
            self.overview_widget.set_zoom(zoom)
        for track in self._tracks:
            track.set_zoom(zoom)

    def set_baseline_visible(self, visible: bool) -> None:
        self._show_baseline = bool(visible)
        if self.overview_widget is not None:
            self.overview_widget.set_baseline_visible(visible)

    def set_selected(self, block_id: str):
        for track in self._tracks:
            track.set_selected(block_id)

    def set_guides(self, guides: list[int]):
        for track in self._tracks:
            track.set_guides(guides)

    def refresh_tracks(self):
        for track in self._tracks:
            track.update()

    def refresh(self):
        if self.overview_widget is not None:
            self.overview_widget.update()
        self.refresh_tracks()


class DataProcessPipelineWidget(QWidget):
    change_finished_signal = Signal(int, str)

    def __init__(self, data_manager: Optional[DataManager] = None, parent=None) -> None:
        super().__init__(parent)
        self.data_manager = data_manager
        self.data_id = 0
        self.em_data: Optional[EMData] = None
        self._channel_refs: list[tuple[int, str]] = []
        self.selected_block_id: str = ""
        self.pipelines: list[ChannelPipeline] = []
        self._lane_widgets: list[ChannelPipelineWidget] = []
        self._editor_dialogs: list[QDialog] = []
        self._plugin_tasks: dict[str, int] = {}
        self._plugin_bridges: dict[str, _PluginTaskBridge] = {}
        self._plugin_generation = 0
        self._pending_lane_rebuild = False
        self._zoom = 6.0
        self._overview_baseline_visible = False
        self._snap = SnapEngine()
        self._ruler_widget: Optional[RulerWidget] = None
        self.pipeline_hint: Optional[QLabel] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        toolbar = QWidget(right_panel)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 6, 8, 6)
        toolbar_layout.setSpacing(8)

        self.pipeline_hint = QLabel(
            "将左侧“数据处理工具”拖到某个 step 轨道以添加 block。",
            toolbar,
        )
        toolbar_layout.addWidget(self.pipeline_hint)
        toolbar_layout.addSpacing(8)

        toolbar_layout.addWidget(QLabel("zoom"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal, toolbar)
        self.zoom_slider.setRange(1, 20)
        self.zoom_slider.setValue(int(self._zoom))
        self.zoom_slider.setTracking(False)
        self.zoom_slider.setFixedWidth(120)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        self.zoom_slider.sliderMoved.connect(self._on_zoom_preview_changed)
        toolbar_layout.addWidget(self.zoom_slider)
        self.zoom_label = QLabel(str(int(self._zoom)))
        toolbar_layout.addWidget(self.zoom_label)
        toolbar_layout.addSpacing(12)

        self.snap_checkbox = QCheckBox("snap", toolbar)
        self.snap_checkbox.setChecked(True)
        self.snap_checkbox.toggled.connect(self._on_snap_toggled)
        toolbar_layout.addWidget(self.snap_checkbox)

        self.overview_baseline_checkbox = QCheckBox("show zero baseline", toolbar)
        self.overview_baseline_checkbox.setChecked(self._overview_baseline_visible)
        self.overview_baseline_checkbox.toggled.connect(self._on_overview_baseline_toggled)
        toolbar_layout.addWidget(self.overview_baseline_checkbox)

        toolbar_layout.addWidget(QLabel("grid"))
        self.grid_spin = QSpinBox(toolbar)
        self.grid_spin.setRange(50, 5000)
        self.grid_spin.setSingleStep(50)
        self.grid_spin.setValue(self._snap.grid_pts)
        self.grid_spin.setFixedWidth(72)
        self.grid_spin.valueChanged.connect(self._on_grid_changed)
        toolbar_layout.addWidget(self.grid_spin)
        toolbar_layout.addWidget(QLabel("pts"))
        toolbar_layout.addStretch(1)

        preview_button = QPushButton("preview", toolbar)
        preview_button.clicked.connect(self._show_preview_summary)
        commit_button = QPushButton("commit", toolbar)
        commit_button.clicked.connect(self._commit_preview)
        export_button = QPushButton("export recipe", toolbar)
        export_button.clicked.connect(self._export_recipe)
        toolbar_layout.addWidget(preview_button)
        toolbar_layout.addWidget(commit_button)
        toolbar_layout.addWidget(export_button)
        right_layout.addWidget(toolbar)

        self.scroll = QScrollArea(right_panel)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_content = QWidget(self.scroll)
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(0)
        self.scroll_layout.addStretch(1)
        self.scroll.setWidget(self.scroll_content)
        right_layout.addWidget(self.scroll, 1)

        root.addWidget(right_panel, 1)

    def _canonical_tool_id(self, tool_id: str) -> str:
        return tool_registry.canonical_id(tool_id)

    def init_data(self, data, data_id: int = 0) -> None:
        self.data_id = data_id
        self.em_data = data
        self.selected_block_id = ""
        if self.em_data is None and not self._channel_refs:
            self.pipelines = []
            self._rebuild_lanes()
            return

        if self.em_data is not None and not self._channel_refs:
            self._channel_refs = [(data_id, channel_name) for channel_name in self.em_data.channel_names()]

        self._sync_pipelines(self._channel_refs)
        self._normalize_pipelines_for_current_window()
        self._rebuild_lanes()
        self._refresh_all_pipeline_statuses()
        self._refresh_selection()

    def add_channels(self, channel_refs: list[tuple[int, str]]) -> tuple[list[tuple[int, str]], list[tuple[int, str]], list[tuple[int, str]]]:
        added: list[tuple[int, str]] = []
        duplicated: list[tuple[int, str]] = []
        failed: list[tuple[int, str]] = []

        for data_id, channel_name in channel_refs:
            if (data_id, channel_name) in self._channel_refs:
                duplicated.append((data_id, channel_name))
                continue
            em_data = self._data_for(data_id)
            if em_data is None or not em_data.has_channel(channel_name):
                failed.append((data_id, channel_name))
                continue
            if not self._has_common_time_overlap(self._channel_refs + [(data_id, channel_name)]):
                failed.append((data_id, channel_name))
                continue
            self._channel_refs.append((data_id, channel_name))
            added.append((data_id, channel_name))

        if added:
            self._sync_pipelines(self._channel_refs)
            self._normalize_pipelines_for_current_window()
            self._rebuild_lanes()
            self._refresh_all_pipeline_statuses()
            self._refresh_selection()

        return added, duplicated, failed

    def remove_data(self, data_id: int) -> None:
        refs = [ref for ref in self._channel_refs if ref[0] == data_id]
        if not refs:
            return
        self._channel_refs = [ref for ref in self._channel_refs if ref[0] != data_id]
        self._sync_pipelines(self._channel_refs)
        self._normalize_pipelines_for_current_window()
        self._rebuild_lanes()
        self._refresh_all_pipeline_statuses()
        self._refresh_selection()

    def channel_count(self) -> int:
        return len(self._channel_refs)

    def channel_refs(self) -> list[tuple[int, str]]:
        return list(self._channel_refs)

    def window_title(self) -> str:
        if not self._channel_refs:
            return "Pipeline Editor"
        labels = [f"#{data_id}:{channel}" for data_id, channel in self._channel_refs]
        if len(labels) > 3:
            labels = labels[:3] + ["..."]
        return "Pipeline Editor - " + ", ".join(labels)

    def _build_default_pipelines(self, channel_refs: list[tuple[int, str]]) -> list[ChannelPipeline]:
        pipelines: list[ChannelPipeline] = []
        for data_id, channel_name in channel_refs:
            pipeline = ChannelPipeline(
                data_id=data_id,
                channel=channel_name,
                baseline_snapshot=self._snapshot_for_channel(data_id, channel_name),
            )
            pipeline.add_layer()
            pipelines.append(pipeline)
        return pipelines

    def _sync_pipelines(self, channel_refs: list[tuple[int, str]]):
        existing_by_ref = {pipeline.ref: pipeline for pipeline in self.pipelines}
        synced: list[ChannelPipeline] = []
        for data_id, channel_name in channel_refs:
            pipeline = existing_by_ref.get((data_id, channel_name))
            if pipeline is None:
                pipeline = ChannelPipeline(
                    data_id=data_id,
                    channel=channel_name,
                    baseline_snapshot=self._snapshot_for_channel(data_id, channel_name),
                )
                pipeline.add_layer()
            elif pipeline.baseline_snapshot is None:
                pipeline.baseline_snapshot = self._snapshot_for_channel(data_id, channel_name)
            synced.append(pipeline)
        self.pipelines = synced

    def _data_for(self, data_id: int) -> Optional[EMData]:
        if self.data_manager is not None:
            return self.data_manager.get(data_id)
        if self.em_data is not None and data_id == self.data_id:
            return self.em_data
        return None

    def _channel_series(self, data_id: int, channel_name: str) -> Optional[pd.Series]:
        data = self._data_for(data_id)
        if data is None:
            return None
        if not data.has_channel(channel_name):
            return None
        return data.get_channel_series(channel_name)

    def _time_range_for_channel(self, data_id: int, channel_name: str) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
        series = self._channel_series(data_id, channel_name)
        if series is None or series.empty:
            return None
        index = series.index
        if len(index) == 0:
            return None
        return pd.Timestamp(index[0]), pd.Timestamp(index[-1])

    def _common_time_range(
        self,
        refs: Optional[list[tuple[int, str]]] = None,
    ) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
        channel_refs = self._channel_refs if refs is None else refs
        if not channel_refs:
            return None

        starts: list[pd.Timestamp] = []
        ends: list[pd.Timestamp] = []
        for data_id, channel_name in channel_refs:
            channel_range = self._time_range_for_channel(data_id, channel_name)
            if channel_range is None:
                return None
            starts.append(channel_range[0])
            ends.append(channel_range[1])

        common_start = max(starts)
        common_end = min(ends)
        if common_start > common_end:
            return None
        return common_start, common_end

    def _has_common_time_overlap(self, refs: list[tuple[int, str]]) -> bool:
        return self._common_time_range(refs) is not None

    def _channel_slice(self, data_id: int, channel_name: str) -> pd.Series:
        series = self._channel_series(data_id, channel_name)
        if series is None:
            return pd.Series(dtype=np.float64)

        common_range = self._common_time_range()
        if common_range is None:
            return series.iloc[0:0]

        start, end = common_range
        return series.loc[start:end]

    def _snapshot_for_channel(self, data_id: int, channel_name: str) -> np.ndarray:
        series = self._channel_slice(data_id, channel_name)
        if series.empty:
            return np.array([], dtype=np.float64)
        return series.to_numpy(dtype=np.float64, copy=True)

    def _find_pipeline(self, data_id: int, channel_name: str) -> Optional[ChannelPipeline]:
        for pipeline in self.pipelines:
            if pipeline.data_id == data_id and pipeline.channel == channel_name:
                return pipeline
        return None

    def _pipeline_baseline(self, pipeline: ChannelPipeline) -> np.ndarray:
        if pipeline.baseline_snapshot is None:
            pipeline.baseline_snapshot = self._snapshot_for_channel(pipeline.data_id, pipeline.channel)
        return np.array(pipeline.baseline_snapshot, copy=True)

    def _pipeline_data_len(self, pipeline: ChannelPipeline) -> int:
        return int(self._channel_slice(pipeline.data_id, pipeline.channel).shape[0])

    def _editor_data_len(self) -> int:
        if not self.pipelines:
            return 0
        return max(self._pipeline_data_len(pipeline) for pipeline in self.pipelines)

    def _normalize_pipelines_for_current_window(self):
        for pipeline in self.pipelines:
            data_len = self._pipeline_data_len(pipeline)
            current_snapshot = self._snapshot_for_channel(pipeline.data_id, pipeline.channel)
            if pipeline.baseline_snapshot is None or pipeline.baseline_snapshot.size != data_len:
                pipeline.baseline_snapshot = current_snapshot
                for layer in pipeline.layers:
                    for block in layer.blocks:
                        self._invalidate_block_cache(block, "time window changed; reopen editor to refresh result")

            for layer in pipeline.layers:
                normalized_blocks: list[PipelineBlock] = []
                for block in layer.blocks:
                    if data_len <= 0:
                        self._invalidate_block_cache(block, "channel has no samples in the common time window")
                        continue
                    block.start = max(0, min(block.start, data_len))
                    block.end = max(block.start, min(block.end, data_len))
                    if block.end <= block.start:
                        self._invalidate_block_cache(block, "block is outside the common time window")
                        continue
                    normalized_blocks.append(block)
                layer.blocks = normalized_blocks

    def _rebuild_lanes(self):
        while self.scroll_layout.count() > 1:
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._lane_widgets.clear()
        self._ruler_widget = None
        if not self.pipelines:
            placeholder = QLabel("拖动一个或多个 channel 到工作区后开始构建 pipeline。", self.scroll_content)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setMargin(24)
            self.scroll_layout.insertWidget(0, placeholder)
            return

        ruler_row = QWidget(self.scroll_content)
        ruler_layout = QHBoxLayout(ruler_row)
        ruler_layout.setContentsMargins(0, 0, 0, 0)
        ruler_layout.setSpacing(0)

        ruler_spacer = QFrame(ruler_row)
        ruler_spacer.setFixedWidth(HEADER_W)
        ruler_spacer.setFixedHeight(RULER_H)
        ruler_spacer.setFrameShape(QFrame.Shape.StyledPanel)
        ruler_layout.addWidget(ruler_spacer)

        self._ruler_widget = RulerWidget(self._snap, max(1, self._editor_data_len()), ruler_row)
        self._ruler_widget.set_zoom(self._zoom)
        ruler_layout.addWidget(self._ruler_widget, 1)
        self.scroll_layout.insertWidget(0, ruler_row)

        for pipeline in self.pipelines:
            data_len = self._pipeline_data_len(pipeline)
            lane = ChannelPipelineWidget(
                pipeline,
                self._pipeline_baseline(pipeline),
                data_len,
                self._snap,
                show_baseline=self._overview_baseline_visible,
                parent=self.scroll_content,
            )
            lane.set_zoom(self._zoom)
            lane.add_layer_requested.connect(self._on_add_layer_requested)
            lane.add_block_requested.connect(self._on_add_block_requested)
            lane.tool_dropped.connect(self._on_tool_dropped)
            lane.block_selected.connect(self._on_block_selected)
            lane.block_double_clicked.connect(self._on_block_double_clicked)
            lane.block_removed.connect(self._on_block_removed)
            lane.guide_changed.connect(self._broadcast_guides)
            lane.layout_changed.connect(self._refresh_lane_tracks)
            lane.block_geometry_changed.connect(self._on_block_geometry_changed)
            self._lane_widgets.append(lane)
            self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, lane)

    def _on_zoom_changed(self, value: int):
        self._zoom = float(value)
        self.zoom_label.setText(str(value))
        if self._ruler_widget is not None:
            self._ruler_widget.set_zoom(self._zoom)
        for lane in self._lane_widgets:
            lane.set_zoom(self._zoom)

    def _on_zoom_preview_changed(self, value: int):
        self.zoom_label.setText(str(value))

    def _on_snap_toggled(self, checked: bool):
        self._snap.enabled = checked
        if self._ruler_widget is not None:
            self._ruler_widget.update()
        self._refresh_lane_tracks()

    def _on_overview_baseline_toggled(self, checked: bool):
        self._overview_baseline_visible = bool(checked)
        for lane in self._lane_widgets:
            lane.set_baseline_visible(checked)

    def _on_grid_changed(self, value: int):
        self._snap.grid_pts = value
        if self._ruler_widget is not None:
            self._ruler_widget.update()
        self._refresh_lane_tracks()

    def _on_add_layer_requested(self, data_id: int, channel_name: str):
        for lane in self._lane_widgets:
            if lane.pipeline.data_id == data_id and lane.pipeline.channel == channel_name:
                lane.add_layer()
                lane.set_zoom(self._zoom)
                return

    def _default_params_for_tool(self, tool_id: str) -> dict:
        meta = TOOL_META.get(tool_id, {})
        params: dict[str, object] = {}
        for key, _label, _parser, default in meta.get("fields", []):
            params[key] = default
        return params

    def _find_layer(self, data_id: int, channel_name: str, layer_id: str) -> Optional[tuple[ChannelPipeline, PipelineLayer]]:
        pipeline = self._find_pipeline(data_id, channel_name)
        if pipeline is None:
            return None
        for layer in pipeline.layers:
            if layer.layer_id == layer_id:
                return pipeline, layer
        return None

    def _block_intervals(self, layer: PipelineLayer, data_len: int, exclude_id: str = "") -> list[tuple[int, int]]:
        intervals: list[tuple[int, int]] = []
        cursor = 0
        for block in layer.sorted_blocks():
            if exclude_id and block.block_id == exclude_id:
                continue
            if block.start > cursor:
                intervals.append((cursor, block.start))
            cursor = max(cursor, block.end)
        if cursor < data_len:
            intervals.append((cursor, data_len))
        return intervals

    def _place_block_start(
        self,
        layer: PipelineLayer,
        span: int,
        raw_start: int,
        data_len: int,
        exclude_id: str = "",
    ) -> Optional[int]:
        valid_intervals = []
        for gap_start, gap_end in self._block_intervals(layer, data_len, exclude_id):
            if gap_end - gap_start >= span:
                valid_intervals.append((gap_start, gap_end - span))
        if not valid_intervals:
            return None

        best_start = valid_intervals[0][0]
        best_distance = None
        for start_min, start_max in valid_intervals:
            candidate = min(max(raw_start, start_min), start_max)
            distance = abs(candidate - raw_start)
            if best_distance is None or distance < best_distance:
                best_start = candidate
                best_distance = distance
        return best_start

    def _add_block_with_tool(self, data_id: int, channel_name: str, layer_id: str, start: int, tool_id: Optional[str]):
        if tool_id is None:
            QMessageBox.information(self, "提示", "请将“数据处理工具”拖到 step 轨道。")
            return
        tool_id = self._canonical_tool_id(tool_id)
        if tool_id not in TOOL_META:
            QMessageBox.information(self, "提示", "该工具暂不支持单通道 Pipeline block。")
            return

        found = self._find_layer(data_id, channel_name, layer_id)
        if found is None:
            return
        pipeline, layer = found
        data_len = self._pipeline_data_len(pipeline)
        if data_len <= 0:
            QMessageBox.information(self, "提示", "当前 channel 在公共时间窗口内没有样点。")
            return

        span = min(
            data_len,
            max(1, min(2000, max(200, data_len // 10))),
        )
        snapped_start, _guides = self._snap.snap_interval(start, start + span, px_per_point(self._zoom), data_len)
        fitted_start = self._place_block_start(layer, span, snapped_start, data_len)
        if fitted_start is None:
            QMessageBox.information(self, "提示", "当前 step 没有足够空间放置新的 block。")
            return

        block = PipelineBlock(
            tool_id=tool_id,
            start=fitted_start,
            end=fitted_start + span,
            params=self._default_params_for_tool(tool_id),
            status="configured" if TOOL_META[tool_id].get("kind") == "param" else "empty",
        )
        layer.blocks.append(block)
        self.selected_block_id = block.block_id
        self._refresh_selection()

    def _on_add_block_requested(self, data_id: int, channel_name: str, layer_id: str, start: int):
        del data_id, channel_name, layer_id, start
        QMessageBox.information(self, "提示", "请将“数据处理工具”拖到 step 轨道。")

    def _on_tool_dropped(self, data_id: int, channel_name: str, layer_id: str, tool_id: str, start: int):
        canonical_tool_id = self._canonical_tool_id(tool_id)
        self._add_block_with_tool(data_id, channel_name, layer_id, start, canonical_tool_id)

    def drop_tool_at_global_position(self, tool_id: str, global_pos: QPoint) -> bool:
        """Route a workspace-level tool drop to the track beneath the cursor."""
        target = QApplication.widgetAt(global_pos)
        while target is not None and not isinstance(target, PipelineTrackWidget):
            target = target.parentWidget()
        if not isinstance(target, PipelineTrackWidget) or target._owner() is not self:
            return False

        local_pos = target.mapFromGlobal(global_pos)
        self._on_tool_dropped(
            target.pipeline.data_id,
            target.pipeline.channel,
            target.layer.layer_id,
            tool_id,
            target._px_to_pts(local_pos.x()),
        )
        return True

    def _on_block_selected(self, block_id: str):
        self.selected_block_id = block_id
        self._refresh_selection()

    def _on_block_removed(self, block_id: str):
        for pipeline in self.pipelines:
            if pipeline.remove_block(block_id):
                break
        if self.selected_block_id == block_id:
            self.selected_block_id = ""
        self._broadcast_guides([])
        self._refresh_all_pipeline_statuses()
        self._refresh_selection()

    def _broadcast_guides(self, guides: list[int]):
        for lane in self._lane_widgets:
            lane.set_guides(guides)

    def _refresh_lane_tracks(self):
        for lane in self._lane_widgets:
            lane.refresh_tracks()

    def _invalidate_block_cache(self, block: PipelineBlock, reason: str):
        self._plugin_tasks.pop(block.block_id, None)
        block.input_slice = None
        block.output_slice = None
        if tool_registry.get_plugin(block.tool_id) is not None and block.params:
            block.status = "stale"
        elif TOOL_META.get(block.tool_id, {}).get("kind") == "interactive":
            block.status = "empty"
        else:
            block.status = "configured" if block.params else "empty"
        block.last_summary = reason

    def _on_block_geometry_changed(self, block_id: str):
        found = self._find_block(block_id)
        if found is None:
            return
        _pipeline, _layer, block = found
        reason = "range/lane changed; reopen editor to refresh result"
        self._invalidate_block_cache(block, reason)
        self.selected_block_id = block.block_id
        self._refresh_all_pipeline_statuses()
        found = self._find_block(block_id)
        if found is not None:
            _pipeline, _layer, block = found
            block.last_summary = reason
        self._refresh_selection()

    def _make_preview_channel(
        self,
        data_id: int,
        channel_name: str,
        samples: np.ndarray,
        start: int,
        end: int,
    ) -> Optional[PreviewChannelSlice]:
        if self.data_manager is None:
            return None
        data = self.data_manager.get(data_id)
        if data is None:
            return None
        if not data.has_channel(channel_name):
            return None
        base_slice = self._channel_slice(data_id, channel_name)

        start = max(0, min(len(samples), start))
        end = max(start, min(len(samples), end))
        sliced = np.asarray(samples[start:end], dtype=np.float64)
        index = base_slice.index[start:end]
        series = pd.Series(sliced, index=index)
        return PreviewChannelSlice(channel_name, series)

    def _register_editor_dialog(self, dialog: QDialog):
        self._editor_dialogs.append(dialog)
        dialog.finished.connect(lambda _result, current=dialog: self._drop_editor_dialog(current))

    def _drop_editor_dialog(self, dialog: QDialog):
        self._editor_dialogs = [existing for existing in self._editor_dialogs if existing is not dialog]

    def _open_interactive_editor(
        self,
        pipeline: ChannelPipeline,
        layer: PipelineLayer,
        block: PipelineBlock,
    ):
        input_snapshot = self._compose_channel_preview(
            pipeline.data_id,
            pipeline.channel,
            stop_before_block_id=block.block_id,
            update_status=False,
        )
        range_start = max(0, min(input_snapshot.size, block.start))
        range_end = max(range_start, min(input_snapshot.size, block.end))
        preview_channel = self._make_preview_channel(
            pipeline.data_id,
            pipeline.channel,
            input_snapshot,
            range_start,
            range_end,
        )
        if preview_channel is None:
            return

        dialog = QDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.setWindowTitle(
            f"{TOOL_META.get(block.tool_id, {}).get('name', block.tool_id)}"
            f" | #{pipeline.data_id} {pipeline.channel} | {layer.label}"
        )
        dialog.resize(1200, 720)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)

        tool_id = self._canonical_tool_id(block.tool_id)
        editor = tool_registry.create_pipeline_editor(
            tool_id, preview_channel, dialog, params=block.params,
        )
        if editor is None:
            dialog.deleteLater()
            return

        block_id = block.block_id
        source_snapshot = input_snapshot

        def _on_editor_result(_channel, result):
            ok = self._on_interactive_result(
                block_id, source_snapshot, range_start, range_end, result,
            )
            if ok:
                dialog.close()

        result_signal = getattr(editor, "result_signal", None)
        if result_signal is None:
            dialog.deleteLater()
            return
        result_signal.connect(_on_editor_result)
        params_applied = getattr(editor, "params_applied", None)
        if params_applied is not None:
            params_applied.connect(
                lambda values, _broadcast, current_id=block_id:
                self._on_plugin_params_applied(current_id, values)
            )

        # 同步信号：交互式编辑器勾选"应用到所有通道/所有数据集"时触发
        channel_sync = getattr(editor, "channel_sync_requested", None)
        if channel_sync is not None:
            channel_sync.connect(
                lambda _did, _channel, _result, current_id=block_id:
                self._on_editor_channel_sync(current_id)
            )
        region_sync = getattr(editor, "region_sync_requested", None)
        if region_sync is not None:
            region_sync.connect(
                lambda _params, current_id=block_id:
                self._on_editor_region_sync(current_id)
            )

        layout.addWidget(editor)
        self._register_editor_dialog(dialog)
        dialog.show()

    def _open_param_editor(
        self,
        pipeline: ChannelPipeline,
        layer: PipelineLayer,
        block: PipelineBlock,
    ):
        plugin = tool_registry.get_plugin(block.tool_id)
        if plugin is not None:
            dialog = PluginParamDialog(plugin.info, block.params, self)
        else:
            dialog = PipelineParamDialog(
                f"#{pipeline.data_id} {pipeline.channel}", layer.label, block, self
            )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        block.params = dialog.params
        block.status = "configured"
        block.last_summary = f"params: {block.params}"
        self._refresh_all_pipeline_statuses()
        self._refresh_selection()

    def _on_plugin_params_applied(self, block_id: str, params: object) -> None:
        found = self._find_block(block_id)
        if found is None or not isinstance(params, dict):
            return
        _pipeline, _layer, block = found
        block.params = dict(params)
        block.status = "configured"

    def _on_block_double_clicked(self, block_id: str):
        found = self._find_block(block_id)
        if found is None:
            return

        pipeline, layer, block = found
        tool_kind = TOOL_META.get(block.tool_id, {}).get("kind")
        if tool_kind == "interactive":
            self._open_interactive_editor(pipeline, layer, block)
        else:
            self._open_param_editor(pipeline, layer, block)

    def _on_interactive_result(
        self,
        block_id: str,
        source_snapshot: np.ndarray,
        range_start: int,
        range_end: int,
        result,
    ) -> bool:
        found = self._find_block(block_id)
        if found is None:
            return False

        _pipeline, _layer, block = found
        result_array = np.asarray(result, dtype=np.float64)
        expected_size = max(0, range_end - range_start)
        if result_array.ndim != 1 or result_array.size != expected_size:
            QMessageBox.warning(self, "结果错误", "processor 返回的数据长度与 block 区间长度不一致。")
            return False

        # 只缓存 block 区间内的输入/输出切片，不复制整条通道
        block.input_slice = np.array(source_snapshot[range_start:range_end], copy=True)
        block.output_slice = np.array(result_array, copy=True)
        block.status = "ready"
        block.last_summary = f"cached {result_array.size} pts in [{range_start}, {range_end})"
        if self._pending_lane_rebuild:
            self._pending_lane_rebuild = False
            self._rebuild_lanes()
        self._refresh_all_pipeline_statuses()
        self.selected_block_id = block.block_id
        self._refresh_selection()
        return True

    # ─── 编辑器同步（通道同步 / 跨数据集区域同步）─────────────

    def _synced_block_range(
        self,
        source_pipeline: ChannelPipeline,
        source_block: PipelineBlock,
        target_pipeline: ChannelPipeline,
    ) -> Optional[tuple[int, int]]:
        """Map a block's inclusive timestamp span to a target channel slice."""
        source = self._channel_slice(source_pipeline.data_id, source_pipeline.channel)
        target = self._channel_slice(target_pipeline.data_id, target_pipeline.channel)
        if source.empty or target.empty:
            return None

        source_start = max(0, min(source_block.start, len(source)))
        source_end = max(source_start, min(source_block.end, len(source)))
        if source_end <= source_start:
            return None

        source_index = source.index
        target_index = target.index
        start_time = source_index[source_start]
        end_time = source_index[source_end - 1]
        target_start = int(target_index.searchsorted(start_time, side="left"))
        target_end = int(target_index.searchsorted(end_time, side="right"))
        target_start = max(0, min(target_start, len(target)))
        target_end = max(target_start, min(target_end, len(target)))
        if target_end <= target_start:
            return None
        return target_start, target_end

    def _synced_block_params(
        self,
        source_pipeline: ChannelPipeline,
        source_block: PipelineBlock,
        target_pipeline: ChannelPipeline,
        target_range: tuple[int, int],
    ) -> Optional[dict]:
        """Copy serializable params, translating manual spike picks by time."""
        params = deepcopy(source_block.params)
        if source_block.tool_id != "remove_spike":
            return params

        source = self._channel_slice(source_pipeline.data_id, source_pipeline.channel)
        target = self._channel_slice(target_pipeline.data_id, target_pipeline.channel)
        source_start = max(0, min(source_block.start, len(source)))
        source_end = max(source_start, min(source_block.end, len(source)))
        picked = []
        for value in params.get("indices", []):
            try:
                position = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= position < source_end - source_start:
                picked.append(position)
        if not picked:
            return None

        picked_times = source.index[source_start + np.asarray(picked, dtype=int)]
        target_start, target_end = target_range
        target_index = target.index[target_start:target_end]
        params["indices"] = np.flatnonzero(target_index.isin(picked_times)).astype(int).tolist()
        if not params["indices"]:
            return None
        return params

    def _clone_block_to_pipelines(
        self,
        source_block_id: str,
        target_pipelines: list[ChannelPipeline],
        scope_label: str,
    ) -> int:
        """Stage synchronized blocks without writing any EMData channel.

        A Pipeline confirmation persists only a block definition.  In
        particular, an editor's direct ``channel_sync`` signal must not
        turn into an immediate NaN write here: every target receives its own
        block, cache, and preview computation instead.
        """
        found = self._find_block(source_block_id)
        if found is None:
            return 0
        source_pipeline, source_layer, source_block = found
        if not source_block.params:
            return 0
        if source_block.tool_id == "remove_spike" and not source_block.params.get("indices"):
            return 0
        if source_block.tool_id == "remove_region" and not source_block.params.get("time_ranges"):
            return 0

        source_layer_index = source_pipeline.layers.index(source_layer)
        cloned = 0
        for target_pipeline in target_pipelines:
            if target_pipeline is source_pipeline:
                continue
            target_range = self._synced_block_range(
                source_pipeline, source_block, target_pipeline,
            )
            if target_range is None:
                continue
            params = self._synced_block_params(
                source_pipeline, source_block, target_pipeline, target_range,
            )
            if params is None:
                continue

            while len(target_pipeline.layers) <= source_layer_index:
                target_pipeline.add_layer()
                self._pending_lane_rebuild = True
            target_layer = target_pipeline.layers[source_layer_index]
            start, end = target_range
            target_layer.blocks.append(PipelineBlock(
                tool_id=source_block.tool_id,
                start=start,
                end=end,
                params=params,
                status="configured",
                last_summary=(
                    f"{scope_label} from #{source_pipeline.data_id} "
                    f"{source_pipeline.channel}; awaiting preview"
                ),
            ))
            cloned += 1
        return cloned

    def _on_editor_channel_sync(
        self,
        source_block_id: str,
    ) -> None:
        """Stage a same-dataset block clone for each channel in this Pipeline."""
        found = self._find_block(source_block_id)
        if found is None:
            return
        source_pipeline, _source_layer, _source_block = found
        targets = [
            pipeline for pipeline in self.pipelines
            if pipeline.data_id == source_pipeline.data_id
        ]
        self._clone_block_to_pipelines(
            source_block_id,
            targets,
            "synchronized to Pipeline channels",
        )

    def _on_editor_region_sync(self, source_block_id: str) -> None:
        """Stage a cross-dataset clone for channels already in this Pipeline."""
        found = self._find_block(source_block_id)
        if found is None:
            return
        source_pipeline, _source_layer, _source_block = found
        targets = [
            pipeline for pipeline in self.pipelines
            if pipeline.data_id != source_pipeline.data_id
        ]
        self._clone_block_to_pipelines(
            source_block_id,
            targets,
            "synchronized to Pipeline datasets",
        )

    def _refresh_all_pipeline_statuses(self):
        previews: dict[tuple[int, str], np.ndarray] = {}
        for pipeline in self.pipelines:
            previews[pipeline.ref] = self._compose_channel_preview(
                pipeline.data_id,
                pipeline.channel,
                update_status=True,
            )
        for lane in self._lane_widgets:
            preview = previews.get(lane.pipeline.ref)
            if preview is not None:
                lane.samples = np.array(preview, copy=True)
                if lane.overview_widget is not None:
                    lane.overview_widget.set_samples(preview)
            lane.refresh()

    @staticmethod
    def _input_fingerprint(values: np.ndarray) -> tuple[int, int]:
        contiguous = np.ascontiguousarray(values, dtype=np.float64)
        return contiguous.size, hash(contiguous.tobytes())

    def _schedule_plugin_block(self, block: PipelineBlock, current_input: np.ndarray) -> None:
        plugin = tool_registry.get_plugin(block.tool_id)
        if plugin is None or block.block_id in self._plugin_tasks:
            return
        found = self._find_block(block.block_id)
        if found is None:
            return
        pipeline, _layer, current_block = found
        base_series = self._channel_slice(pipeline.data_id, pipeline.channel)
        start = max(0, min(current_block.start, len(base_series)))
        end = max(start, min(current_block.end, len(base_series)))
        values = np.asarray(current_input, dtype=np.float64).copy()
        if values.size != end - start:
            return
        series = pd.Series(
            values,
            index=base_series.index[start:end],
            name=pipeline.channel,
        )
        if len(series.index) > 1:
            dt = (series.index[1] - series.index[0]).total_seconds()
        else:
            dt = 0.0
        channel_block = ChannelBlock(
            pipeline.channel,
            series,
            ChannelMeta(sample_rate=(1.0 / dt if dt > 0 else 0.0)),
        )
        fingerprint = self._input_fingerprint(values)
        self._plugin_generation += 1
        generation = self._plugin_generation
        self._plugin_tasks[block.block_id] = generation
        block.last_summary = "processing plugin…"

        bridge = _PluginTaskBridge(
            (block.block_id, generation, fingerprint),
            self,
        )
        self._plugin_bridges[block.block_id] = bridge
        bridge.done.connect(self._on_plugin_task_done)
        bridge.failed.connect(self._on_plugin_task_failed)

        run_in_pool(
            "compute",
            run_plugin_process,
            plugin,
            channel_block,
            dict(block.params),
            on_done=bridge.relay_done,
            on_error=bridge.relay_failed,
            owner=bridge,
        )

    def _on_plugin_task_done(self, envelope) -> None:
        metadata, payload = envelope
        block_id, expected, fp = metadata
        if self._plugin_tasks.get(block_id) != expected:
            return
        self._plugin_tasks.pop(block_id, None)
        self._plugin_bridges.pop(block_id, None)
        located = self._find_block(block_id)
        if located is None:
            return
        target_pipeline, _target_layer, target = located
        result, completed_params = payload
        latest_snapshot = self._compose_channel_preview(
            target_pipeline.data_id,
            target_pipeline.channel,
            stop_before_block_id=block_id,
            update_status=False,
        )
        latest_input = latest_snapshot[target.start:target.end]
        if self._input_fingerprint(latest_input) != fp:
            target.status = "stale"
            target.last_summary = "upstream changed while processing; rerunning"
            self._refresh_all_pipeline_statuses()
            return
        target.params = dict(completed_params)
        target.input_slice = np.array(latest_input, copy=True)
        target.output_slice = result.series.to_numpy(dtype=np.float64, copy=True)
        target.status = "ready"
        target.last_summary = f"processed {target.output_slice.size} pts"
        self._refresh_all_pipeline_statuses()
        self._refresh_selection()

    def _on_plugin_task_failed(self, envelope) -> None:
        metadata, exc = envelope
        block_id, expected, _fp = metadata
        if self._plugin_tasks.get(block_id) != expected:
            return
        self._plugin_tasks.pop(block_id, None)
        self._plugin_bridges.pop(block_id, None)
        located = self._find_block(block_id)
        if located is None:
            return
        located[2].status = "error"
        located[2].last_summary = repr(exc)
        self._refresh_selection()
        for lane in self._lane_widgets:
            lane.refresh()

    @staticmethod
    def _replay_cached_edit(
        cached_input: np.ndarray,
        cached_output: np.ndarray,
        current_input: np.ndarray,
    ) -> np.ndarray:
        """把编辑器确认过的修改重放到新的上游输入。

        交互编辑器返回的是确认前后的切片，而不是可序列化算法参数。这里把
        两者的差异解释为一个确定性的编辑操作：有限值修改保存为加性校正，
        删除操作保存为 NaN，其他非有限值转换保存为绝对替换。未被编辑器
        修改的位置始终沿用新的上游输入。
        """
        cached_input = np.asarray(cached_input, dtype=np.float64)
        cached_output = np.asarray(cached_output, dtype=np.float64)
        current_input = np.asarray(current_input, dtype=np.float64)
        if not (
            cached_input.ndim == cached_output.ndim == current_input.ndim == 1
            and cached_input.shape == cached_output.shape == current_input.shape
        ):
            raise ValueError("cached edit slices must be one-dimensional and equal length")

        result = np.array(current_input, copy=True)
        same = (cached_input == cached_output) | (
            np.isnan(cached_input) & np.isnan(cached_output)
        )
        changed = ~same
        additive = changed & np.isfinite(cached_input) & np.isfinite(cached_output)
        result[additive] = (
            current_input[additive]
            + cached_output[additive]
            - cached_input[additive]
        )

        # NaN 删除、NaN→数值恢复以及 inf 等少见转换不能表示为有限差值，
        # 直接重放编辑器确认的目标值。
        absolute = changed & ~additive
        result[absolute] = cached_output[absolute]
        return result

    def _apply_block_preview(
        self,
        block: PipelineBlock,
        current_snapshot: np.ndarray,
        update_status: bool,
    ) -> np.ndarray:
        """把单个 block 的缓存修改叠加到当前快照上。"""
        tool_kind = TOOL_META.get(block.tool_id, {}).get("kind")
        n = current_snapshot.size
        start = max(0, min(block.start, n))
        end = max(start, min(block.end, n))
        region_len = end - start

        has_cache = block.input_slice is not None and block.output_slice is not None
        if not has_cache:
            if update_status:
                if block.status == "error":
                    return current_snapshot
                if tool_kind == "interactive" and block.status == "empty":
                    block.status = "empty"
                    block.last_summary = "double click to configure"
                else:
                    block.status = "configured" if tool_kind == "param" or block.params else "empty"
                    if block.status == "configured":
                        self._schedule_plugin_block(block, current_snapshot[start:end])
            return current_snapshot

        if block.input_slice.size != region_len or block.output_slice.size != region_len:
            if update_status:
                block.status = "stale"
                block.last_summary = "block range changed; refresh result"
                self._schedule_plugin_block(block, current_snapshot[start:end])
            return current_snapshot

        current_input = current_snapshot[start:end]
        cache_matches = np.array_equal(current_input, block.input_slice, equal_nan=True)
        output_slice = block.output_slice
        if not cache_matches:
            # Compatibility for pre-plugin pipeline snapshots created by the
            # previous WIP implementation (they contain no serialized params).
            if not block.params:
                output_slice = self._replay_cached_edit(
                    block.input_slice,
                    block.output_slice,
                    current_input,
                )
                if update_status:
                    block.input_slice = np.array(current_input, copy=True)
                    block.output_slice = np.array(output_slice, copy=True)
                    block.status = "ready"
                    block.last_summary = f"replayed {output_slice.size} pts on legacy block"
                out = np.array(current_snapshot, copy=True)
                out[start:end] = output_slice
                return out
            if update_status:
                block.status = "stale"
                block.last_summary = "upstream changed; rerunning plugin"
                self._schedule_plugin_block(block, current_input)
            return current_snapshot

        if update_status:
            block.status = "ready"
            if cache_matches:
                block.last_summary = f"cached {block.output_slice.size} pts"
        out = np.array(current_snapshot, copy=True)
        out[start:end] = output_slice
        return out

    def _compose_channel_preview(
        self,
        data_id: int,
        channel_name: str,
        stop_before_block_id: str = "",
        update_status: bool = False,
    ) -> np.ndarray:
        pipeline = self._find_pipeline(data_id, channel_name)
        if pipeline is None:
            return np.array([], dtype=np.float64)

        current_snapshot = self._pipeline_baseline(pipeline)
        for layer in pipeline.layers:
            for block in layer.sorted_blocks():
                if stop_before_block_id and block.block_id == stop_before_block_id:
                    return np.array(current_snapshot, copy=True)
                current_snapshot = self._apply_block_preview(block, current_snapshot, update_status)
        return np.array(current_snapshot, copy=True)

    def _collect_preview_results(self, update_status: bool) -> dict[tuple[int, str], np.ndarray]:
        results: dict[tuple[int, str], np.ndarray] = {}
        for pipeline in self.pipelines:
            results[pipeline.ref] = self._compose_channel_preview(
                pipeline.data_id,
                pipeline.channel,
                update_status=update_status,
            )
        return results

    def _pipeline_status_counts(self, pipeline: ChannelPipeline) -> dict[str, int]:
        counts = {key: 0 for key in BLOCK_STATUS_STYLE}
        for layer in pipeline.layers:
            for block in layer.blocks:
                counts[block.status] = counts.get(block.status, 0) + 1
        return counts

    def _show_preview_summary(self):
        if not self.pipelines:
            return

        self._refresh_all_pipeline_statuses()
        self._refresh_selection()

        lines = [
            "Preview 语义：每个 channel 从 baseline 开始，按 layer 顺序、再按 block 起点顺序串行叠加。",
            "已确认 block 会在上游变化后自动重放；未确认或区间损坏的 block 不会被提交。",
            "",
        ]
        for pipeline in self.pipelines:
            counts = self._pipeline_status_counts(pipeline)
            lines.append(
                f"#{pipeline.data_id} {pipeline.channel}: "
                f"ready={counts.get('ready', 0)}, "
                f"stale={counts.get('stale', 0)}, "
                f"pending={counts.get('empty', 0)}, "
                f"configured={counts.get('configured', 0)}"
            )

        QMessageBox.information(self, "Preview", "\n".join(lines))

    def _export_recipe(self) -> None:
        if not self.pipelines:
            QMessageBox.information(self, "Export Recipe", "当前没有可导出的 pipeline。")
            return
        pipeline = self.pipelines[0]
        if self.selected_block_id:
            found = self._find_block(self.selected_block_id)
            if found is not None:
                pipeline = found[0]
        data = self._data_for(pipeline.data_id)
        if data is None or not data.has_channel(pipeline.channel):
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出 Pipeline Recipe",
            f"{pipeline.channel}-recipe.toml",
            "MTPro Recipe (*.toml)",
        )
        if not path:
            return
        if not path.lower().endswith(".toml"):
            path += ".toml"
        recipe = recipe_from_pipeline(
            pipeline,
            data.get_channel_series(pipeline.channel).index,
            f"#{pipeline.data_id} {pipeline.channel}",
        )
        try:
            save_recipe(recipe, path)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", repr(exc))

    def _reset_pipeline_after_commit(self, pipeline: ChannelPipeline):
        for layer in pipeline.layers:
            for block in layer.blocks:
                block.input_slice = None
                block.output_slice = None
                if TOOL_META.get(block.tool_id, {}).get("kind") == "interactive":
                    block.status = "empty"
                    block.last_summary = "committed into baseline; reopen editor for next pass"
                else:
                    block.status = "configured" if block.params else "empty"
                    block.last_summary = "committed into baseline; rerun to regenerate"

    def _unready_blocks(self) -> list[tuple[ChannelPipeline, PipelineLayer, PipelineBlock]]:
        result: list[tuple[ChannelPipeline, PipelineLayer, PipelineBlock]] = []
        for pipeline in self.pipelines:
            for layer in pipeline.layers:
                for block in layer.blocks:
                    if block.status != "ready":
                        result.append((pipeline, layer, block))
        return result

    def _commit_preview(self) -> None:
        if not self.pipelines:
            return

        preview_results = self._collect_preview_results(update_status=True)
        unready = self._unready_blocks()
        if unready:
            details = [
                f"#{pipeline.data_id} {pipeline.channel} / {layer.label} / "
                f"{TOOL_META.get(block.tool_id, {}).get('name', block.tool_id)}: "
                f"{BLOCK_STATUS_STYLE.get(block.status, {}).get('label', block.status)}"
                for pipeline, layer, block in unready[:8]
            ]
            if len(unready) > len(details):
                details.append(f"……另有 {len(unready) - len(details)} 个 block")
            QMessageBox.warning(
                self,
                "Pipeline 未完成",
                "以下 block 尚未产生有效结果，已取消提交：\n" + "\n".join(details),
            )
            self._refresh_selection()
            return

        changed_channels: list[str] = []

        for pipeline in self.pipelines:
            data = self._data_for(pipeline.data_id)
            preview_snapshot = preview_results.get(pipeline.ref)
            if data is None or not data.has_channel(pipeline.channel) or preview_snapshot is None:
                continue
            target_series = self._channel_slice(pipeline.data_id, pipeline.channel)
            if target_series.shape[0] != preview_snapshot.shape[0]:
                continue

            current_snapshot = target_series.to_numpy(dtype=np.float64, copy=True)
            pipeline.baseline_snapshot = np.array(preview_snapshot, copy=True)
            if np.array_equal(current_snapshot, preview_snapshot, equal_nan=True):
                continue

            updated_series = data.get_channel_series(pipeline.channel).copy()
            updated_series.loc[target_series.index] = preview_snapshot
            data.set_channel_series(pipeline.channel, updated_series)
            changed_channels.append(f"#{pipeline.data_id} {pipeline.channel}")
            self._reset_pipeline_after_commit(pipeline)
            self.change_finished_signal.emit(pipeline.data_id, pipeline.channel)

        if not changed_channels:
            QMessageBox.information(self, "Commit", "没有可提交的新 preview 结果。")
            self._refresh_all_pipeline_statuses()
            return

        self._rebuild_lanes()
        self._refresh_all_pipeline_statuses()
        self._refresh_selection()
        QMessageBox.information(
            self,
            "Commit",
            "已写回 channel:\n"
            + "\n".join(changed_channels)
            + "\n\n处理结果已合并到 baseline；继续处理时请重新打开对应 block。",
        )

    def _refresh_selection(self):
        found = self._find_block(self.selected_block_id)
        if found is None:
            for lane in self._lane_widgets:
                lane.set_selected("")
            if self.pipeline_hint is not None:
                self.pipeline_hint.setText(
                    "将左侧“数据处理工具”拖到某个 step 轨道以添加 block。"
                )
            return

        pipeline, layer, block = found
        for lane in self._lane_widgets:
            lane.set_selected(block.block_id)

        if self.pipeline_hint is not None:
            meta = TOOL_META.get(block.tool_id, {})
            self.pipeline_hint.setText(
                f"#{pipeline.data_id} {pipeline.channel} / {layer.label} / "
                f"{meta.get('name', block.tool_id)} / {block.status}"
            )

    def _find_block(
        self,
        block_id: str,
    ) -> tuple[ChannelPipeline, PipelineLayer, PipelineBlock] | None:
        if not block_id:
            return None
        for pipeline in self.pipelines:
            found = pipeline.find_block(block_id)
            if found is not None:
                layer, block = found
                return pipeline, layer, block
        return None

    def block_count(self) -> int:
        return sum(
            len(layer.blocks)
            for pipeline in self.pipelines
            for layer in pipeline.layers
        )
