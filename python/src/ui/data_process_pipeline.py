from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from PySide6.QtCore import (
    QByteArray,
    QItemSelectionModel,
    QMimeData,
    QModelIndex,
    QPoint,
    QRect,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QDrag,
    QFont,
    QPainter,
    QPalette,
    QPen,
    QStandardItem,
    QStandardItemModel,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
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
    QTextEdit,
    QTreeView,
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
from ui.tools import registry as tool_registry


LAYER_H = 34
HEADER_W = 120
CHANNEL_HEADER_H = LAYER_H * 2
ADD_ROW_H = 24
RULER_H = 24
HANDLE_W = 7
MIN_BLOCK_SPAN = 100
OVERVIEW_MAX_BUCKETS = 65536
PIPELINE_TOOL_MIME = "application/x-pipeline-tool"


def px_per_point(zoom: float) -> float:
    return max(0.02, zoom * 8 / 1000)


def points_to_pixels(points: int, zoom: float) -> float:
    return points * px_per_point(zoom)


def _is_pipeline_capable(spec) -> bool:
    if spec.kind == "interactive":
        return spec.pipeline_editor_factory is not None
    if spec.kind == "param":
        return bool(spec.fields)
    return False


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


BLOCK_STATUS_STYLE: Dict[str, Dict[str, str]] = {
    "empty": {"accent": "#8b8f97", "label": "待处理"},
    "configured": {"accent": "#7e8b99", "label": "已配置"},
    "ready": {"accent": "#778f7c", "label": "就绪"},
    "stale": {"accent": "#9a7b7b", "label": "需刷新"},
}

TOOL_KIND_LABELS: Dict[str, str] = {
    "interactive": "交互工具",
    "param": "参数工具",
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


class PipelineToolTreeView(QTreeView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setItemsExpandable(True)
        self.setUniformRowHeights(True)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragEnabled(True)

    def startDrag(self, supported_actions):
        index = self.currentIndex()
        if not index.isValid():
            return

        tool_id = index.data(Qt.ItemDataRole.UserRole)
        if not tool_id:
            return

        mime_data = QMimeData()
        mime_data.setData(PIPELINE_TOOL_MIME, QByteArray(str(tool_id).encode()))
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec(supported_actions, Qt.DropAction.CopyAction)


class ChannelOverviewWidget(QWidget):
    def __init__(self, samples: np.ndarray, data_len: int, parent=None):
        super().__init__(parent)
        self._samples = np.asarray(samples, dtype=np.float64)
        self._data_len = max(1, data_len)
        self._zoom = 6.0
        self._envelope_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.setFixedHeight(CHANNEL_HEADER_H)
        self.setMinimumWidth(600)

    def set_zoom(self, zoom: float):
        self._zoom = zoom
        self.setMinimumWidth(int(points_to_pixels(self._data_len, zoom)) + 40)
        self.update()

    def set_samples(self, samples: np.ndarray):
        self._samples = np.asarray(samples, dtype=np.float64)
        self._envelope_cache.clear()
        self.update()

    def _get_envelope(self, bucket_count: int) -> tuple[np.ndarray, np.ndarray]:
        bucket_count = max(1, bucket_count)
        cached = self._envelope_cache.get(bucket_count)
        if cached is not None:
            return cached

        samples = self._samples
        if samples.size == 0:
            mins = np.zeros(bucket_count, dtype=np.float64)
            maxs = np.zeros(bucket_count, dtype=np.float64)
            self._envelope_cache[bucket_count] = (mins, maxs)
            return mins, maxs

        if bucket_count >= samples.size:
            finite = np.isfinite(samples)
            clean = np.where(finite, samples, np.nan)
            mins = np.nan_to_num(clean.copy(), nan=0.0)
            maxs = np.nan_to_num(clean.copy(), nan=0.0)
            self._envelope_cache[bucket_count] = (mins, maxs)
            return mins, maxs

        edges = np.linspace(0, samples.size, bucket_count + 1, dtype=int)
        mins = np.zeros(bucket_count, dtype=np.float64)
        maxs = np.zeros(bucket_count, dtype=np.float64)
        for index in range(bucket_count):
            start = edges[index]
            end = edges[index + 1]
            segment = samples[start:end]
            finite_segment = segment[np.isfinite(segment)]
            if finite_segment.size == 0:
                mins[index] = 0.0
                maxs[index] = 0.0
                continue
            mins[index] = float(finite_segment.min())
            maxs[index] = float(finite_segment.max())

        self._envelope_cache[bucket_count] = (mins, maxs)
        return mins, maxs

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.color(QPalette.ColorRole.Base))

        if self.width() <= 2:
            painter.end()
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 1))

        requested_buckets = max(1, min(self.width() - 2, OVERVIEW_MAX_BUCKETS))
        mins, maxs = self._get_envelope(requested_buckets)
        bucket_count = max(1, mins.size)

        all_values = np.concatenate((mins, maxs))
        finite = all_values[np.isfinite(all_values)]
        if finite.size == 0:
            painter.end()
            return

        y_min = float(finite.min())
        y_max = float(finite.max())
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

        width_span = max(1, self.width() - 3)
        for index in range(bucket_count):
            x_pos = 1 + int(index * width_span / max(1, bucket_count - 1))
            painter.drawLine(x_pos, to_y(mins[index]), x_pos, to_y(maxs[index]))

        painter.setPen(palette.color(QPalette.ColorRole.Mid))
        mid_y = self.height() // 2
        painter.drawLine(0, mid_y, self.width(), mid_y)
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
        if event.mimeData().hasFormat(PIPELINE_TOOL_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasFormat(PIPELINE_TOOL_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        if not event.mimeData().hasFormat(PIPELINE_TOOL_MIME):
            event.ignore()
            return

        tool_id = bytes(event.mimeData().data(PIPELINE_TOOL_MIME)).decode()
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
        parent=None,
    ):
        super().__init__(parent)
        self.pipeline = pipeline
        self.samples = np.asarray(samples, dtype=np.float64)
        self.data_len = data_len
        self.snap = snap
        self.zoom = 6.0
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
        self.selected_tool_id: Optional[str] = None
        self.selected_block_id: str = ""
        self.pipelines: list[ChannelPipeline] = []
        self._lane_widgets: list[ChannelPipelineWidget] = []
        self._editor_dialogs: list[QDialog] = []
        self._zoom = 6.0
        self._snap = SnapEngine()
        self._ruler_widget: Optional[RulerWidget] = None
        self.tool_tree: Optional[PipelineToolTreeView] = None
        self.tool_model: Optional[QStandardItemModel] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        left_panel = QWidget(self)
        left_panel.setFixedWidth(210)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        title = QLabel("Pipeline Tools")
        left_layout.addWidget(title)

        self.tool_tree = PipelineToolTreeView(left_panel)
        self.tool_model = self._build_tool_model(self.tool_tree)
        self.tool_tree.setModel(self.tool_model)
        self.tool_tree.expandAll()
        self.tool_tree.selectionModel().currentChanged.connect(self._on_tool_changed)
        left_layout.addWidget(self.tool_tree)

        self.info_text = QTextEdit(left_panel)
        self.info_text.setReadOnly(True)
        self.info_text.setPlainText(
            "选择左侧工具，然后点击某个 step 轨道空白处放置 block。\n"
            "交互式 block 双击会打开真实编辑器，参数型 block 双击会打开参数面板。"
        )
        left_layout.addWidget(self.info_text, 1)

        clear_tool_button = QPushButton("clear tool", left_panel)
        clear_tool_button.clicked.connect(self._clear_selected_tool)
        left_layout.addWidget(clear_tool_button)

        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        toolbar = QWidget(right_panel)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 6, 8, 6)
        toolbar_layout.setSpacing(8)

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
        toolbar_layout.addWidget(preview_button)
        toolbar_layout.addWidget(commit_button)
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

        root.addWidget(left_panel)
        root.addWidget(right_panel, 1)

    def _build_tool_model(self, parent: QWidget) -> QStandardItemModel:
        model = QStandardItemModel(parent)
        root_item = model.invisibleRootItem()
        groups: dict[str, QStandardItem] = {}

        for kind, label in TOOL_KIND_LABELS.items():
            group_item = QStandardItem(label)
            group_item.setEditable(False)
            group_item.setSelectable(False)
            group_item.setDragEnabled(False)
            group_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            root_item.appendRow(group_item)
            groups[kind] = group_item

        for tool_id, meta in TOOL_META.items():
            kind = str(meta.get("kind", "param"))
            parent_item = groups[kind]
            tool_item = QStandardItem(str(meta["name"]))
            tool_item.setEditable(False)
            tool_item.setData(tool_id, Qt.ItemDataRole.UserRole)
            tool_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsDragEnabled
            )
            parent_item.appendRow(tool_item)

        return model

    def _tool_id_from_index(self, index: QModelIndex) -> Optional[str]:
        if not index.isValid():
            return None
        value = index.data(Qt.ItemDataRole.UserRole)
        return None if value is None else self._canonical_tool_id(str(value))

    def _tool_index_from_id(self, tool_id: str) -> QModelIndex:
        if self.tool_model is None:
            return QModelIndex()

        tool_id = self._canonical_tool_id(tool_id)
        root_item = self.tool_model.invisibleRootItem()
        for row in range(root_item.rowCount()):
            group_item = root_item.child(row)
            for child_row in range(group_item.rowCount()):
                child_item = group_item.child(child_row)
                if child_item.data(Qt.ItemDataRole.UserRole) == tool_id:
                    return child_item.index()
        return QModelIndex()

    def _set_selected_tool(self, tool_id: Optional[str]):
        self.selected_tool_id = None if tool_id is None else self._canonical_tool_id(tool_id)
        if self.tool_tree is not None:
            if self.selected_tool_id is None:
                self.tool_tree.clearSelection()
                self.tool_tree.setCurrentIndex(QModelIndex())
            else:
                index = self._tool_index_from_id(self.selected_tool_id)
                if index.isValid():
                    self.tool_tree.setCurrentIndex(index)
                    self.tool_tree.selectionModel().select(
                        index,
                        QItemSelectionModel.SelectionFlag.ClearAndSelect
                        | QItemSelectionModel.SelectionFlag.Rows,
                    )
        self._update_tool_info()

    def _canonical_tool_id(self, tool_id: str) -> str:
        return tool_registry.canonical_id(tool_id)

    def _update_tool_info(self):
        if self.selected_tool_id is None:
            self.info_text.setPlainText("请选择一个工具，然后点击或拖到某个 step 轨道中放置 block。")
            return

        meta = TOOL_META.get(self.selected_tool_id, {})
        kind = TOOL_KIND_LABELS.get(str(meta.get("kind", "unknown")), str(meta.get("kind", "unknown")))
        self.info_text.setPlainText(
            f"当前工具：{meta.get('name', self.selected_tool_id)}\n"
            f"类型：{kind}\n"
            "点击轨道空白处或直接拖到轨道中即可放置 block。"
        )

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
                self.scroll_content,
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

    def _on_tool_changed(self, current: QModelIndex, previous: QModelIndex):
        del previous
        self.selected_tool_id = self._tool_id_from_index(current)
        self._update_tool_info()

    def _clear_selected_tool(self):
        self._set_selected_tool(None)

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
            QMessageBox.information(self, "提示", "请先在左侧选择一个工具。")
            return
        tool_id = self._canonical_tool_id(tool_id)

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
        self._add_block_with_tool(data_id, channel_name, layer_id, start, self.selected_tool_id)

    def _on_tool_dropped(self, data_id: int, channel_name: str, layer_id: str, tool_id: str, start: int):
        canonical_tool_id = self._canonical_tool_id(tool_id)
        self._set_selected_tool(canonical_tool_id)
        self._add_block_with_tool(data_id, channel_name, layer_id, start, canonical_tool_id)

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
        self._refresh_selection()

    def _broadcast_guides(self, guides: list[int]):
        for lane in self._lane_widgets:
            lane.set_guides(guides)

    def _refresh_lane_tracks(self):
        for lane in self._lane_widgets:
            lane.refresh_tracks()

    def _invalidate_block_cache(self, block: PipelineBlock, reason: str):
        block.input_slice = None
        block.output_slice = None
        if TOOL_META.get(block.tool_id, {}).get("kind") == "interactive":
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
        editor = tool_registry.create_pipeline_editor(tool_id, preview_channel, dialog)
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

        # 同步信号：交互式编辑器勾选"应用到所有通道/所有数据集"时触发
        slice_index = preview_channel.cts.index
        channel_sync = getattr(editor, "channel_sync_requested", None)
        if channel_sync is not None:
            channel_sync.connect(
                lambda _did, ch_name, result, idx=slice_index:
                self._on_editor_channel_sync(pipeline.data_id, ch_name, result, idx)
            )
        region_sync = getattr(editor, "region_sync_requested", None)
        if region_sync is not None:
            region_sync.connect(self._on_editor_region_sync)

        layout.addWidget(editor)
        self._register_editor_dialog(dialog)
        dialog.show()

    def _open_param_editor(
        self,
        pipeline: ChannelPipeline,
        layer: PipelineLayer,
        block: PipelineBlock,
    ):
        dialog = PipelineParamDialog(f"#{pipeline.data_id} {pipeline.channel}", layer.label, block, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        block.params = dialog.params
        block.status = "configured"
        block.last_summary = f"params: {block.params}"
        self._refresh_selection()

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
        self._refresh_all_pipeline_statuses()
        self.selected_block_id = block.block_id
        self._refresh_selection()
        return True

    # ─── 编辑器同步（通道同步 / 跨数据集区域同步）─────────────

    def _on_editor_channel_sync(
        self,
        data_id: int,
        channel_name: str,
        result_slice,
        slice_index,
    ) -> None:
        """把 block 切片里的 NaN 位置按时间戳同步到同数据集的其他通道。"""
        data = self._data_for(data_id)
        if data is None:
            return
        result_arr = np.asarray(result_slice, dtype=np.float64)
        if result_arr.ndim != 1 or result_arr.size != len(slice_index):
            return
        nan_timestamps = slice_index[np.isnan(result_arr)]
        if len(nan_timestamps) == 0:
            return

        for ch in data.channel_names():
            if ch == channel_name or not data.has_channel(ch):
                continue
            series = data.get_channel_series(ch)
            mask = series.index.isin(nan_timestamps)
            if not mask.any():
                continue
            vals = series.to_numpy(dtype=np.float64, copy=True)
            vals[mask] = np.nan
            data.set_channel_series(
                ch, pd.Series(vals, index=series.index, name=series.name)
            )
            self.change_finished_signal.emit(data_id, ch)

    def _on_editor_region_sync(self, params: dict) -> None:
        """remove_region 跨数据集同步：按时间范围对所有数据集置 NaN。"""
        from core.region_sync import apply_nan_intervals, compute_time_range_intervals

        time_ranges = params.get("time_ranges", [])
        source_id = params.get("source_data_id", -1)
        if not time_ranges or self.data_manager is None:
            return

        for data_id, em_data in self.data_manager.items():
            if data_id == source_id:
                continue
            start_ts = em_data.start_time.timestamp()
            dt_sec = em_data.dt.total_seconds()
            for ch in em_data.channel_names():
                series = em_data.get_channel_series(ch)
                intervals = compute_time_range_intervals(
                    time_ranges, len(series), start_ts, dt_sec
                )
                if not intervals:
                    continue
                vals = apply_nan_intervals(
                    series.to_numpy(dtype=np.float64, copy=True), intervals
                )
                em_data.set_channel_series(
                    ch, pd.Series(vals, index=series.index, name=series.name)
                )
                self.change_finished_signal.emit(data_id, ch)

    def _refresh_all_pipeline_statuses(self):
        for pipeline in self.pipelines:
            self._compose_channel_preview(pipeline.data_id, pipeline.channel, update_status=True)
        for lane in self._lane_widgets:
            lane.refresh()

    def _apply_block_preview(
        self,
        block: PipelineBlock,
        current_snapshot: np.ndarray,
        update_status: bool,
    ) -> np.ndarray:
        """把单个 block 的缓存结果叠加到当前快照上。

        interactive 与 param block 统一走"切片缓存"：block 只存
        ``[start:end]`` 的输入/输出切片。当前快照对应区段与缓存输入一致时
        把输出切片拼回去；否则标记 stale。param block 未执行过时保持
        configured，等待后台执行引擎生成缓存。
        """
        tool_kind = TOOL_META.get(block.tool_id, {}).get("kind")
        n = current_snapshot.size
        start = max(0, min(block.start, n))
        end = max(start, min(block.end, n))
        region_len = end - start

        has_cache = block.input_slice is not None and block.output_slice is not None
        if not has_cache:
            if update_status:
                if tool_kind == "interactive":
                    block.status = "empty"
                    block.last_summary = "double click to edit and cache result"
                else:
                    block.status = "configured" if block.params else "empty"
                    block.last_summary = "param block waiting to run"
            return current_snapshot

        if block.input_slice.size != region_len or block.output_slice.size != region_len:
            if update_status:
                block.status = "stale"
                block.last_summary = "block range changed; refresh result"
            return current_snapshot

        if not np.array_equal(current_snapshot[start:end], block.input_slice, equal_nan=True):
            if update_status:
                block.status = "stale"
                block.last_summary = "upstream changed; refresh result"
            return current_snapshot

        if update_status:
            block.status = "ready"
            block.last_summary = f"cached {block.output_slice.size} pts"
        out = np.array(current_snapshot, copy=True)
        out[start:end] = block.output_slice
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

        self._collect_preview_results(update_status=True)
        self._refresh_selection()

        lines = [
            "Preview 语义：每个 channel 从 baseline 开始，按 layer 顺序、再按 block 起点顺序串行叠加。",
            "interactive block 只在当前输入快照匹配时生效；上游变化后会变成 stale。",
            "param block 目前只保存参数，还不参与实际计算。",
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

    def _commit_preview(self) -> None:
        if not self.pipelines:
            return

        preview_results = self._collect_preview_results(update_status=True)
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
            + "\n\ninteractive block 的缓存结果已经烘焙进 baseline，后续继续编辑需要重新打开对应 block。",
        )

    def _refresh_selection(self):
        found = self._find_block(self.selected_block_id)
        if found is None:
            for lane in self._lane_widgets:
                lane.set_selected("")
            if self.selected_tool_id is None:
                self.info_text.setPlainText("请选择一个工具，然后点击或拖到某个 step 轨道中放置 block。")
            return

        pipeline, layer, block = found
        for lane in self._lane_widgets:
            lane.set_selected(block.block_id)

        meta = TOOL_META.get(block.tool_id, {})
        self.info_text.setPlainText(
            f"selected block\n"
            f"channel: #{pipeline.data_id} {pipeline.channel}\n"
            f"layer: {layer.label}\n"
            f"tool: {meta.get('name', block.tool_id)}\n"
            f"range: {block.start} ~ {block.end}\n"
            f"status: {block.status}\n"
            f"params: {block.params}\n"
            f"summary: {block.last_summary or '-'}"
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
