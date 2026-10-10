from __future__ import annotations

from copy import deepcopy
import json
import math

import numpy as np
import pandas as pd

from PySide6.QtCore import (
    QMimeData,
    QPoint,
    QRect,
    Qt,
    QObject,
    Signal,
    QTimer,
)
from PySide6.QtGui import (
    QAction,
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
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from base.data_manager import DataManager
from core.em_data import EMData
from core.pipeline_model import (
    ChannelPipeline,
    PreviewChannelSlice,
    PipelineBlock,
    PipelineLayer,
    PipelineSettingsGroup,
    SnapEngine,
    uid,
)
from core.pipeline_layout import (
    fit_timeline_px_per_point,
    linked_group_label,
    linked_group_label_index,
    px_per_point,
)
from core.recipe import recipe_from_pipeline, save_recipe
from plugin_api import ChannelBlock, ChannelMeta
from ui.dialogs.kp_selection_dialog import KpSelectionDialog
from ui.dialogs.pipeline_param_dialog import PipelineParamDialog
from base.plugin_runtime import run_plugin_process
from ui.dialogs.plugin_parameters import (
    PIPELINE_LINK_ALL_TEXT,
    PluginParamDialog,
    create_pipeline_link_all_checkbox,
)
from ui.models.processing_tools import PROCESSING_TOOL_MIME, PIPELINE_TOOL_MIME
from ui.widgets.pipeline_overview import (
    CHANNEL_HEADER_H,
    LAYER_H,
    RULER_H,
    ChannelOverviewWidget,
    RulerWidget,
)
from ui.tools import registry as tool_registry
from base.qt_bridge import run as run_in_pool


HEADER_W = 120
ADD_ROW_H = 24
HANDLE_W = 7
MIN_BLOCK_SPAN = 100
PIPELINE_BLOCK_CLIPBOARD_MIME = "application/x-mtpro-pipeline-block"
READY_BY_DEFAULT_INTERACTIVE_TOOLS = frozenset({"remove_spike", "remove_region"})
GLOBAL_TIME_TOOL_IDS = frozenset({"demean", "detrend", "kp_selection"})
LOCAL_SETTING_KEYS = frozenset({"indices", "operations", "region", "time_ranges"})
TIMESTAMP_INDEX_TOOLS = frozenset({"remove_spike", "remove_step_diff"})
POINT_SETTING_KEYS = frozenset({
    "anchor_spacing_points",
    "context_points",
    "embedding_points",
    "max_pulse_width",
    "min_segment_points",
    "padding",
    "smoothing_window_points",
    "window",
    "window_points",
})
ODD_POINT_SETTING_KEYS = frozenset({"window_points"})
LINKED_GROUP_COLORS = (
    "#2f80ed",
    "#9b51e0",
    "#00a884",
    "#e67e22",
    "#d64562",
    "#008fb3",
    "#795548",
    "#607d8b",
)


def _is_pipeline_capable(spec) -> bool:
    # Pipeline block 必须产出与输入同长的单通道序列。robust estimate 与
    # remote reference 属于站级 output 工具，虽然有参数表单，却产出阻抗等
    # 新物理量，不能伪装成单通道变换 block。
    return tool_registry.get_plugin(spec.tool_id) is not None


def _build_tool_meta() -> dict[str, dict[str, object]]:
    meta: dict[str, dict[str, object]] = {}
    for spec in tool_registry.all_specs():
        if not _is_pipeline_capable(spec):
            continue
        entry: dict[str, object] = {"name": spec.display_name, "kind": spec.kind}
        if spec.fields:
            entry["fields"] = list(spec.fields)
        meta[spec.tool_id] = entry
    return meta


TOOL_META: dict[str, dict[str, object]] = _build_tool_meta()


def refresh_tool_meta() -> None:
    """Refresh the pipeline catalogue after startup plugin discovery."""

    TOOL_META.clear()
    TOOL_META.update(_build_tool_meta())


BLOCK_STATUS_STYLE: dict[str, dict[str, str]] = {
    "empty": {"accent": "#8b8f97", "label": "待处理"},
    "configured": {"accent": "#7e8b99", "label": "已配置"},
    "ready": {"accent": "#778f7c", "label": "就绪"},
    "stale": {"accent": "#9a7b7b", "label": "需刷新"},
    "error": {"accent": "#b34b4b", "label": "执行失败"},
}

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


def pipeline_tool_id_from_mime(mime_data) -> str | None:
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


def is_global_time_tool(tool_id: str) -> bool:
    return tool_registry.canonical_id(tool_id) in GLOBAL_TIME_TOOL_IDS


class PipelineTrackWidget(QWidget):
    add_block_requested = Signal(int, str, str, int)
    tool_dropped = Signal(int, str, str, str, int)
    block_selected = Signal(str)
    block_double_clicked = Signal(str)
    block_removed = Signal(str)
    block_unlink_requested = Signal(str)
    block_copy_requested = Signal(str)
    block_cut_requested = Signal(str)
    block_paste_requested = Signal(int, str, str, int)
    guide_changed = Signal(list)
    layout_changed = Signal()
    block_geometry_preview_changed = Signal(str)
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
        self._pixels_per_point = px_per_point(self.zoom)
        self.selected_block_id = ""
        self.selected_group_id = ""
        self._guides: list[int] = []
        self._drag_block: PipelineBlock | None = None
        self._drag_mode = ""
        self._drag_origin_start = 0
        self._drag_origin_end = 0
        self._drag_origin_layer_id = ""
        self._drag_origin_ref = pipeline.ref
        self._drag_pointer_offset_pts = 0
        self._drag_target_layer: PipelineLayer | None = None
        self._drag_target_pipeline: ChannelPipeline | None = None
        self.setFixedHeight(LAYER_H)
        self.setMinimumWidth(600)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)

    @property
    def _px_per_pt(self) -> float:
        return self._pixels_per_point

    def _pts_to_px(self, pts: int) -> float:
        return pts * self._pixels_per_point

    def _px_to_pts(self, px: int) -> int:
        return int(px / self._px_per_pt)

    def set_zoom(self, zoom: float, pixels_per_point: float | None = None):
        self.zoom = zoom
        self._pixels_per_point = (
            px_per_point(zoom) if pixels_per_point is None else pixels_per_point
        )
        self.setMinimumWidth(max(1, int(self._pts_to_px(self.data_len))))
        self.update()

    def set_selected(self, block_id: str, group_id: str = ""):
        self.selected_block_id = block_id
        self.selected_group_id = group_id
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
        locked = is_global_time_tool(block.tool_id)
        if not locked:
            handle_color = palette.color(QPalette.ColorRole.Light)
            handle_color.setAlpha(24)
            painter.fillRect(x1, 3, HANDLE_W, height, handle_color)
            painter.fillRect(x2 - HANDLE_W, 3, HANDLE_W, height, handle_color)

        owner = self._owner()
        group_label = ""
        group_color: QColor | None = None
        if block.settings_group_id and owner is not None:
            group_label, group_color = owner._settings_group_visual(
                block.settings_group_id
            )
            painter.fillRect(
                QRect(x1 + HANDLE_W, 3, min(4, max(0, width - HANDLE_W)), height),
                group_color,
            )

        border_color = palette.color(QPalette.ColorRole.Mid)
        border_width = 1
        if group_color is not None and block.settings_group_id == self.selected_group_id:
            border_color = group_color
            border_width = 3 if block.block_id == self.selected_block_id else 2
        elif block.block_id == self.selected_block_id:
            border_color = palette.color(QPalette.ColorRole.Highlight)
            border_width = 2
        elif block.status == "stale":
            border_color = QColor(status_meta["accent"])
        painter.setPen(QPen(border_color, border_width))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

        content_left = HANDLE_W + 8
        if group_color is not None:
            badge_font = QFont("Sans Serif", 8, QFont.Weight.Bold)
            painter.setFont(badge_font)
            badge_text = f"🔗{group_label}"
            badge_width = painter.fontMetrics().horizontalAdvance(badge_text) + 10
            available_width = max(0, width - HANDLE_W * 2 - 8)
            if available_width >= badge_width:
                badge_rect = QRect(
                    x1 + HANDLE_W + 6,
                    rect.top() + 6,
                    badge_width,
                    max(14, rect.height() - 12),
                )
                badge_fill = QColor(group_color)
                badge_fill.setAlpha(220)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(badge_fill)
                painter.drawRoundedRect(badge_rect, 5, 5)
                painter.setPen(QColor("#ffffff"))
                painter.drawText(
                    badge_rect,
                    Qt.AlignmentFlag.AlignCenter,
                    badge_text,
                )
                content_left += badge_width + 5

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(palette.color(QPalette.ColorRole.ButtonText))
        painter.setFont(QFont("Courier New", 9, QFont.Weight.Bold))
        tool_text = str(TOOL_META.get(block.tool_id, {}).get("name", block.tool_id))
        status_text = str(status_meta["label"])
        painter.drawText(
            rect.adjusted(content_left, 0, -(HANDLE_W + 4), -2),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"{'🔒 ' if locked else ''}{tool_text} [{status_text}]",
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

    def _hit_test(self, x_pos: int) -> tuple[PipelineBlock | None, str]:
        for block in reversed(self.layer.sorted_blocks()):
            x1 = int(self._pts_to_px(block.start))
            x2 = max(int(self._pts_to_px(block.end)), x1 + 12)
            if is_global_time_tool(block.tool_id) and x1 <= x_pos <= x2:
                return block, "locked"
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

    def _fit_start_in_layer(self, layer: PipelineLayer, exclude_id: str, span: int, raw_start: int) -> int | None:
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

    def _owner(self) -> DataProcessPipelineWidget | None:
        widget = self.parentWidget()
        while widget is not None and not isinstance(widget, DataProcessPipelineWidget):
            widget = widget.parentWidget()
        return widget

    def _iter_tracks(self):
        owner = self._owner()
        if owner is None:
            return []
        return [track for lane in owner._lane_widgets for track in lane._tracks]

    def _find_track(self, data_id: int, channel_name: str, layer_id: str) -> PipelineTrackWidget | None:
        for track in self._iter_tracks():
            if (
                track.pipeline.data_id == data_id
                and track.pipeline.channel == channel_name
                and track.layer.layer_id == layer_id
            ):
                return track
        return None

    def _resolve_target_track(self, global_pos: QPoint) -> PipelineTrackWidget | None:
        widget = QApplication.widgetAt(global_pos)
        while widget is not None and not isinstance(widget, PipelineTrackWidget):
            widget = widget.parentWidget()
        return widget

    def mousePressEvent(self, event) -> None:
        x_pos = event.position().toPoint().x()
        block, mode = self._hit_test(x_pos)
        if event.button() == Qt.MouseButton.MiddleButton:
            if block is not None:
                self.block_removed.emit(block.block_id)
                event.accept()
            else:
                event.ignore()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return

        if block is not None:
            self.selected_block_id = block.block_id
            self.block_selected.emit(block.block_id)
            if mode == "locked":
                self.setToolTip("全局处理覆盖完整时间范围，不能移动或调整时长。")
                self.update()
                event.accept()
                return
            self._drag_block = block
            self._drag_mode = mode
            self._drag_origin_start = block.start
            self._drag_origin_end = block.end
            self._drag_origin_layer_id = self.layer.layer_id
            self._drag_origin_ref = self.pipeline.ref
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
            block, mode = self._hit_test(x_pos)
            owner = self._owner()
            if block is not None and block.settings_group_id and owner is not None:
                self.setToolTip(owner._settings_group_tooltip(block.settings_group_id))
            elif mode == "locked":
                self.setToolTip("全局处理覆盖完整时间范围，不能移动或调整时长。")
            else:
                self.setToolTip("")
            if mode in ("resize_left", "resize_right"):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif mode == "move":
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        drag_before = (
            self._drag_block.start,
            self._drag_block.end,
            (self._drag_target_pipeline or self.pipeline).ref,
            (self._drag_target_layer or self.layer).layer_id,
        )
        guides: list[int] = []
        target_track = self
        if self._drag_mode == "move":
            current_pipeline = self._drag_target_pipeline or self.pipeline
            current_layer = self._drag_target_layer or self.layer
            current_track = self._find_track(current_pipeline.data_id, current_pipeline.channel, current_layer.layer_id)
            if current_track is None:
                current_track = self
            resolved = self._resolve_target_track(event.globalPosition().toPoint())
            if (
                resolved is not None
                and self._drag_block.settings_group_id
                and resolved.pipeline.ref != self.pipeline.ref
            ):
                resolved = None
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
        drag_after = (
            self._drag_block.start,
            self._drag_block.end,
            (self._drag_target_pipeline or self.pipeline).ref,
            (self._drag_target_layer or self.layer).layer_id,
        )
        if drag_after != drag_before:
            self.block_geometry_preview_changed.emit(self._drag_block.block_id)
        self.layout_changed.emit()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        del event
        if self._drag_block is None:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        changed = (
            (self._drag_target_pipeline or self.pipeline).ref != self._drag_origin_ref
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

    def _build_block_context_menu(
        self,
        block: PipelineBlock | None,
    ) -> tuple[QMenu, dict[str, QAction]]:
        menu = QMenu(self)
        actions: dict[str, QAction] = {}
        if block is not None:
            actions["settings"] = menu.addAction("参数设置…")
            if block.settings_group_id:
                actions["unlink"] = menu.addAction("解除当前 Block 联动")
            menu.addSeparator()
            actions["cut"] = menu.addAction("剪切 Block")
            actions["copy"] = menu.addAction("复制 Block")
        clipboard = QApplication.clipboard()
        mime_data = None if clipboard is None else clipboard.mimeData()
        if (
            mime_data is not None
            and mime_data.hasFormat(PIPELINE_BLOCK_CLIPBOARD_MIME)
        ):
            if block is not None:
                menu.addSeparator()
            actions["paste"] = menu.addAction("粘贴 Block")
        return menu, actions

    def contextMenuEvent(self, event) -> None:
        block, _mode = self._hit_test(event.pos().x())
        if block is not None:
            self.selected_block_id = block.block_id
            self.block_selected.emit(block.block_id)
        menu, actions = self._build_block_context_menu(block)
        if not actions:
            event.ignore()
            return
        selected = menu.exec(event.globalPos())
        if selected is actions.get("settings") and block is not None:
            self.block_double_clicked.emit(block.block_id)
        elif selected is actions.get("unlink") and block is not None:
            self.block_unlink_requested.emit(block.block_id)
        elif selected is actions.get("cut") and block is not None:
            self.block_cut_requested.emit(block.block_id)
        elif selected is actions.get("copy") and block is not None:
            self.block_copy_requested.emit(block.block_id)
        elif selected is actions.get("paste"):
            self.block_paste_requested.emit(
                self.pipeline.data_id,
                self.pipeline.channel,
                self.layer.layer_id,
                self._px_to_pts(event.pos().x()),
            )
        event.accept()


class ChannelPipelineWidget(QWidget):
    add_layer_requested = Signal(int, str)
    add_block_requested = Signal(int, str, str, int)
    tool_dropped = Signal(int, str, str, str, int)
    block_selected = Signal(str)
    block_double_clicked = Signal(str)
    block_removed = Signal(str)
    block_unlink_requested = Signal(str)
    block_copy_requested = Signal(str)
    block_cut_requested = Signal(str)
    block_paste_requested = Signal(int, str, str, int)
    guide_changed = Signal(list)
    layout_changed = Signal()
    block_geometry_preview_changed = Signal(str)
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
        self._pixels_per_point = px_per_point(self.zoom)
        self._show_baseline = bool(show_baseline)
        self._tracks: list[PipelineTrackWidget] = []
        self.overview_widget: ChannelOverviewWidget | None = None
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
        self.overview_widget.set_zoom(self.zoom, self._pixels_per_point)
        right_layout.addWidget(self.overview_widget)

        self._tracks.clear()
        for layer in self.pipeline.layers:
            left_label = QLabel(layer.label)
            left_label.setFixedHeight(LAYER_H)
            left_label.setIndent(18)
            left_label.setFrameShape(QFrame.Shape.StyledPanel)
            left_layout.addWidget(left_label)

            track = PipelineTrackWidget(self.pipeline, layer, self.data_len, self.snap, self)
            track.set_zoom(self.zoom, self._pixels_per_point)
            track.add_block_requested.connect(self.add_block_requested)
            track.tool_dropped.connect(self.tool_dropped)
            track.block_selected.connect(self.block_selected)
            track.block_double_clicked.connect(self.block_double_clicked)
            track.block_removed.connect(self.block_removed)
            track.block_unlink_requested.connect(self.block_unlink_requested)
            track.block_copy_requested.connect(self.block_copy_requested)
            track.block_cut_requested.connect(self.block_cut_requested)
            track.block_paste_requested.connect(self.block_paste_requested)
            track.guide_changed.connect(self.guide_changed)
            track.layout_changed.connect(self.layout_changed)
            track.block_geometry_preview_changed.connect(
                self.block_geometry_preview_changed
            )
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
        self.set_zoom(self.zoom, self._pixels_per_point)

    def set_zoom(self, zoom: float, pixels_per_point: float | None = None):
        self.zoom = zoom
        self._pixels_per_point = (
            px_per_point(zoom) if pixels_per_point is None else pixels_per_point
        )
        if self.overview_widget is not None:
            self.overview_widget.set_zoom(zoom, self._pixels_per_point)
        for track in self._tracks:
            track.set_zoom(zoom, self._pixels_per_point)

    def set_baseline_visible(self, visible: bool) -> None:
        self._show_baseline = bool(visible)
        if self.overview_widget is not None:
            self.overview_widget.set_baseline_visible(visible)

    def set_selected(self, block_id: str, group_id: str = ""):
        for track in self._tracks:
            track.set_selected(block_id, group_id)

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

    def __init__(self, data_manager: DataManager | None = None, parent=None) -> None:
        super().__init__(parent)
        self.data_manager = data_manager
        self.data_id = 0
        self.em_data: EMData | None = None
        self._channel_refs: list[tuple[int, str]] = []
        self._hidden_channel_refs: set[tuple[int, str]] = set()
        self.selected_block_id: str = ""
        self.pipelines: list[ChannelPipeline] = []
        self._lane_widgets: list[ChannelPipelineWidget] = []
        self._editor_dialogs: list[QDialog] = []
        self._settings_groups: dict[str, PipelineSettingsGroup] = {}
        self._group_editor_dialogs: dict[str, QDialog] = {}
        self._plugin_tasks: dict[str, int] = {}
        self._plugin_bridges: dict[str, _PluginTaskBridge] = {}
        self._plugin_generation = 0
        self._preview_cache: dict[
            tuple[int, str],
            tuple[tuple, np.ndarray],
        ] = {}
        self._pending_lane_rebuild = False
        self._zoom = 6.0
        self._overview_baseline_visible = False
        self._snap = SnapEngine()
        self._ruler_widget: RulerWidget | None = None
        self.pipeline_hint: QLabel | None = None
        self._timeline_scale_timer = QTimer(self)
        self._timeline_scale_timer.setSingleShot(True)
        self._timeline_scale_timer.timeout.connect(self._apply_timeline_scale)
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
        self.zoom_slider.setToolTip("1 = 全部时间序列适配当前窗口宽度")
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

        self.channel_visibility_button = QToolButton(toolbar)
        self.channel_visibility_button.setText("通道")
        self.channel_visibility_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.channel_visibility_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.channel_visibility_menu = QMenu(self.channel_visibility_button)
        self.channel_visibility_button.setMenu(self.channel_visibility_menu)
        toolbar_layout.addWidget(self.channel_visibility_button)

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

        commit_button = QPushButton("commit", toolbar)
        commit_button.clicked.connect(self._commit_preview)
        self.clear_blocks_after_commit_checkbox = QCheckBox("提交后清空 block", toolbar)
        self.clear_blocks_after_commit_checkbox.setChecked(False)
        self.clear_blocks_after_commit_checkbox.setToolTip(
            "仅在至少一个 channel 成功提交后，清空当前 Pipeline 窗口内的全部 block。"
        )
        export_button = QPushButton("export recipe", toolbar)
        export_button.clicked.connect(self._export_recipe)
        toolbar_layout.addWidget(commit_button)
        toolbar_layout.addWidget(self.clear_blocks_after_commit_checkbox)
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

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._timeline_scale_timer.start()

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

    def has_blocks_for_channel(self, data_id: int, channel: str) -> bool:
        pipeline = self._find_pipeline(data_id, channel)
        return bool(
            pipeline is not None
            and any(layer.blocks for layer in pipeline.layers)
        )

    def reset_restored_data(self, data_id: int, channel: str | None) -> None:
        """Discard only restored lanes, including open editors and late worker results."""
        affected = {
            ref for ref in self._channel_refs
            if ref[0] == data_id and (channel is None or ref[1] == channel)
        }
        if not affected:
            return
        block_ids = {
            block.block_id for pipeline in self.pipelines if pipeline.ref in affected
            for layer in pipeline.layers for block in layer.blocks
        }
        for block_id in block_ids:
            self._plugin_tasks.pop(block_id, None)
            # The worker may still emit; generation lookup now rejects it.
            self._plugin_bridges.pop(block_id, None)
        for dialog in list(self._editor_dialogs):
            if dialog.property("pipeline_block_id") in block_ids:
                dialog.reject()
        self.pipelines = [pipeline for pipeline in self.pipelines if pipeline.ref not in affected]
        for ref in affected:
            self._preview_cache.pop(ref, None)
        if self.selected_block_id in block_ids:
            self.selected_block_id = ""
        if channel is None:
            data = self._data_for(data_id)
            refs = []
            inserted = False
            for ref in self._channel_refs:
                if ref[0] != data_id:
                    refs.append(ref)
                elif not inserted:
                    refs.extend((data_id, name) for name in data.channel_names())
                    inserted = True
            self._channel_refs = refs
        self._sync_pipelines(self._channel_refs)
        self._normalize_pipelines_for_current_window()
        self._rebuild_lanes()
        self._refresh_all_pipeline_statuses()
        self._refresh_selection()

    def channel_refs(self) -> list[tuple[int, str]]:
        return list(self._channel_refs)

    def visible_channel_refs(self) -> list[tuple[int, str]]:
        return [ref for ref in self._channel_refs if ref not in self._hidden_channel_refs]

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
        valid_refs = set(channel_refs)
        self._preview_cache = {
            ref: cached
            for ref, cached in self._preview_cache.items()
            if ref in valid_refs
        }
        self._hidden_channel_refs.intersection_update(channel_refs)
        self._prune_settings_groups()

    def _data_for(self, data_id: int) -> EMData | None:
        if self.data_manager is not None:
            return self.data_manager.get(data_id)
        if self.em_data is not None and data_id == self.data_id:
            return self.em_data
        return None

    def _channel_series(self, data_id: int, channel_name: str) -> pd.Series | None:
        data = self._data_for(data_id)
        if data is None:
            return None
        if not data.has_channel(channel_name):
            return None
        return data.get_channel_series(channel_name)

    def _time_range_for_channel(self, data_id: int, channel_name: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        series = self._channel_series(data_id, channel_name)
        if series is None or series.empty:
            return None
        index = series.index
        if len(index) == 0:
            return None
        return pd.Timestamp(index[0]), pd.Timestamp(index[-1])

    def _common_time_range(
        self,
        refs: list[tuple[int, str]] | None = None,
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
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

    def _find_pipeline(self, data_id: int, channel_name: str) -> ChannelPipeline | None:
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

    def _timeline_pixels_per_point(self) -> float:
        """Return the shared scale for ruler, overviews, and block tracks."""
        viewport_width = self.scroll.viewport().width() - HEADER_W
        if viewport_width <= 0:
            return px_per_point(self._zoom)
        return fit_timeline_px_per_point(
            self._zoom,
            self._editor_data_len(),
            viewport_width,
        )

    def _apply_timeline_scale(self) -> None:
        pixels_per_point = self._timeline_pixels_per_point()
        if self._ruler_widget is not None:
            self._ruler_widget.set_zoom(self._zoom, pixels_per_point)
        for lane in self._lane_widgets:
            lane.set_zoom(self._zoom, pixels_per_point)

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
                    if is_global_time_tool(block.tool_id):
                        if block.start != 0 or block.end != data_len:
                            block.start = 0
                            block.end = data_len
                            self._invalidate_block_cache(
                                block,
                                "global time range changed; regenerating full-channel preview",
                            )
                        normalized_blocks.append(block)
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
        self._refresh_channel_visibility_menu()
        if not self.pipelines:
            placeholder = QLabel("拖动一个或多个 channel 到工作区后开始构建 pipeline。", self.scroll_content)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setMargin(24)
            self.scroll_layout.insertWidget(0, placeholder)
            return

        visible_pipelines = [
            pipeline for pipeline in self.pipelines if pipeline.ref not in self._hidden_channel_refs
        ]
        if not visible_pipelines:
            placeholder = QLabel("所有通道均已隐藏；使用工具栏“通道”菜单重新显示。", self.scroll_content)
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
        pixels_per_point = self._timeline_pixels_per_point()
        self._ruler_widget.set_zoom(self._zoom, pixels_per_point)
        ruler_layout.addWidget(self._ruler_widget, 1)
        self.scroll_layout.insertWidget(0, ruler_row)

        for pipeline in visible_pipelines:
            data_len = self._pipeline_data_len(pipeline)
            lane = ChannelPipelineWidget(
                pipeline,
                self._pipeline_baseline(pipeline),
                data_len,
                self._snap,
                show_baseline=self._overview_baseline_visible,
                parent=self.scroll_content,
            )
            lane.set_zoom(self._zoom, pixels_per_point)
            lane.add_layer_requested.connect(self._on_add_layer_requested)
            lane.add_block_requested.connect(self._on_add_block_requested)
            lane.tool_dropped.connect(self._on_tool_dropped)
            lane.block_selected.connect(self._on_block_selected)
            lane.block_double_clicked.connect(self._on_block_double_clicked)
            lane.block_removed.connect(self._on_block_removed)
            lane.block_unlink_requested.connect(self._on_block_unlink_requested)
            lane.block_copy_requested.connect(self._copy_block_to_clipboard)
            lane.block_cut_requested.connect(self._cut_block_to_clipboard)
            lane.block_paste_requested.connect(self._paste_block_from_clipboard)
            lane.guide_changed.connect(self._broadcast_guides)
            lane.layout_changed.connect(self._refresh_lane_tracks)
            lane.block_geometry_preview_changed.connect(
                self._on_block_geometry_preview_changed
            )
            lane.block_geometry_changed.connect(self._on_block_geometry_changed)
            self._lane_widgets.append(lane)
            self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, lane)

        self._timeline_scale_timer.start()

    def _refresh_channel_visibility_menu(self) -> None:
        self.channel_visibility_menu.clear()
        visible_count = 0
        for ref in self._channel_refs:
            data_id, channel_name = ref
            action = QAction(f"#{data_id} {channel_name}", self.channel_visibility_menu)
            action.setCheckable(True)
            is_visible = ref not in self._hidden_channel_refs
            action.setChecked(is_visible)
            action.toggled.connect(
                lambda checked, channel_ref=ref: self._set_channel_visible(channel_ref, checked)
            )
            self.channel_visibility_menu.addAction(action)
            visible_count += int(is_visible)

        self.channel_visibility_button.setEnabled(bool(self._channel_refs))
        self.channel_visibility_button.setText(
            f"通道 ({visible_count}/{len(self._channel_refs)})"
        )

    def _set_channel_visible(self, channel_ref: tuple[int, str], visible: bool) -> None:
        if channel_ref not in self._channel_refs:
            return
        if visible:
            self._hidden_channel_refs.discard(channel_ref)
        else:
            self._hidden_channel_refs.add(channel_ref)

        found = self._find_block(self.selected_block_id)
        if found is not None and found[0].ref == channel_ref:
            self.selected_block_id = ""
        self._rebuild_lanes()
        self._refresh_selection()

    def _on_zoom_changed(self, value: int):
        self._zoom = float(value)
        self.zoom_label.setText(str(value))
        self._apply_timeline_scale()

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
                self._apply_timeline_scale()
                return

    def _default_params_for_tool(self, tool_id: str) -> dict:
        meta = TOOL_META.get(tool_id, {})
        params: dict[str, object] = {}
        for key, _label, _parser, default in meta.get("fields", []):
            params[key] = default
        return params

    def _find_layer(self, data_id: int, channel_name: str, layer_id: str) -> tuple[ChannelPipeline, PipelineLayer] | None:
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
    ) -> int | None:
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

    def _add_block_with_tool(self, data_id: int, channel_name: str, layer_id: str, start: int, tool_id: str | None):
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

        global_time = is_global_time_tool(tool_id)
        if global_time:
            full_series = self._channel_series(data_id, channel_name)
            common_series = self._channel_slice(data_id, channel_name)
            if (
                full_series is None
                or len(full_series) != len(common_series)
                or not full_series.index.equals(common_series.index)
            ):
                QMessageBox.warning(
                    self,
                    "无法添加全局处理",
                    "当前 Pipeline 的公共时间范围没有覆盖该通道全部数据。\n"
                    "请为该通道单独创建 Pipeline，或在数据管理器中右键立即处理。",
                )
                return
        span = data_len if global_time else min(
            data_len,
            max(1, min(2000, max(200, data_len // 10))),
        )
        if global_time:
            start = 0
        snapped_start, _guides = self._snap.snap_interval(
            start,
            start + span,
            self._timeline_pixels_per_point(),
            data_len,
        )
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
        if tool_id in READY_BY_DEFAULT_INTERACTIVE_TOOLS:
            preview_before_block = self._compose_channel_preview(
                data_id,
                channel_name,
                stop_before_block_id=block.block_id,
                update_status=True,
            )
            block.input_slice = np.array(
                preview_before_block[block.start:block.end],
                copy=True,
            )
            block.output_slice = np.array(block.input_slice, copy=True)
            block.status = "ready"
            block.last_summary = "ready; no points or regions selected"
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

    def _block_clipboard_payload(self, block_id: str) -> dict | None:
        found = self._find_block(block_id)
        if found is None:
            return None
        _pipeline, _layer, block = found
        params = self._default_params_for_tool(block.tool_id)
        params.update(self._shared_settings(block.tool_id, block.params))
        return {
            "version": 1,
            "tool_id": block.tool_id,
            "span_points": max(1, block.end - block.start),
            "params": params,
        }

    def _copy_block_to_clipboard(self, block_id: str) -> bool:
        payload = self._block_clipboard_payload(block_id)
        if payload is None:
            return False
        try:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "复制失败", f"Block 参数无法复制：{exc}")
            return False
        mime_data = QMimeData()
        mime_data.setData(PIPELINE_BLOCK_CLIPBOARD_MIME, encoded)
        mime_data.setText(
            f"MTPro Pipeline Block: {payload['tool_id']}"
        )
        QApplication.clipboard().setMimeData(mime_data)
        return True

    def _cut_block_to_clipboard(self, block_id: str) -> None:
        if self._copy_block_to_clipboard(block_id):
            self._on_block_removed(block_id)

    def _pipeline_block_from_clipboard(self) -> dict | None:
        clipboard = QApplication.clipboard()
        mime_data = None if clipboard is None else clipboard.mimeData()
        if (
            mime_data is None
            or not mime_data.hasFormat(PIPELINE_BLOCK_CLIPBOARD_MIME)
        ):
            return None
        try:
            payload = json.loads(
                bytes(mime_data.data(PIPELINE_BLOCK_CLIPBOARD_MIME)).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or payload.get("version") != 1:
            return None
        tool_id = payload.get("tool_id")
        params = payload.get("params")
        span_points = payload.get("span_points")
        if (
            not isinstance(tool_id, str)
            or not isinstance(params, dict)
            or isinstance(span_points, bool)
            or not isinstance(span_points, int)
            or span_points <= 0
        ):
            return None
        return payload

    def _paste_block_from_clipboard(
        self,
        data_id: int,
        channel_name: str,
        layer_id: str,
        start: int,
    ) -> None:
        payload = self._pipeline_block_from_clipboard()
        if payload is None:
            QMessageBox.warning(self, "粘贴失败", "剪贴板中没有有效的 Pipeline Block。")
            return
        tool_id = self._canonical_tool_id(payload["tool_id"])
        if tool_id not in TOOL_META:
            QMessageBox.warning(self, "粘贴失败", "当前版本不支持剪贴板中的处理工具。")
            return
        found = self._find_layer(data_id, channel_name, layer_id)
        if found is None:
            return
        pipeline, layer = found
        data_len = self._pipeline_data_len(pipeline)
        if data_len <= 0:
            return
        global_time = is_global_time_tool(tool_id)
        if global_time:
            full_series = self._channel_series(data_id, channel_name)
            common_series = self._channel_slice(data_id, channel_name)
            if (
                full_series is None
                or len(full_series) != len(common_series)
                or not full_series.index.equals(common_series.index)
            ):
                QMessageBox.warning(
                    self,
                    "无法粘贴全局处理",
                    "当前 Pipeline 的公共时间范围没有覆盖该通道全部数据。",
                )
                return
        span = data_len if global_time else min(data_len, int(payload["span_points"]))
        start = 0 if global_time else max(0, min(data_len - span, int(start)))
        snapped_start, _guides = self._snap.snap_interval(
            start,
            start + span,
            self._timeline_pixels_per_point(),
            data_len,
        )
        fitted_start = self._place_block_start(
            layer,
            span,
            snapped_start,
            data_len,
        )
        if fitted_start is None:
            QMessageBox.information(
                self,
                "无法粘贴",
                "当前 step 没有足够空间放置复制的 Block。",
            )
            return
        params = self._default_params_for_tool(tool_id)
        params.update(deepcopy(payload["params"]))
        block = PipelineBlock(
            tool_id=tool_id,
            start=fitted_start,
            end=fitted_start + span,
            params=params,
            status="configured" if params else "empty",
            last_summary="pasted independent block; awaiting preview",
        )
        layer.blocks.append(block)
        self.selected_block_id = block.block_id
        self._refresh_all_pipeline_statuses()
        self._refresh_selection()

    def _on_block_removed(self, block_id: str):
        found = self._find_block(block_id)
        group_id = "" if found is None else found[2].settings_group_id
        for pipeline in self.pipelines:
            if pipeline.remove_block(block_id):
                break
        if group_id:
            remaining = self._group_members(group_id)
            if len(remaining) <= 1:
                for _pipeline, _layer, member in remaining:
                    member.settings_group_id = ""
                self._settings_groups.pop(group_id, None)
            else:
                group = self._settings_groups.get(group_id)
                if group is not None and group.reference_block_id == block_id:
                    self._promote_group_reference(group, remaining[0])
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

    def _on_block_geometry_preview_changed(self, block_id: str) -> None:
        """Mirror a linked range while dragging without committing side effects."""
        found = self._find_block(block_id)
        if found is None:
            return
        source_pipeline, source_layer, source_block = found
        if not source_block.settings_group_id:
            return
        source_layer_index = source_pipeline.layers.index(source_layer)

        for target_pipeline, target_layer, target_block in self._group_members(
            source_block.settings_group_id
        ):
            if target_block is source_block:
                continue
            target_range = self._synced_block_range(
                source_pipeline,
                source_block,
                target_pipeline,
            )
            if target_range is None:
                continue
            start, end = target_range
            if source_layer_index < len(target_pipeline.layers):
                destination = target_pipeline.layers[source_layer_index]
                if (
                    target_layer is not destination
                    and not self._layer_has_overlap(
                        destination,
                        start,
                        end,
                        source_block.settings_group_id,
                    )
                ):
                    target_layer.blocks.remove(target_block)
                    destination.blocks.append(target_block)
            target_block.start, target_block.end = start, end

    def _on_block_geometry_changed(self, block_id: str):
        found = self._find_block(block_id)
        if found is None:
            return
        pipeline, layer, block = found
        reason = "range/lane changed; reopen editor to refresh result"
        self._invalidate_block_cache(block, reason)
        if block.settings_group_id:
            source_layer_index = pipeline.layers.index(layer)
            for target_pipeline, target_layer, target_block in self._group_members(
                block.settings_group_id
            ):
                if target_block is block:
                    continue
                target_range = self._synced_block_range(
                    pipeline,
                    block,
                    target_pipeline,
                )
                if target_range is None:
                    continue
                start, end = target_range
                destination = self._target_layer_for_linked_block(
                    target_pipeline,
                    source_layer_index,
                    start,
                    end,
                    block.settings_group_id,
                )
                if target_layer is not destination:
                    target_layer.blocks.remove(target_block)
                    destination.blocks.append(target_block)
                    self._pending_lane_rebuild = True
                target_block.start = start
                target_block.end = end
                self._invalidate_block_cache(
                    target_block,
                    "linked range changed; awaiting preview",
                )
        self.selected_block_id = block.block_id
        if self._pending_lane_rebuild:
            self._pending_lane_rebuild = False
            self._rebuild_lanes()
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
    ) -> PreviewChannelSlice | None:
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

    def _group_members(
        self,
        group_id: str,
    ) -> list[tuple[ChannelPipeline, PipelineLayer, PipelineBlock]]:
        if not group_id:
            return []
        return [
            (pipeline, layer, block)
            for pipeline in self.pipelines
            for layer in pipeline.layers
            for block in layer.blocks
            if block.settings_group_id == group_id
        ]

    def _next_settings_group_label(self) -> str:
        used = {
            group.display_label
            for group in self._settings_groups.values()
            if group.display_label
        }
        index = 0
        while linked_group_label(index) in used:
            index += 1
        return linked_group_label(index)

    def _settings_group_visual(self, group_id: str) -> tuple[str, QColor]:
        group = self._settings_groups.get(group_id)
        label = "" if group is None else group.display_label
        if not label:
            label = group_id[:2].upper() or "?"
        color_index = linked_group_label_index(label)
        return label, QColor(LINKED_GROUP_COLORS[color_index % len(LINKED_GROUP_COLORS)])

    def _settings_group_tooltip(self, group_id: str) -> str:
        label, _color = self._settings_group_visual(group_id)
        member_count = len(self._group_members(group_id))
        return (
            f"联动组 {label} · {member_count} 个通道 · 共用设置\n"
            "点击任一成员会打开同一个属性设置框"
        )

    def _prune_settings_groups(self) -> None:
        for group_id, group in list(self._settings_groups.items()):
            members = self._group_members(group_id)
            if len(members) <= 1:
                for _pipeline, _layer, block in members:
                    block.settings_group_id = ""
                self._settings_groups.pop(group_id, None)
                continue
            if not group.display_label:
                group.display_label = self._next_settings_group_label()
            if not any(
                block.block_id == group.reference_block_id
                for _pipeline, _layer, block in members
            ):
                self._promote_group_reference(group, members[0])

    def _promote_group_reference(
        self,
        group: PipelineSettingsGroup,
        member: tuple[ChannelPipeline, PipelineLayer, PipelineBlock],
    ) -> None:
        pipeline, _layer, block = member
        group.reference_block_id = block.block_id
        group.shared_params = self._shared_settings(block.tool_id, block.params)
        group.source_sample_interval_seconds = self._sample_interval_seconds(pipeline)
        group.revision += 1

    def _shared_settings(self, tool_id: str, params: dict) -> dict:
        del tool_id
        return {
            key: deepcopy(value)
            for key, value in params.items()
            if key not in LOCAL_SETTING_KEYS
        }

    def _sample_interval_seconds(self, pipeline: ChannelPipeline) -> float:
        index = self._channel_slice(pipeline.data_id, pipeline.channel).index
        if len(index) < 2:
            return 0.0
        return float((index[1] - index[0]).total_seconds())

    def _settings_for_target_rate(
        self,
        shared_params: dict,
        source_pipeline: ChannelPipeline,
        target_pipeline: ChannelPipeline,
    ) -> dict:
        values = deepcopy(shared_params)
        source_dt = self._sample_interval_seconds(source_pipeline)
        target_dt = self._sample_interval_seconds(target_pipeline)
        if source_dt <= 0 or target_dt <= 0 or math.isclose(source_dt, target_dt):
            return values
        ratio = source_dt / target_dt
        for key in POINT_SETTING_KEYS:
            raw_value = values.get(key)
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                continue
            if key == "padding" and values.get("padding_unit") == "seconds":
                continue
            scaled = max(1, int(round(float(raw_value) * ratio)))
            if key in ODD_POINT_SETTING_KEYS and scaled % 2 == 0:
                scaled += 1
            values[key] = scaled
        return values

    def _editor_params(self, block: PipelineBlock) -> dict:
        values = deepcopy(block.params)
        group = self._settings_groups.get(block.settings_group_id)
        if group is not None:
            values.update(deepcopy(group.shared_params))
        return values

    @staticmethod
    def _layer_has_overlap(
        layer: PipelineLayer,
        start: int,
        end: int,
        exclude_group_id: str = "",
    ) -> bool:
        return any(
            block.settings_group_id != exclude_group_id
            and start < block.end
            and end > block.start
            for block in layer.blocks
        )

    def _target_layer_for_linked_block(
        self,
        pipeline: ChannelPipeline,
        preferred_index: int,
        start: int,
        end: int,
        group_id: str,
    ) -> PipelineLayer:
        while len(pipeline.layers) <= preferred_index:
            pipeline.add_layer()
            self._pending_lane_rebuild = True
        for layer in pipeline.layers[preferred_index:]:
            if not self._layer_has_overlap(layer, start, end, group_id):
                return layer
        self._pending_lane_rebuild = True
        return pipeline.add_layer()

    def _linked_target_params(
        self,
        source_pipeline: ChannelPipeline,
        source_block: PipelineBlock,
        target_pipeline: ChannelPipeline,
        target_range: tuple[int, int],
        shared_params: dict,
    ) -> dict:
        values = self._default_params_for_tool(source_block.tool_id)
        values.update(self._settings_for_target_rate(
            shared_params,
            source_pipeline,
            target_pipeline,
        ))
        if source_block.tool_id in TIMESTAMP_INDEX_TOOLS:
            translated = self._synced_block_params(
                source_pipeline,
                source_block,
                target_pipeline,
                target_range,
            )
            if translated is not None:
                values["indices"] = translated.get("indices", [])
        return values

    def _ensure_settings_group(self, block_id: str, params: dict) -> str:
        found = self._find_block(block_id)
        if found is None:
            return ""
        source_pipeline, source_layer, source_block = found
        group_id = source_block.settings_group_id
        if not group_id:
            group = PipelineSettingsGroup(
                display_label=self._next_settings_group_label(),
                tool_id=source_block.tool_id,
                shared_params=self._shared_settings(source_block.tool_id, params),
                reference_block_id=source_block.block_id,
                source_sample_interval_seconds=self._sample_interval_seconds(source_pipeline),
            )
            group_id = group.group_id
            self._settings_groups[group_id] = group
            source_block.settings_group_id = group_id
        elif group_id not in self._settings_groups:
            self._settings_groups[group_id] = PipelineSettingsGroup(
                group_id=group_id,
                display_label=self._next_settings_group_label(),
                tool_id=source_block.tool_id,
                shared_params=self._shared_settings(source_block.tool_id, params),
                reference_block_id=source_block.block_id,
                source_sample_interval_seconds=self._sample_interval_seconds(source_pipeline),
            )

        source_layer_index = source_pipeline.layers.index(source_layer)
        existing_refs = {
            pipeline.ref
            for pipeline, _layer, _block in self._group_members(group_id)
        }
        shared_params = self._shared_settings(source_block.tool_id, params)
        for target_pipeline in self.pipelines:
            if target_pipeline.ref in existing_refs:
                continue
            target_range = self._synced_block_range(
                source_pipeline,
                source_block,
                target_pipeline,
            )
            if target_range is None:
                continue
            start, end = target_range
            target_layer = self._target_layer_for_linked_block(
                target_pipeline,
                source_layer_index,
                start,
                end,
                group_id,
            )
            target_layer.blocks.append(PipelineBlock(
                tool_id=source_block.tool_id,
                start=start,
                end=end,
                params=self._linked_target_params(
                    source_pipeline,
                    source_block,
                    target_pipeline,
                    target_range,
                    shared_params,
                ),
                status="configured",
                last_summary=(
                    f"linked with #{source_pipeline.data_id} "
                    f"{source_pipeline.channel}; awaiting preview"
                ),
                settings_group_id=group_id,
            ))
        return group_id

    def _update_settings_group(self, block_id: str, params: dict) -> None:
        found = self._find_block(block_id)
        if found is None:
            return
        source_pipeline, _source_layer, source_block = found
        group = self._settings_groups.get(source_block.settings_group_id)
        if group is None:
            return
        shared_params = self._shared_settings(source_block.tool_id, params)
        shared_changed = shared_params != group.shared_params
        group.shared_params = deepcopy(shared_params)
        group.revision += 1

        for target_pipeline, _target_layer, target_block in self._group_members(group.group_id):
            target_range = (target_block.start, target_block.end)
            values = self._default_params_for_tool(target_block.tool_id)
            values.update(self._settings_for_target_rate(
                shared_params,
                source_pipeline,
                target_pipeline,
            ))
            if target_block is source_block:
                for key in LOCAL_SETTING_KEYS:
                    if key in params:
                        values[key] = deepcopy(params[key])
            elif target_block.tool_id in TIMESTAMP_INDEX_TOOLS:
                translated = self._synced_block_params(
                    source_pipeline,
                    source_block,
                    target_pipeline,
                    target_range,
                )
                values["indices"] = [] if translated is None else translated.get("indices", [])
            elif not shared_changed:
                for key in LOCAL_SETTING_KEYS:
                    if key in target_block.params:
                        values[key] = deepcopy(target_block.params[key])

            self._plugin_tasks.pop(target_block.block_id, None)
            target_block.params = values
            target_block.input_slice = None
            target_block.output_slice = None
            target_block.status = "configured"
            target_block.last_summary = (
                f"linked settings revision {group.revision}; awaiting preview"
            )

    def _detach_settings_group_member(self, block_id: str) -> bool:
        """Detach one Block while preserving links between remaining members."""
        found = self._find_block(block_id)
        if found is None:
            return False
        _pipeline, _layer, block = found
        group_id = block.settings_group_id
        if not group_id:
            return False

        group = self._settings_groups.get(group_id)
        block.settings_group_id = ""
        remaining = self._group_members(group_id)
        if len(remaining) <= 1:
            for _pipeline, _layer, member in remaining:
                member.settings_group_id = ""
            self._settings_groups.pop(group_id, None)
        elif group is not None and group.reference_block_id == block_id:
            self._promote_group_reference(group, remaining[0])
        elif group is not None:
            group.revision += 1
        return True

    def _on_block_unlink_requested(self, block_id: str) -> None:
        found = self._find_block(block_id)
        group_id = "" if found is None else found[2].settings_group_id
        if not self._detach_settings_group_member(block_id):
            return

        dialog = self._group_editor_dialogs.get(group_id)
        dialog_member_detached = bool(
            dialog is not None
            and dialog.property("pipeline_block_id") == block_id
        )
        group_dissolved = group_id not in self._settings_groups
        if dialog is not None and (dialog_member_detached or group_dissolved):
            for checkbox in dialog.findChildren(QCheckBox):
                if checkbox.text() == PIPELINE_LINK_ALL_TEXT:
                    checkbox.setChecked(False)
                    break
            self._group_editor_dialogs.pop(group_id, None)
        self._refresh_lane_tracks()
        self._refresh_selection()

    def _should_detach_for_linked_param_change(
        self,
        block: PipelineBlock,
        params: dict,
    ) -> bool:
        group = self._settings_groups.get(block.settings_group_id)
        if group is None:
            return False
        shared_params = self._shared_settings(block.tool_id, params)
        if shared_params == group.shared_params:
            return False

        group_label, _color = self._settings_group_visual(group.group_id)
        member_count = len(self._group_members(group.group_id))
        answer = QMessageBox.question(
            self,
            "联动参数已修改",
            (
                f"当前 Block 属于联动组 {group_label}，检测到参数已修改。\n\n"
                "选择「是」：解除当前 Block 的联动，仅当前通道使用新参数。\n"
                f"选择「否」：保持联动，新参数将应用到组内全部 "
                f"{member_count} 个通道。\n\n"
                "是否解除当前 Block 的联动？"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _dissolve_settings_group(self, block: PipelineBlock) -> None:
        group_id = block.settings_group_id
        if not group_id:
            return
        for _pipeline, _layer, member in self._group_members(group_id):
            member.settings_group_id = ""
        self._settings_groups.pop(group_id, None)

    def _set_block_linked(self, block_id: str, params: dict, linked: bool) -> None:
        found = self._find_block(block_id)
        if found is None:
            return
        _pipeline, _layer, block = found
        if not linked:
            self._dissolve_settings_group(block)
            return
        group_id = self._ensure_settings_group(block_id, params)
        if group_id:
            self._update_settings_group(block_id, params)

    def _register_editor_dialog(self, dialog: QDialog, group_id: str = ""):
        self._editor_dialogs.append(dialog)
        if group_id:
            self._group_editor_dialogs[group_id] = dialog
        dialog.finished.connect(
            lambda _result, current=dialog, current_group=group_id:
            self._drop_editor_dialog(current, current_group)
        )

    def _drop_editor_dialog(self, dialog: QDialog, group_id: str = ""):
        self._editor_dialogs = [existing for existing in self._editor_dialogs if existing is not dialog]
        stale_groups = [
            current_group
            for current_group, current_dialog in self._group_editor_dialogs.items()
            if current_dialog is dialog
        ]
        for current_group in stale_groups:
            self._group_editor_dialogs.pop(current_group, None)

    def _open_interactive_editor(
        self,
        pipeline: ChannelPipeline,
        layer: PipelineLayer,
        block: PipelineBlock,
    ):
        existing_dialog = self._group_editor_dialogs.get(block.settings_group_id)
        if existing_dialog is not None:
            existing_block_id = existing_dialog.property("pipeline_block_id")
            if existing_block_id == block.block_id:
                existing_dialog.showNormal()
                existing_dialog.raise_()
                existing_dialog.activateWindow()
                return
            # 联动组共享参数，但预览必须来自用户当前点击的通道。切换组成员时
            # 关闭旧成员的编辑器，再为当前 Block 构造对应通道的预览数据。
            existing_dialog.close()
            self._drop_editor_dialog(existing_dialog, block.settings_group_id)

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
        dialog.setProperty("pipeline_block_id", block.block_id)
        member_count = len(self._group_members(block.settings_group_id))
        group_title = ""
        if member_count > 1:
            group_label, _color = self._settings_group_visual(
                block.settings_group_id
            )
            group_title = f" | 联动组 {group_label} · {member_count} 个通道"
        dialog.setWindowTitle(
            f"{TOOL_META.get(block.tool_id, {}).get('name', block.tool_id)}"
            f" | #{pipeline.data_id} {pipeline.channel} | {layer.label}{group_title}"
        )
        dialog.resize(1200, 720)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)

        link_all_checkbox = create_pipeline_link_all_checkbox(
            dialog,
            checked=bool(block.settings_group_id),
        )

        tool_id = self._canonical_tool_id(block.tool_id)
        editor = tool_registry.create_pipeline_editor(
            tool_id, preview_channel, dialog, params=self._editor_params(block),
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
            def _on_params_applied(values, _broadcast):
                detached = self._on_plugin_params_applied(
                    block_id,
                    values,
                    linked=link_all_checkbox.isChecked(),
                )
                if detached:
                    link_all_checkbox.setChecked(False)
                current = self._find_block(block_id)
                if current is not None and current[2].settings_group_id:
                    self._group_editor_dialogs[current[2].settings_group_id] = dialog

            params_applied.connect(_on_params_applied)

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
        attach_link_checkbox = getattr(
            editor,
            "attach_pipeline_link_checkbox",
            None,
        )
        attached = bool(
            callable(attach_link_checkbox)
            and attach_link_checkbox(link_all_checkbox)
        )
        if not attached:
            action_row = QHBoxLayout()
            action_row.addStretch(1)
            action_row.addWidget(link_all_checkbox)
            layout.addLayout(action_row)
        self._register_editor_dialog(dialog, block.settings_group_id)
        dialog.show()

    def _open_param_editor(
        self,
        pipeline: ChannelPipeline,
        layer: PipelineLayer,
        block: PipelineBlock,
    ):
        plugin = tool_registry.get_plugin(block.tool_id)
        data = self._data_for(pipeline.data_id)
        if plugin is not None and self._canonical_tool_id(block.tool_id) == "kp_selection":
            series = self._channel_series(pipeline.data_id, pipeline.channel)
            if data is None or series is None:
                return
            dialog = KpSelectionDialog(
                f"#{pipeline.data_id} {data.name}",
                series.index,
                getattr(data, "kp_data", None),
                self._editor_params(block),
                self,
                mode="pipeline",
                allow_link_all=True,
                link_all=bool(block.settings_group_id),
            )
        elif plugin is not None:
            dialog = PluginParamDialog(
                plugin.info,
                self._editor_params(block),
                self,
                allow_link_all=True,
                link_all=bool(block.settings_group_id),
            )
            if block.settings_group_id:
                group_label, _color = self._settings_group_visual(
                    block.settings_group_id
                )
                member_count = len(self._group_members(block.settings_group_id))
                dialog.setWindowTitle(
                    f"{plugin.info.display_name} | "
                    f"联动组 {group_label} · {member_count} 个通道"
                )
        else:
            dialog = PipelineParamDialog(
                f"#{pipeline.data_id} {pipeline.channel}",
                layer.label,
                block,
                TOOL_META,
                self,
            )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        link_checkbox = getattr(dialog, "link_all_checkbox", None)
        self._on_plugin_params_applied(
            block.block_id,
            dialog.params,
            linked=bool(link_checkbox is not None and link_checkbox.isChecked()),
        )
        block.last_summary = f"params: {block.params}"
        if self._pending_lane_rebuild:
            self._pending_lane_rebuild = False
            self._rebuild_lanes()
        self._refresh_all_pipeline_statuses()
        self._refresh_selection()

    def _on_plugin_params_applied(
        self,
        block_id: str,
        params: object,
        *,
        linked: bool = False,
    ) -> bool:
        found = self._find_block(block_id)
        if found is None or not isinstance(params, dict):
            return False
        _pipeline, _layer, block = found
        group = self._settings_groups.get(block.settings_group_id)
        linked_params_changed = bool(
            group is not None
            and self._shared_settings(block.tool_id, params) != group.shared_params
        )
        detached = self._should_detach_for_linked_param_change(block, params)
        if detached:
            self._detach_settings_group_member(block_id)

        block.params = dict(params)
        block.status = "configured"
        block.input_slice = None
        block.output_slice = None
        if not detached:
            self._set_block_linked(
                block_id,
                block.params,
                linked=True if linked_params_changed else linked,
            )
        return detached

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
    ) -> tuple[int, int] | None:
        """Map a block's inclusive timestamp span to a target channel slice."""
        source = self._channel_slice(source_pipeline.data_id, source_pipeline.channel)
        target = self._channel_slice(target_pipeline.data_id, target_pipeline.channel)
        if source.empty or target.empty:
            return None

        if is_global_time_tool(source_block.tool_id):
            full_target = self._channel_series(
                target_pipeline.data_id,
                target_pipeline.channel,
            )
            if (
                full_target is None
                or len(full_target) != len(target)
                or not full_target.index.equals(target.index)
            ):
                return None
            return 0, len(target)

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
    ) -> dict | None:
        """Copy serializable params, translating point selections by time."""
        params = deepcopy(source_block.params)
        if source_block.tool_id not in TIMESTAMP_INDEX_TOOLS:
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
        source_pipeline, _source_layer, source_block = found
        if source_block.settings_group_id:
            return
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
        source_pipeline, _source_layer, source_block = found
        if source_block.settings_group_id:
            return
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
                lane.samples = preview
                if lane.overview_widget is not None:
                    lane.overview_widget.set_samples(preview)
            lane.refresh()

    @staticmethod
    def _array_identity(values: np.ndarray | None) -> tuple | None:
        if values is None:
            return None
        return id(values), values.shape, values.dtype.str

    def _pipeline_preview_signature(
        self,
        pipeline: ChannelPipeline,
        stop_before_block_id: str = "",
    ) -> tuple:
        """Return an O(block-count) token for preview/cache invalidation.

        Pipeline arrays and parameter mappings are replaced, not edited in
        place, by this widget.  Their identities therefore provide a stable
        generation token without serializing every sample to bytes.
        """
        blocks: list[tuple] = []
        for layer in pipeline.layers:
            for block in layer.sorted_blocks():
                if stop_before_block_id and block.block_id == stop_before_block_id:
                    return (
                        self._array_identity(pipeline.baseline_snapshot),
                        tuple(blocks),
                    )
                blocks.append((
                    block.block_id,
                    block.tool_id,
                    block.start,
                    block.end,
                    block.status,
                    id(block.params),
                    self._array_identity(block.input_slice),
                    self._array_identity(block.output_slice),
                ))
        return self._array_identity(pipeline.baseline_snapshot), tuple(blocks)

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
        data = self._data_for(pipeline.data_id)
        auxiliary = {}
        if data is not None:
            kp_data = getattr(data, "kp_data", None)
            if isinstance(kp_data, pd.DataFrame):
                auxiliary["kp_data"] = kp_data.copy(deep=True)
        channel_block = ChannelBlock(
            pipeline.channel,
            series,
            ChannelMeta(sample_rate=(1.0 / dt if dt > 0 else 0.0)),
            auxiliary,
        )
        input_generation = self._pipeline_preview_signature(
            pipeline,
            stop_before_block_id=current_block.block_id,
        )
        self._plugin_generation += 1
        generation = self._plugin_generation
        self._plugin_tasks[block.block_id] = generation
        block.last_summary = "processing plugin…"

        bridge = _PluginTaskBridge(
            (block.block_id, generation, input_generation, values),
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
        block_id, expected, input_generation, input_snapshot = metadata
        if self._plugin_tasks.get(block_id) != expected:
            return
        self._plugin_tasks.pop(block_id, None)
        self._plugin_bridges.pop(block_id, None)
        located = self._find_block(block_id)
        if located is None:
            return
        target_pipeline, _target_layer, target = located
        result, completed_params = payload
        latest_generation = self._pipeline_preview_signature(
            target_pipeline,
            stop_before_block_id=block_id,
        )
        if latest_generation != input_generation:
            target.status = "stale"
            target.last_summary = "upstream changed while processing; rerunning"
            self._refresh_all_pipeline_statuses()
            return
        target.params = dict(completed_params)
        target.input_slice = input_snapshot
        target.output_slice = result.series.to_numpy(dtype=np.float64, copy=True)
        target.status = "ready"
        target.last_summary = f"processed {target.output_slice.size} pts"
        self._refresh_all_pipeline_statuses()
        self._refresh_selection()

    def _on_plugin_task_failed(self, envelope) -> None:
        metadata, exc = envelope
        block_id, expected, _input_generation, _input_snapshot = metadata
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
                current_snapshot[start:end] = output_slice
                return current_snapshot
            if update_status:
                block.status = "stale"
                block.last_summary = "upstream changed; rerunning plugin"
                self._schedule_plugin_block(block, current_input)
            return current_snapshot

        if update_status:
            block.status = "ready"
            if cache_matches:
                block.last_summary = f"cached {block.output_slice.size} pts"
        current_snapshot[start:end] = output_slice
        return current_snapshot

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

        use_cache = update_status and not stop_before_block_id
        signature = self._pipeline_preview_signature(pipeline)
        if use_cache:
            cached = self._preview_cache.get(pipeline.ref)
            if cached is not None and cached[0] == signature:
                return cached[1]

        # _pipeline_baseline returns an owned copy.  Blocks can therefore be
        # applied into this single working buffer instead of allocating a full
        # channel copy after every step.
        current_snapshot = self._pipeline_baseline(pipeline)
        for layer in pipeline.layers:
            for block in layer.sorted_blocks():
                if stop_before_block_id and block.block_id == stop_before_block_id:
                    return current_snapshot
                current_snapshot = self._apply_block_preview(block, current_snapshot, update_status)
        if use_cache:
            final_signature = self._pipeline_preview_signature(pipeline)
            self._preview_cache[pipeline.ref] = (final_signature, current_snapshot)
        return current_snapshot

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

    def _clear_blocks_after_commit(self) -> None:
        for pipeline in self.pipelines:
            for layer in pipeline.layers:
                layer.blocks.clear()
        self._settings_groups.clear()
        self.selected_block_id = ""

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

        # Commit is the durability boundary: compose once without the UI cache
        # so a caller that edited an ndarray in place cannot make a stale
        # preview authoritative.  Routine screen refreshes still use the fast
        # identity/version cache above.
        preview_results = self._collect_preview_results(update_status=False)

        changed_channels: list[str] = []
        changed_data_ids: set[int] = set()
        committed_pipelines: list[ChannelPipeline] = []

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

            if self.data_manager is not None:
                self.data_manager.capture_channel_processing_baseline(
                    pipeline.data_id,
                    pipeline.channel,
                )
            updated_series = data.get_channel_series(pipeline.channel).copy()
            updated_series.loc[target_series.index] = preview_snapshot
            data.set_channel_series(pipeline.channel, updated_series)
            changed_channels.append(f"#{pipeline.data_id} {pipeline.channel}")
            changed_data_ids.add(pipeline.data_id)
            committed_pipelines.append(pipeline)
            self.change_finished_signal.emit(pipeline.data_id, pipeline.channel)

        if not changed_channels:
            QMessageBox.information(self, "Commit", "没有可提交的新 preview 结果。")
            self._refresh_all_pipeline_statuses()
            return

        if self.data_manager is not None:
            for changed_data_id in sorted(changed_data_ids):
                self.data_manager.notify_changed(changed_data_id, structural=False)

        if self.clear_blocks_after_commit_checkbox.isChecked():
            self._clear_blocks_after_commit()
            commit_detail = "处理结果已合并到 baseline，当前窗口的 block 已清空。"
        else:
            for pipeline in committed_pipelines:
                self._reset_pipeline_after_commit(pipeline)
            commit_detail = "处理结果已合并到 baseline；继续处理时请重新打开对应 block。"

        self._rebuild_lanes()
        self._refresh_all_pipeline_statuses()
        self._refresh_selection()
        QMessageBox.information(
            self,
            "Commit",
            "已写回 channel:\n"
            + "\n".join(changed_channels)
            + "\n\n"
            + commit_detail,
        )

    def _refresh_selection(self):
        found = self._find_block(self.selected_block_id)
        if found is None:
            for lane in self._lane_widgets:
                lane.set_selected("", "")
            if self.pipeline_hint is not None:
                self.pipeline_hint.setText(
                    "将左侧“数据处理工具”拖到某个 step 轨道以添加 block。"
                )
            return

        pipeline, layer, block = found
        for lane in self._lane_widgets:
            lane.set_selected(block.block_id, block.settings_group_id)

        if self.pipeline_hint is not None:
            meta = TOOL_META.get(block.tool_id, {})
            group_members = len(self._group_members(block.settings_group_id))
            group_text = ""
            if group_members > 1:
                group_label, _color = self._settings_group_visual(
                    block.settings_group_id
                )
                group_text = f" / 联动组 {group_label} · {group_members} 个通道"
            self.pipeline_hint.setText(
                f"#{pipeline.data_id} {pipeline.channel} / {layer.label} / "
                f"{meta.get('name', block.tool_id)} / {block.status}{group_text}"
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
