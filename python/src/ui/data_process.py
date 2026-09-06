from __future__ import annotations

import json
from concurrent.futures import as_completed
from dataclasses import dataclass
import sys
from typing import Callable, Optional

import numpy as np
import pandas as pd
import xarray as xr

from PySide6.QtCore import QPoint, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMdiArea,
    QMdiSubWindow,
    QMessageBox,
    QProgressDialog,
    QTabBar,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from base.processing_broker import ProcessingBroker

from base.data_manager import DataManager
from core.em_data import EMData
from ui.data_process_pipeline import DataProcessPipelineWidget, pipeline_tool_id_from_mime
from ui.batch_process_dialog import BatchProcessDialog
from ui.data_process_shared import (
    CLASSIC_TOOL_MIME,
    DATA_TREE_SELECTION_MIME,
    create_classic_tool_model,
)
from ui.tools import registry as tool_registry
from utils.concurrency import compute_pool


@dataclass(frozen=True)
class ToolWindowState:
    tool_type: str
    data_id: int
    channel_name: str


def _mdi_target_size(
    mdi_area: QMdiArea,
    min_width: int,
    min_height: int,
) -> tuple[int, int]:
    viewport_size = mdi_area.viewport().size()
    available_width = viewport_size.width() or mdi_area.size().width()
    available_height = viewport_size.height() or mdi_area.size().height()
    target_width = max(min_width, available_width - 24) if available_width > 0 else min_width
    target_height = max(min_height, available_height - 24) if available_height > 0 else min_height
    return target_width, target_height


class ClassicMdiArea(QMdiArea):
    def __init__(self, parent: "DataProcessLegacyWidget" = None):
        super().__init__(parent)
        self.parent_widget = parent
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(CLASSIC_TOOL_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasFormat(CLASSIC_TOOL_MIME):
            return

        widget_type = event.mimeData().data(CLASSIC_TOOL_MIME).data().decode()
        position = event.position().toPoint()

        if tool_registry.is_station_tool(widget_type):
            # Station tools do NOT require channel context — use dataset directly.
            if self.parent_widget.data_id != 0 and self.parent_widget.em_data is not None:
                data_id = self.parent_widget.data_id
            else:
                return
            sub_window = self.parent_widget.open_tool_window(
                widget_type, data_id, None, position=position,
            )
        else:
            target = self.parent_widget.current_context()
            if target is None:
                return
            data_id, channel_name = target
            sub_window = self.parent_widget.open_tool_window(
                widget_type, data_id, channel_name, position=position,
            )
        if sub_window:
            self.parent_widget.present_subwindow(sub_window)
            event.acceptProposedAction()


class DataProcessLegacyWidget(QWidget):
    change_finished_signal = Signal(int, str)

    def __init__(self, data_manager: Optional[DataManager] = None, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager

        # 创建数据处理中介（后台线程）
        self.broker = ProcessingBroker(data_manager) if data_manager else None
        self._broker_thread: Optional[QThread] = None
        if self.broker is not None:
            self._broker_thread = QThread(self)
            self.broker.moveToThread(self._broker_thread)
            self._broker_thread.start()
            # Broker 数据变更 → 刷新 UI
            self.broker.data_changed.connect(self.on_change_data_finished_for)
            # Broker 结果确认 → 关闭对应工具窗口
            self.broker.result_applied.connect(self._on_tool_result_applied)
            # 应用退出时安全停止后台线程
            QApplication.instance().aboutToQuit.connect(self._stop_broker)

        self.mdi_area: Optional[ClassicMdiArea] = None
        self.data_id: int = 0
        self.em_data: Optional[EMData] = None
        self._selected_channel_name: Optional[str] = None
        self._context_resolver: Optional[Callable[[], Optional[tuple[int, str]]]] = None
        self._init_ui()

    def _stop_broker(self) -> None:
        if self._broker_thread is not None and self._broker_thread.isRunning():
            self._broker_thread.quit()
            if not self._broker_thread.wait(1000):
                self._broker_thread.terminate()
                self._broker_thread.wait()

    def _init_ui(self):
        self.mdi_area = ClassicMdiArea(self)
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.mdi_area)
        self.setLayout(main_layout)

    def _create_placeholder_tool_widget(self, title: str, description: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title_label = QLabel(title, widget)
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        body_label = QLabel(description, widget)
        body_label.setWordWrap(True)
        body_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(title_label)
        layout.addWidget(body_label)
        layout.addStretch(1)
        return widget

    def init_data(self, data, data_id: int = 0):
        self.data_id = data_id
        self.em_data = data
        if self.em_data is None:
            self._selected_channel_name = None
        self._close_invalid_tool_windows()

    @staticmethod
    def create_tool_model() -> QStandardItemModel:
        return create_classic_tool_model()

    def set_context_resolver(
        self,
        resolver: Optional[Callable[[], Optional[tuple[int, str]]]],
    ) -> None:
        self._context_resolver = resolver

    def current_context(self) -> Optional[tuple[int, str]]:
        if self._context_resolver is not None:
            target = self._context_resolver()
            if target is not None:
                data_id, channel_name = target
                data = self.data_manager.get(data_id) if self.data_manager is not None else None
                if data is not None:
                    self.data_id = data_id
                    self.em_data = data
                    self._selected_channel_name = channel_name
                return target

        channel_name = self.current_channel()
        if self.data_id == 0 or channel_name is None:
            return None
        return self.data_id, channel_name

    def current_channel(self) -> Optional[str]:
        if self.em_data is None:
            return None

        if self._selected_channel_name is not None:
            if self.em_data.has_channel(self._selected_channel_name):
                return self._selected_channel_name

        first_channel_name = self.em_data.first_channel_name()
        if first_channel_name is None:
            return None
        self._selected_channel_name = first_channel_name
        return first_channel_name

    def _apply_subwindow_size(
        self,
        sub_window: QMdiSubWindow,
        position: Optional[QPoint] = None,
    ) -> None:
        if self.mdi_area is None:
            return
        target_width, target_height = _mdi_target_size(self.mdi_area, 900, 680)
        if position is not None:
            sub_window.setGeometry(position.x(), position.y(), target_width, target_height)
        else:
            sub_window.resize(target_width, target_height)

    def present_subwindow(self, sub_window: Optional[QMdiSubWindow]) -> None:
        if sub_window is None or self.mdi_area is None:
            return
        sub_window.show()
        sub_window.showMaximized()
        self.mdi_area.setActiveSubWindow(sub_window)

    def open_tool_window(
        self,
        tool_type: str,
        data_id: int | str,
        channel_name: Optional[str] = None,
        position: Optional[QPoint] = None,
    ) -> Optional[QMdiSubWindow]:
        is_station_tool = (
            tool_registry.is_station_tool(tool_type)
            or tool_registry.is_station_tool(self._canonical_tool_type(tool_type))
        )

        if isinstance(data_id, str) and channel_name is None:
            channel_name = data_id
            data_id = self.data_id

        if not isinstance(data_id, int) or self.mdi_area is None:
            return None
        if not is_station_tool and channel_name is None:
            return None

        data = self.data_manager.get(data_id) if self.data_manager is not None else None
        if data is None and data_id == self.data_id:
            data = self.em_data
        if data is None:
            return None

        if not is_station_tool and not data.has_channel(channel_name):
            return None

        tool_key = self._canonical_tool_type(tool_type)
        if is_station_tool:
            lookup_channel = ""
        else:
            lookup_channel = channel_name or ""
        existing = self._find_tool_window(tool_key, data_id, lookup_channel)
        if existing is not None:
            content_widget = existing.widget()
            if content_widget is not None and content_widget.isHidden():
                content_widget.show()
            existing.show()
            self.mdi_area.setActiveSubWindow(existing)
            return existing

        sub_window = QMdiSubWindow()
        sub_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        sub_window.setWindowFlags(
            sub_window.windowFlags() & ~Qt.WindowType.WindowSystemMenuHint
        )
        sub_window.setProperty("tool_type", tool_type)
        sub_window.setProperty("tool_key", tool_key)
        sub_window.setProperty("data_id", data_id)
        if is_station_tool:
            sub_window.setProperty("channel_name", "")
            sub_window.setWindowTitle(f"{tool_type} - #{data_id} {data.name}")
        else:
            sub_window.setProperty("channel_name", channel_name)
            sub_window.setWindowTitle(f"{tool_type} - #{data_id} {data.name} - {channel_name}")
        self._apply_subwindow_size(sub_window, position)

        # ─── 创建工具并接线到 Broker ───────────────────────
        spec = tool_registry.get(tool_key)
        content_widget = tool_registry.create_classic_widget(tool_key, data_id, channel_name)
        if content_widget is None:
            content_widget = QTextEdit()
            content_widget.setPlainText(f"未知工具类型: {tool_type}")
        else:
            wire_channel = "" if is_station_tool else (channel_name or "")
            self._wire_tool(content_widget, data_id, wire_channel)
            if spec is not None and getattr(content_widget, "channel_sync_requested", None) is not None and spec.has_channel_sync:
                content_widget.channel_sync_requested.connect(self.broker.apply_result_all_channels)
            if spec is not None and getattr(content_widget, "region_sync_requested", None) is not None and spec.has_region_sync:
                content_widget.region_sync_requested.connect(self._on_region_sync)

        sub_window.setWidget(content_widget)
        self.mdi_area.addSubWindow(sub_window)
        return sub_window

    def open_tool_window_states(self) -> list[ToolWindowState]:
        if self.mdi_area is None:
            return []

        states: list[ToolWindowState] = []
        for sub_window in self.mdi_area.subWindowList():
            tool_type = sub_window.property("tool_type")
            data_id = sub_window.property("data_id")
            channel_name = sub_window.property("channel_name")
            if isinstance(tool_type, str) and isinstance(data_id, int) and isinstance(channel_name, str):
                states.append(ToolWindowState(tool_type, data_id, channel_name))
        return states

    def _wire_tool(self, tool, data_id: int, channel_name: str) -> None:
        """将工具的信号接线到 Broker，然后触发数据请求。"""
        tool.data_requested.connect(self.broker.request_data)
        tool.result_submitted.connect(self.broker.apply_result)
        self.broker.data_ready.connect(tool._on_data_ready)
        tool.data_requested.emit(data_id, channel_name)

    def on_change_data_finished_for(self, data_id: int, channel: str, result):
        if data_id == 0 or self.data_manager is None:
            return
        data = self.data_manager.get(data_id)
        if data is None:
            return
        existing_series = data.get_channel_series(channel)
        if len(existing_series) == len(result):
            data.set_channel_series(channel, pd.Series(
                result,
                index=existing_series.index,
                name=existing_series.name,
            ))
        else:
            data.set_channel_data(channel, result)
        self.change_finished_signal.emit(data_id, channel)

    def _on_region_sync(self, params: dict) -> None:
        """RemoveRegion 跨数据集同步——线程池并行，每个数据集一个 worker。"""
        import logging
        from core.region_sync import compute_time_range_intervals
        _log = logging.getLogger(__name__)
        time_ranges: list[tuple[float, float]] = params["time_ranges"]
        source_id: int = params.get("source_data_id", -1)
        _log.info("_on_region_sync START: source=#%d, ranges=%s", source_id, time_ranges)

        # ── Phase 1: 准备任务 ──────────────────────────
        tasks = []
        for data_id, em_data in self.data_manager.items():
            if data_id == source_id:
                continue
            npts = em_data.npts
            start_ts = em_data.start_time.timestamp()
            dt_sec = em_data.dt.total_seconds()
            intervals = compute_time_range_intervals(time_ranges, npts, start_ts, dt_sec)
            if intervals:
                tasks.append((data_id, em_data, intervals))
            else:
                _log.info("_on_region_sync: skip #%d (no time overlap)", data_id)

        _log.info("_on_region_sync: %d/%d datasets have overlapping ranges",
                  len(tasks), len(list(self.data_manager.items())) - 1)
        if not tasks:
            return

        # ── Phase 2: 线程池并行修改 + 预计算显示数据 ────
        from plugins.common.display import decimate_for_display
        from utils.series import dti_to_numpy, safe_nanmean

        def _modify(task):
            data_id, em_data, intervals = task
            x = dti_to_numpy(em_data.datetime_index)
            display = {}
            for ch in em_data.channel_names():
                vals = em_data.data[ch].values.copy()
                for left, right in intervals:
                    vals[left:right] = np.nan
                em_data.set_channel_series(ch, pd.Series(
                    vals, index=em_data.get_channel_series(ch).index, name=ch,
                ))
                m = safe_nanmean(vals)
                dy = vals - (m if np.isfinite(m) else 0.0)
                dx, dy_d = decimate_for_display(x, dy)
                display[ch] = (dx, dy_d)
            return (data_id, display)

        # ── 进度弹窗：Phase 2 ────────────────────────
        total = len(tasks)
        progress = QProgressDialog("正在同步到其他数据集...", None, 0, total, self)
        progress.setWindowTitle("批量同步")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()

        # 用全局共享 compute 池（大小 max(1, cpu//2)，与原每动作池同策）。
        # _modify 在池线程里原地改各数据集通道并预抽稀显示数据；
        # 池尺寸即并发上限，不再每动作新建 executor。
        displays: dict[int, dict] = {}  # data_id → {ch: (dx, dy)}
        completed = 0
        futures = {compute_pool.submit(_modify, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                data_id, display = future.result()
                displays[data_id] = display
            except Exception:
                data_id = futures[future][0]
                _log.exception("_on_region_sync: failed for dataset #%d", data_id)
            completed += 1
            progress.setValue(completed)
            progress.setLabelText(f"已完成 {completed}/{total} 个数据集")

        progress.setLabelText(f"全部完成 ({total} 个数据集)")
        progress.setValue(total)

        # ── Phase 3: 主线程快速应用显示数据 ────────────
        self._apply_displays(displays, progress)
        QTimer.singleShot(5000, progress.close)

    def _apply_displays(self, displays: dict, progress) -> None:
        """在主线程应用预计算的显示数据。"""
        from PySide6.QtWidgets import QApplication
        mainwindow = None
        for widget in QApplication.instance().topLevelWidgets():
            if hasattr(widget, 'data_viewers'):
                mainwindow = widget
                break
        if mainwindow is None:
            return

        applied = 0
        for data_id, channels in displays.items():
            viewer = mainwindow.data_viewers.get(data_id)
            if viewer is None:
                continue
            for ch, (dx, dy) in channels.items():
                if ch in viewer.plot_rows:
                    widget = viewer.plot_rows[ch].widget
                    if ch in widget.curves:
                        widget.curves[ch].setData(dx, dy)
            applied += 1
            progress.setLabelText(f"正在刷新波形 {applied}/{len(displays)} 个数据集")
        progress.setLabelText(f"刷新完成 ({len(displays)} 个数据集)")

    def _on_tool_result_applied(self, data_id: int, channel: str) -> None:
        """Broker 确认结果写入后，关闭对应工具的子窗口。"""
        if self.mdi_area is None:
            return
        for sub_window in self.mdi_area.subWindowList():
            if (sub_window.property("data_id") == data_id
                    and sub_window.property("channel_name") == channel):
                sub_window.close()
                break

    def _find_tool_window(
        self,
        tool_type: str,
        data_id: int,
        channel_name: str,
    ) -> Optional[QMdiSubWindow]:
        if self.mdi_area is None:
            return None
        for sub_window in self.mdi_area.subWindowList():
            if (
                sub_window.property("tool_key") == tool_type
                and sub_window.property("data_id") == data_id
                and sub_window.property("channel_name") == channel_name
            ):
                return sub_window
        return None

    def close_windows_for_data(self, data_id: int):
        if self.mdi_area is None:
            return
        for sub_window in list(self.mdi_area.subWindowList()):
            if sub_window.property("data_id") == data_id:
                self._dispose_sub_window(sub_window)

    def _canonical_tool_type(self, tool_type: str) -> str:
        return tool_registry.canonical_display_name(tool_type)

    def _close_invalid_tool_windows(self):
        if self.mdi_area is None:
            return
        if self.em_data is None:
            for sub_window in list(self.mdi_area.subWindowList()):
                self._dispose_sub_window(sub_window)
            return

        valid_channels = set(self.em_data.channel_names())
        for sub_window in list(self.mdi_area.subWindowList()):
            data_id = sub_window.property("data_id")
            channel_name = sub_window.property("channel_name")
            if data_id == self.data_id and isinstance(channel_name, str) and channel_name and channel_name not in valid_channels:
                self._dispose_sub_window(sub_window)

    def _dispose_sub_window(self, sub_window: QMdiSubWindow):
        if self.mdi_area is not None:
            self.mdi_area.removeSubWindow(sub_window)
        sub_window.close()

    def closeEvent(self, event):
        if self.mdi_area is not None:
            for sub_window in list(self.mdi_area.subWindowList()):
                self._dispose_sub_window(sub_window)
        super().closeEvent(event)


class WorkspaceMdiArea(QMdiArea):
    change_finished_signal = Signal(int, str)

    def __init__(self, data_manager: Optional[DataManager] = None, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.data_id = 0
        self.em_data: Optional[EMData] = None
        self._context_resolver: Optional[Callable[[], Optional[tuple[int, str]]]] = None
        self._selection_payload_resolver: Optional[Callable[[], Optional[dict]]] = None
        self._clear_checked_channels: Optional[Callable[[list[tuple[int, str]]], None]] = None
        self._last_pipeline_window: Optional[DataProcessPipelineWidget] = None
        self.setAcceptDrops(True)

    def init_data(self, data, data_id: int = 0):
        self.data_id = data_id
        self.em_data = data

    def set_context_resolver(
        self,
        resolver: Optional[Callable[[], Optional[tuple[int, str]]]],
    ) -> None:
        self._context_resolver = resolver

    def set_selection_payload_resolver(
        self,
        resolver: Optional[Callable[[], Optional[dict]]],
    ) -> None:
        self._selection_payload_resolver = resolver

    def set_clear_checked_channels(
        self,
        clearer: Optional[Callable[[list[tuple[int, str]]], None]],
    ) -> None:
        self._clear_checked_channels = clearer

    def active_pipeline_window(self) -> Optional[DataProcessPipelineWidget]:
        active = self.activeSubWindow()
        if active is None:
            return None
        widget = active.widget()
        return widget if isinstance(widget, DataProcessPipelineWidget) else None

    def last_pipeline_window(self) -> Optional[DataProcessPipelineWidget]:
        return self._last_pipeline_window

    def pipeline_subwindow(self, pipeline_widget: DataProcessPipelineWidget) -> Optional[QMdiSubWindow]:
        for sub_window in self.subWindowList():
            if sub_window.widget() is pipeline_widget:
                return sub_window
        return None

    def activate_pipeline_window(self, pipeline_widget: Optional[DataProcessPipelineWidget]) -> bool:
        if pipeline_widget is None:
            return False
        sub_window = self.pipeline_subwindow(pipeline_widget)
        if sub_window is None:
            return False
        sub_window.show()
        self.setActiveSubWindow(sub_window)
        self._last_pipeline_window = pipeline_widget
        return True

    def _apply_subwindow_size(
        self,
        sub_window: QMdiSubWindow,
        position: Optional[QPoint] = None,
    ) -> None:
        target_width, target_height = _mdi_target_size(self, 1100, 720)
        if position is not None:
            sub_window.setGeometry(position.x(), position.y(), target_width, target_height)
        else:
            sub_window.resize(target_width, target_height)

    def _present_subwindow(self, sub_window: Optional[QMdiSubWindow]) -> None:
        if sub_window is None:
            return
        sub_window.show()
        sub_window.showMaximized()
        self.setActiveSubWindow(sub_window)

    def dragEnterEvent(self, event):
        if (
            event.mimeData().hasFormat(DATA_TREE_SELECTION_MIME)
            or pipeline_tool_id_from_mime(event.mimeData()) is not None
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if (
            event.mimeData().hasFormat(DATA_TREE_SELECTION_MIME)
            or pipeline_tool_id_from_mime(event.mimeData()) is not None
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasFormat(DATA_TREE_SELECTION_MIME):
            self._handle_data_selection_drop(event)
            return

        tool_id = pipeline_tool_id_from_mime(event.mimeData())
        if tool_id is not None:
            drop_point = event.position().toPoint()
            sub_window = self._subwindow_at(drop_point)
            target = None if sub_window is None else sub_window.widget()
            global_pos = self.mapToGlobal(drop_point)
            if (
                isinstance(target, DataProcessPipelineWidget)
                and target.drop_tool_at_global_position(tool_id, global_pos)
            ):
                event.acceptProposedAction()
                return
            event.ignore()
            return

        event.ignore()

    def _handle_data_selection_drop(self, event):
        payload = None
        if event.mimeData().hasFormat(DATA_TREE_SELECTION_MIME):
            try:
                raw_payload = bytes(event.mimeData().data(DATA_TREE_SELECTION_MIME)).decode("utf-8")
                payload = json.loads(raw_payload)
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = None

        if payload is None and self._selection_payload_resolver is not None:
            payload = self._selection_payload_resolver()

        if payload is None:
            event.ignore()
            return

        refs = self._channel_refs_from_payload(payload)
        if not refs:
            event.ignore()
            return

        drop_position = event.position()
        if hasattr(drop_position, "toPoint"):
            drop_point = drop_position.toPoint()
        else:
            drop_point = drop_position

        target_subwindow = self._subwindow_at(drop_point)
        target_widget = None if target_subwindow is None else target_subwindow.widget()

        if isinstance(target_widget, DataProcessPipelineWidget):
            added, duplicated, failed = self.add_channels_to_pipeline_window(target_widget, refs)
        else:
            pipeline_widget = self.create_pipeline_window(refs, position=drop_point)
            if pipeline_widget is None:
                added, duplicated, failed = [], [], refs
            else:
                added = pipeline_widget.channel_refs()
                duplicated = []
                failed = [ref for ref in refs if ref not in added]

        if added and self._clear_checked_channels is not None:
            self._clear_checked_channels(added)

        if duplicated or failed:
            parts = []
            if duplicated:
                dup_text = "\n".join(f"#{data_id} {channel_name}" for data_id, channel_name in duplicated)
                parts.append(f"已存在，已忽略:\n{dup_text}")
            if failed:
                fail_text = "\n".join(f"#{data_id} {channel_name}" for data_id, channel_name in failed)
                parts.append(f"时间范围不兼容，未加入:\n{fail_text}")
            QMessageBox.information(self, "部分通道未加入", "\n\n".join(parts))

        if added:
            event.acceptProposedAction()
        else:
            event.ignore()

    def _subwindow_at(self, pos: QPoint) -> Optional[QMdiSubWindow]:
        widget = self.childAt(pos)
        while widget is not None and not isinstance(widget, QMdiSubWindow):
            widget = widget.parentWidget()
        return widget if isinstance(widget, QMdiSubWindow) else None

    def _channel_refs_from_payload(self, payload: dict) -> list[tuple[int, str]]:
        kind = payload.get("kind")
        if kind == "dataset":
            data_id = payload.get("data_id")
            if not isinstance(data_id, int) or self.data_manager is None:
                return []
            em_data = self.data_manager.get(data_id)
            if em_data is None:
                return []
            return [(data_id, channel_name) for channel_name in em_data.channel_names()]

        refs = payload.get("refs")
        if kind == "channels" and isinstance(refs, list):
            result = []
            for item in refs:
                if not isinstance(item, dict):
                    continue
                data_id = item.get("data_id")
                channel_name = item.get("channel_name")
                if isinstance(data_id, int) and isinstance(channel_name, str):
                    result.append((data_id, channel_name))
            return result
        return []

    def create_pipeline_window(
        self,
        channel_refs: list[tuple[int, str]],
        position: Optional[QPoint] = None,
    ) -> Optional[DataProcessPipelineWidget]:
        pipeline_widget = DataProcessPipelineWidget(self.data_manager, self)
        pipeline_widget.data_id = self.data_id
        pipeline_widget.em_data = self.em_data
        pipeline_widget.change_finished_signal.connect(self.change_finished_signal)
        added, duplicated, failed = pipeline_widget.add_channels(channel_refs)
        if not added:
            pipeline_widget.deleteLater()
            return None

        sub_window = QMdiSubWindow()
        sub_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        sub_window.setWindowFlags(
            sub_window.windowFlags() & ~Qt.WindowType.WindowSystemMenuHint
        )
        sub_window.setWidget(pipeline_widget)
        sub_window.setWindowTitle(pipeline_widget.window_title())
        self._apply_subwindow_size(sub_window, position)
        self.addSubWindow(sub_window)
        self._present_subwindow(sub_window)
        self._last_pipeline_window = pipeline_widget

        if duplicated or failed:
            parts = []
            if duplicated:
                dup_text = "\n".join(f"#{data_id} {channel_name}" for data_id, channel_name in duplicated)
                parts.append(f"已存在，已忽略:\n{dup_text}")
            if failed:
                fail_text = "\n".join(f"#{data_id} {channel_name}" for data_id, channel_name in failed)
                parts.append(f"时间范围不兼容，未加入:\n{fail_text}")
            QMessageBox.information(self, "部分通道未加入", "\n\n".join(parts))
        return pipeline_widget

    def ensure_pipeline_window(self) -> Optional[DataProcessPipelineWidget]:
        if self.em_data is None or self.data_id == 0:
            return None

        channel_refs = [(self.data_id, channel_name) for channel_name in self.em_data.channel_names()]
        if not channel_refs:
            return None

        for sub_window in self.subWindowList():
            pipeline_widget = sub_window.widget()
            if (
                isinstance(pipeline_widget, DataProcessPipelineWidget)
                and any(data_id == self.data_id for data_id, _channel_name in pipeline_widget.channel_refs())
            ):
                self.activate_pipeline_window(pipeline_widget)
                return pipeline_widget

        return self.create_pipeline_window(channel_refs)

    def add_channels_to_pipeline_window(
        self,
        pipeline_widget: DataProcessPipelineWidget,
        channel_refs: list[tuple[int, str]],
    ) -> tuple[list[tuple[int, str]], list[tuple[int, str]], list[tuple[int, str]]]:
        added, duplicated, failed = pipeline_widget.add_channels(channel_refs)
        subwindow = pipeline_widget.parentWidget()
        while subwindow is not None and not isinstance(subwindow, QMdiSubWindow):
            subwindow = subwindow.parentWidget()
        if isinstance(subwindow, QMdiSubWindow):
            subwindow.setWindowTitle(pipeline_widget.window_title())
        self._last_pipeline_window = pipeline_widget
        return added, duplicated, failed

    def close_pipeline_windows_for_data(self, data_id: int):
        for sub_window in list(self.subWindowList()):
            widget = sub_window.widget()
            if isinstance(widget, DataProcessPipelineWidget):
                widget.remove_data(data_id)
                if widget.channel_count() == 0:
                    self.removeSubWindow(sub_window)
                    sub_window.close()

    def closeEvent(self, event):
        for sub_window in list(self.subWindowList()):
            self.removeSubWindow(sub_window)
            sub_window.close()
        super().closeEvent(event)


class WorkspaceTabPanel(QWidget):
    change_finished_signal = Signal(int, str)

    def __init__(self, data_manager: Optional[DataManager] = None, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.data_id = 0
        self.em_data: Optional[EMData] = None
        self._context_resolver: Optional[Callable[[], Optional[tuple[int, str]]]] = None
        self._selection_payload_resolver: Optional[Callable[[], Optional[dict]]] = None
        self._clear_checked_channels: Optional[Callable[[list[tuple[int, str]]], None]] = None
        self._workspace_counter = 0
        self._workspaces: list[QWidget] = []
        self._current_workspace_index = 0
        self.workspace_tabs = QTabWidget(self)
        self.workspace_tabs.setMovable(True)
        self.workspace_tabs.setTabsClosable(True)
        self.workspace_tabs.currentChanged.connect(self._on_tab_changed)
        self.workspace_tabs.tabCloseRequested.connect(self._close_workspace)
        self._creating_workspace = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.workspace_tabs, 1)

        self._ensure_add_tab()
        self._create_workspace()

    def _tab_label(self, number: int) -> str:
        return f"工作区 {number}"

    def _ensure_add_tab(self):
        if self.workspace_tabs.count() == 0 or self.workspace_tabs.tabText(self.workspace_tabs.count() - 1) != "+":
            current_index = self.workspace_tabs.currentIndex()
            self.workspace_tabs.blockSignals(True)
            self.workspace_tabs.addTab(QWidget(self.workspace_tabs), "+")
            if current_index >= 0:
                self.workspace_tabs.setCurrentIndex(current_index)
            self.workspace_tabs.blockSignals(False)
        self._update_tab_buttons()

    def _workspace_container(self, mdi_area: WorkspaceMdiArea) -> QWidget:
        container = QWidget(self.workspace_tabs)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(mdi_area, 1)
        container.setLayout(layout)
        container.setProperty("mdi_area", mdi_area)
        return container

    def _create_workspace(self) -> WorkspaceMdiArea:
        if not hasattr(self, "_workspaces"):
            self._workspaces: list[QWidget] = []
            self._current_workspace_index = 0
        self._workspace_counter += 1
        mdi_area = WorkspaceMdiArea(self.data_manager, self.workspace_tabs)
        mdi_area.set_context_resolver(self._context_resolver)
        mdi_area.set_selection_payload_resolver(self._selection_payload_resolver)
        mdi_area.set_clear_checked_channels(self._clear_checked_channels)
        mdi_area.change_finished_signal.connect(self.change_finished_signal)
        mdi_area.init_data(self.em_data, self.data_id)
        container = self._workspace_container(mdi_area)
        self._workspaces.append(container)
        insert_index = max(0, self.workspace_tabs.count() - 1)
        self.workspace_tabs.blockSignals(True)
        self.workspace_tabs.insertTab(insert_index, container, self._tab_label(self._workspace_counter))
        self.workspace_tabs.setCurrentIndex(insert_index)
        self.workspace_tabs.blockSignals(False)
        self._current_workspace_index = len(self._workspaces) - 1
        self._update_tab_buttons()
        mdi_area.ensure_pipeline_window()
        return mdi_area

    def _update_tab_buttons(self) -> None:
        tab_bar = self.workspace_tabs.tabBar()
        add_index = self.workspace_tabs.count() - 1
        show_close = len(self._workspaces) > 1

        for index in range(self.workspace_tabs.count()):
            for side in (QTabBar.LeftSide, QTabBar.RightSide):
                button = tab_bar.tabButton(index, side)
                if button is None:
                    continue
                if index == add_index:
                    button.hide()
                else:
                    button.setVisible(show_close)

    def _close_workspace(self, index: int) -> None:
        add_index = self.workspace_tabs.count() - 1
        if index < 0 or index >= add_index:
            return
        if len(self._workspaces) <= 1:
            self._update_tab_buttons()
            return

        container = self._workspaces.pop(index)
        mdi_area = container.property("mdi_area")
        if isinstance(mdi_area, WorkspaceMdiArea):
            mdi_area.close()

        self.workspace_tabs.blockSignals(True)
        self.workspace_tabs.removeTab(index)
        next_index = min(index, len(self._workspaces) - 1)
        self.workspace_tabs.setCurrentIndex(max(0, next_index))
        self.workspace_tabs.blockSignals(False)

        container.deleteLater()
        self._current_workspace_index = max(0, min(next_index, len(self._workspaces) - 1))
        self._update_tab_buttons()

    def _on_tab_changed(self, index: int):
        add_index = self.workspace_tabs.count() - 1
        if index == add_index:
            if self._creating_workspace:
                return
            self._creating_workspace = True
            try:
                self._create_workspace()
            finally:
                self._creating_workspace = False
            return
        if 0 <= index < add_index:
            self._current_workspace_index = index

    def current_workspace(self) -> Optional[WorkspaceMdiArea]:
        if not self._workspaces:
            return None
        index = min(self._current_workspace_index, len(self._workspaces) - 1)
        container = self._workspaces[index]
        widget = container.property("mdi_area")
        return widget if isinstance(widget, WorkspaceMdiArea) else None

    def init_data(self, data, data_id: int = 0):
        self.data_id = data_id
        self.em_data = data
        for container in self._workspaces:
            mdi_area = container.property("mdi_area")
            if isinstance(mdi_area, WorkspaceMdiArea):
                mdi_area.init_data(data, data_id)

    def set_context_resolver(
        self,
        resolver: Optional[Callable[[], Optional[tuple[int, str]]]],
    ) -> None:
        self._context_resolver = resolver
        for container in self._workspaces:
            mdi_area = container.property("mdi_area")
            if isinstance(mdi_area, WorkspaceMdiArea):
                mdi_area.set_context_resolver(resolver)

    def set_selection_payload_resolver(
        self,
        resolver: Optional[Callable[[], Optional[dict]]],
    ) -> None:
        self._selection_payload_resolver = resolver
        for container in self._workspaces:
            mdi_area = container.property("mdi_area")
            if isinstance(mdi_area, WorkspaceMdiArea):
                mdi_area.set_selection_payload_resolver(resolver)

    def set_clear_checked_channels(
        self,
        clearer: Optional[Callable[[list[tuple[int, str]]], None]],
    ) -> None:
        self._clear_checked_channels = clearer
        for container in self._workspaces:
            mdi_area = container.property("mdi_area")
            if isinstance(mdi_area, WorkspaceMdiArea):
                mdi_area.set_clear_checked_channels(clearer)

    def close_windows_for_data(self, data_id: int):
        for container in self._workspaces:
            mdi_area = container.property("mdi_area")
            if not isinstance(mdi_area, WorkspaceMdiArea):
                continue
            mdi_area.close_pipeline_windows_for_data(data_id)


class DataProcessWidget(QWidget):
    change_finished_signal = Signal(int, str)

    def __init__(self, data_manager: Optional[DataManager] = None, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.data_id = 0
        self.em_data: Optional[EMData] = None
        self._context_resolver: Optional[Callable[[], Optional[tuple[int, str]]]] = None
        self._selection_payload_resolver: Optional[Callable[[], Optional[dict]]] = None
        self._clear_checked_channels: Optional[Callable[[list[tuple[int, str]]], None]] = None
        self.batch_button = QToolButton(self)
        self.batch_button.setText("批处理…")
        self.batch_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.batch_button.clicked.connect(self._open_batch_process)

        self.pipeline_panel = WorkspaceTabPanel(data_manager, self)
        self.pipeline_panel.change_finished_signal.connect(self.change_finished_signal)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.setSpacing(10)
        header_layout.addWidget(QLabel("Pipeline", self))
        header_layout.addStretch(1)
        header_layout.addWidget(self.batch_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(header_layout)
        layout.addWidget(self.pipeline_panel, 1)

    def _open_batch_process(self) -> None:
        if self.data_manager is None:
            return
        dialog = BatchProcessDialog(self.data_manager, self)
        dialog.result_applied.connect(self.change_finished_signal)
        dialog.exec()

    def open_tool_for(
        self,
        data_id: int,
        channel_name: str,
        tool_type: str,
    ) -> Optional[DataProcessPipelineWidget]:
        data = self.data_manager.get(data_id) if self.data_manager is not None else None
        if data is None:
            return None
        self.init_data(data, data_id)
        workspace = self.pipeline_panel.current_workspace()
        if workspace is None:
            workspace = self.pipeline_panel._create_workspace()
        pipeline_widget = workspace.ensure_pipeline_window()
        if pipeline_widget is None:
            return None

        channel_pipeline = pipeline_widget._find_pipeline(data_id, channel_name)
        if channel_pipeline is None:
            workspace.add_channels_to_pipeline_window(pipeline_widget, [(data_id, channel_name)])
            channel_pipeline = pipeline_widget._find_pipeline(data_id, channel_name)
        if channel_pipeline is None or not channel_pipeline.layers:
            return pipeline_widget

        pipeline_widget._on_tool_dropped(
            data_id,
            channel_name,
            channel_pipeline.layers[0].layer_id,
            tool_type,
            0,
        )
        workspace.activate_pipeline_window(pipeline_widget)
        return pipeline_widget

    def current_workspace(self) -> Optional[WorkspaceMdiArea]:
        return self.pipeline_panel.current_workspace()

    def init_data(self, data, data_id: int = 0):
        self.data_id = data_id
        self.em_data = data
        self.pipeline_panel.init_data(data, data_id)
        workspace = self.pipeline_panel.current_workspace()
        if workspace is not None:
            workspace.ensure_pipeline_window()

    def set_context_resolver(
        self,
        resolver: Optional[Callable[[], Optional[tuple[int, str]]]],
    ) -> None:
        self._context_resolver = resolver
        self.pipeline_panel.set_context_resolver(resolver)

    def set_selection_payload_resolver(
        self,
        resolver: Optional[Callable[[], Optional[dict]]],
    ) -> None:
        self._selection_payload_resolver = resolver
        self.pipeline_panel.set_selection_payload_resolver(resolver)

    def set_clear_checked_channels(
        self,
        clearer: Optional[Callable[[list[tuple[int, str]]], None]],
    ) -> None:
        self._clear_checked_channels = clearer
        self.pipeline_panel.set_clear_checked_channels(clearer)

    def close_windows_for_data(self, data_id: int):
        self.pipeline_panel.close_windows_for_data(data_id)

    @property
    def pipeline_widget(self) -> Optional[DataProcessPipelineWidget]:
        workspace = self.pipeline_panel.current_workspace()
        if workspace is None:
            return None
        return workspace.active_pipeline_window() or workspace.last_pipeline_window()


if __name__ == "__main__":
    from io_utils.mat_io import MatLoader

    app = QApplication(sys.argv)
    window = DataProcessWidget()
    window.init_data(
        MatLoader.load("/home/transen5/Project/atm_rpc/039BE-20240501-20240515-dt5_struct.mat"),
        1,
    )
    window.show()
    sys.exit(app.exec())
