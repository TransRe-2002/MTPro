from __future__ import annotations

import json
import sys
from typing import Callable, Optional

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMdiArea,
    QMdiSubWindow,
    QMessageBox,
    QTabBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from base.data_manager import DataManager
from core.em_data import EMData
from ui.data_process_pipeline import DataProcessPipelineWidget, pipeline_tool_id_from_mime
from ui.batch_process_dialog import BatchProcessDialog
from ui.data_process_shared import DATA_TREE_SELECTION_MIME


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
