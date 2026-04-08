from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Dict, Optional

import pandas as pd

from PySide6.QtCore import QByteArray, QMimeData, Qt, Signal
from PySide6.QtGui import QDrag, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QMdiArea,
    QMdiSubWindow,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from base.data_manager import DataManager
from core.em_data import Channel, EMData
from processor.remove_spike import RemoveSpike
from processor.remove_step_by_diff import RemoveStepByDiff
from processor.remove_step_by_window import RemoveStepByWindow
from ui.data_process_pipeline import DataProcessPipelineWidget


@dataclass(frozen=True)
class ToolWindowState:
    tool_type: str
    channel_name: str


class DraggableTreeView(QTreeView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setEditTriggers(QTreeView.NoEditTriggers)
        self.setSelectionMode(QTreeView.SingleSelection)

    def startDrag(self, supportedActions):
        index = self.currentIndex()
        if not index.isValid():
            return

        item = self.model().itemFromIndex(index)
        widget_type = item.data(Qt.UserRole + 1)
        if widget_type:
            mime_data = QMimeData()
            mime_data.setData(
                "application/x-widgettype",
                QByteArray(widget_type.encode()),
            )
            drag = QDrag(self)
            drag.setMimeData(mime_data)
            drag.exec(supportedActions, Qt.CopyAction)


class CustomMdiArea(QMdiArea):
    def __init__(self, parent: DataProcessLegacyWidget = None):
        super().__init__(parent)
        self.parent_widget = parent
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-widgettype"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasFormat("application/x-widgettype"):
            return

        widget_type = event.mimeData().data(
            "application/x-widgettype"
        ).data().decode()
        position = event.position().toPoint()
        channel = self.parent_widget.current_channel()
        if channel is None:
            return

        sub_window = self.parent_widget.open_tool_window(
            widget_type,
            channel.name,
            position=position,
        )
        if sub_window:
            sub_window.show()
            self.setActiveSubWindow(sub_window)
            event.acceptProposedAction()


class DataProcessLegacyWidget(QWidget):
    change_finished_signal = Signal(int, str)

    def __init__(self, data_manager: Optional[DataManager] = None, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.tree_view = None
        self.mdi_area = None
        self.data_id: int = 0
        self.em_data: Optional[EMData] = None
        self._selected_channel_name: Optional[str] = None
        self._init_ui()
        self._init_tree_view()

    def init_data(self, data, data_id: int = 0):
        self.data_id = data_id
        self.em_data = data
        if self.em_data is None:
            self._selected_channel_name = None
        self._enable_fields()
        self._close_invalid_tool_windows()

    def _init_ui(self):
        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Horizontal)
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)

        radio_button_group = QGroupBox("选择处理的曲线")
        self.radio_button_group_layout = QVBoxLayout(radio_button_group)
        self.data_group: Dict[str, QRadioButton] = {}

        self.tree_view = DraggableTreeView(self)
        self.tree_view.setSizePolicy(
            QSizePolicy.Policy.Minimum,
            QSizePolicy.Policy.Expanding,
        )
        left_layout.addWidget(radio_button_group)
        left_layout.addWidget(self.tree_view)

        self.mdi_area = CustomMdiArea(self)
        self.mdi_area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        splitter.addWidget(left_widget)
        splitter.addWidget(self.mdi_area)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def _init_tree_view(self):
        self.tree_model = QStandardItemModel()
        self.tree_model.setHorizontalHeaderLabels(["数据处理工具"])

        cat_manual_process = QStandardItem("手动数据处理")
        cat_auto_process = QStandardItem("自动数据处理")
        cat_output = QStandardItem("阻抗输出")

        remove_spike_item = QStandardItem("手动框选去噪")
        remove_spike_item.setData("remove spike", Qt.UserRole + 1)
        cat_manual_process.appendRow(remove_spike_item)

        remove_step_diff_item = QStandardItem("差分去阶跃")
        remove_step_diff_item.setData("remove step diff", Qt.UserRole + 1)
        cat_manual_process.appendRow(remove_step_diff_item)

        remove_step_window_item = QStandardItem("窗口均值去阶跃")
        remove_step_window_item.setData("remove step window", Qt.UserRole + 1)
        cat_manual_process.appendRow(remove_step_window_item)

        root = self.tree_model.invisibleRootItem()
        root.appendRow(cat_manual_process)
        root.appendRow(cat_auto_process)
        root.appendRow(cat_output)

        self.tree_view.setModel(self.tree_model)
        self.tree_view.expandAll()

    def _enable_fields(self):
        layout = self.radio_button_group_layout
        self.data_group.clear()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if self.em_data is None:
            return

        for key in self.em_data.data.keys():
            button = QRadioButton(f"{key}")
            button.toggled.connect(
                lambda checked, channel_name=key:
                self._on_channel_toggled(channel_name, checked)
            )
            self.data_group[key] = button
            self.radio_button_group_layout.addWidget(button)

        button = None
        if self._selected_channel_name is not None:
            button = self.data_group.get(self._selected_channel_name)
        if button is None:
            button = next(iter(self.data_group.values()), None)
        if button is not None:
            button.setChecked(True)

    def _on_channel_toggled(self, channel_name: str, checked: bool):
        if checked:
            self._selected_channel_name = channel_name

    def current_channel(self) -> Optional[Channel]:
        if self.em_data is None:
            return None

        if self._selected_channel_name is not None:
            channel = self.em_data.data.get(self._selected_channel_name)
            button = self.data_group.get(self._selected_channel_name)
            if channel is not None and button is not None and button.isChecked():
                return channel

        for key, radio_button in self.data_group.items():
            if radio_button.isChecked():
                self._selected_channel_name = key
                return self.em_data.data.get(key)
        return None

    def open_tool_window(
        self,
        tool_type: str,
        channel_name: str,
        position=None,
    ) -> Optional[QMdiSubWindow]:
        if self.em_data is None or self.mdi_area is None:
            return None

        channel = self.em_data.data.get(channel_name)
        if channel is None:
            return None

        tool_key = self._canonical_tool_type(tool_type)
        existing = self._find_tool_window(tool_key, channel_name)
        if existing is not None:
            content_widget = existing.widget()
            if content_widget is not None and content_widget.isHidden():
                content_widget.show()
            existing.show()
            self.mdi_area.setActiveSubWindow(existing)
            return existing

        sub_window = QMdiSubWindow()
        sub_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        sub_window.setProperty("tool_type", tool_type)
        sub_window.setProperty("tool_key", tool_key)
        sub_window.setProperty("channel_name", channel_name)
        sub_window.setWindowTitle(f"{tool_type} - {channel_name}")
        if position is not None:
            sub_window.setGeometry(position.x(), position.y(), 800, 600)
        else:
            sub_window.resize(800, 600)

        if tool_key == "remove spike":
            content_widget = RemoveSpike(channel)
            content_widget.result_signal.connect(
                lambda ch, result, data_id=self.data_id:
                self.on_change_data_finished_for(data_id, ch, result)
            )
        elif tool_key == "remove step diff":
            content_widget = RemoveStepByDiff(channel)
            content_widget.result_signal.connect(
                lambda ch, result, data_id=self.data_id:
                self.on_change_data_finished_for(data_id, ch, result)
            )
        elif tool_key == "remove step window":
            content_widget = RemoveStepByWindow(channel)
            content_widget.result_signal.connect(
                lambda ch, result, data_id=self.data_id:
                self.on_change_data_finished_for(data_id, ch, result)
            )
        else:
            content_widget = QTextEdit()
            content_widget.setPlainText(f"未知工具类型: {tool_type}")

        sub_window.setWidget(content_widget)
        self.mdi_area.addSubWindow(sub_window)
        return sub_window

    def open_tool_window_states(self) -> list[ToolWindowState]:
        if self.mdi_area is None:
            return []

        states: list[ToolWindowState] = []
        for sub_window in self.mdi_area.subWindowList():
            tool_type = sub_window.property("tool_type")
            channel_name = sub_window.property("channel_name")
            if isinstance(tool_type, str) and isinstance(channel_name, str):
                states.append(ToolWindowState(tool_type, channel_name))
        return states

    def on_change_data_finished_for(self, data_id: int, channel: str, result):
        if data_id == 0:
            return

        data = self.data_manager.get(data_id) if self.data_manager is not None else None
        if data is None:
            return

        data.data[channel].cts = pd.Series(result)
        self.change_finished_signal.emit(data_id, channel)

    def _find_tool_window(
        self,
        tool_type: str,
        channel_name: str,
    ) -> Optional[QMdiSubWindow]:
        if self.mdi_area is None:
            return None

        for sub_window in self.mdi_area.subWindowList():
            if (
                sub_window.property("tool_key") == tool_type
                and sub_window.property("channel_name") == channel_name
            ):
                return sub_window
        return None

    def _canonical_tool_type(self, tool_type: str) -> str:
        if tool_type == "remove step":
            return "remove step diff"
        return tool_type

    def _close_invalid_tool_windows(self):
        if self.mdi_area is None:
            return

        if self.em_data is None:
            for sub_window in list(self.mdi_area.subWindowList()):
                self._dispose_sub_window(sub_window)
            return

        valid_channels = set(self.em_data.data.keys())
        for sub_window in list(self.mdi_area.subWindowList()):
            channel_name = sub_window.property("channel_name")
            if isinstance(channel_name, str) and channel_name not in valid_channels:
                self._dispose_sub_window(sub_window)

    def _dispose_sub_window(self, sub_window: QMdiSubWindow):
        if self.mdi_area is not None:
            self.mdi_area.removeSubWindow(sub_window)
        sub_window.close()
        sub_window.deleteLater()

    def closeEvent(self, event):
        if self.mdi_area is not None:
            for sub_window in list(self.mdi_area.subWindowList()):
                self._dispose_sub_window(sub_window)
        super().closeEvent(event)


class DataProcessWidget(QWidget):
    change_finished_signal = Signal(int, str)

    def __init__(self, data_manager: Optional[DataManager] = None, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.data_id = 0
        self.em_data: Optional[EMData] = None
        self._mode = "classic"

        self.mode_buttons = QButtonGroup(self)
        self.mode_buttons.setExclusive(True)
        self.classic_button = QToolButton(self)
        self.classic_button.setText("Classic")
        self.classic_button.setCheckable(True)
        self.classic_button.setChecked(True)
        self.pipeline_button = QToolButton(self)
        self.pipeline_button.setText("Pipeline")
        self.pipeline_button.setCheckable(True)

        self.mode_buttons.addButton(self.classic_button, 0)
        self.mode_buttons.addButton(self.pipeline_button, 1)

        self.stack = QStackedWidget(self)
        self.legacy_widget = DataProcessLegacyWidget(data_manager, self.stack)
        self.pipeline_widget = DataProcessPipelineWidget(data_manager, self.stack)
        self.stack.addWidget(self.legacy_widget)
        self.stack.addWidget(self.pipeline_widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        switch_row = QWidget(self)
        switch_layout = QHBoxLayout(switch_row)
        switch_layout.setContentsMargins(8, 8, 8, 4)
        switch_layout.addWidget(self.classic_button)
        switch_layout.addWidget(self.pipeline_button)
        switch_layout.addStretch(1)

        layout.addWidget(switch_row)
        layout.addWidget(self.stack, 1)

        self.mode_buttons.idClicked.connect(self._on_mode_changed)
        self.legacy_widget.change_finished_signal.connect(self.change_finished_signal)
        self.pipeline_widget.change_finished_signal.connect(self.change_finished_signal)

    def init_data(self, data, data_id: int = 0):
        self.data_id = data_id
        self.em_data = data
        self.legacy_widget.init_data(data, data_id)
        self.pipeline_widget.init_data(data, data_id)

    def current_mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str):
        if mode == self._mode:
            return
        if mode == "pipeline":
            self.pipeline_button.setChecked(True)
            self._on_mode_changed(1)
        else:
            self.classic_button.setChecked(True)
            self._on_mode_changed(0)

    def _on_mode_changed(self, button_id: int):
        if button_id == 1:
            self._mode = "pipeline"
            self.stack.setCurrentWidget(self.pipeline_widget)
        else:
            self._mode = "classic"
            self.stack.setCurrentWidget(self.legacy_widget)


if __name__ == '__main__':
    from io_utils.mat_io import MatLoader

    app = QApplication(sys.argv)
    window = DataProcessWidget()
    window.init_data(
        MatLoader.load("/home/transen5/Project/atm_rpc/039BE-20240501-20240515-dt5_struct.mat"),
        1,
    )
    window.show()
    sys.exit(app.exec())
