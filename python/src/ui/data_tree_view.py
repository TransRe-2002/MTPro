import logging
import json

from PySide6.QtWidgets import (
    QTreeView, QVBoxLayout, QWidget, QSplitter,
    QMenu, QPlainTextEdit, QLabel, QFrame, QGroupBox, QStackedWidget
)
from PySide6.QtCore import QByteArray, QMimeData, Qt, QModelIndex, Signal
from PySide6.QtGui import QDrag, QFont
from typing import Dict, Optional

from core.em_data import EMData
from ui.data_process_shared import CLASSIC_TOOL_MIME, DATA_TREE_SELECTION_MIME, create_classic_tool_model
from ui.data_tree_model import DataTreeModel
from ui.compare_widget import CompareWidget
from ui.plot_panel import PlotPanel
from base.data_manager import DataManager

class DataSelectionTreeView(QTreeView):
    def __init__(self, owner: "DataTreeViewer", parent=None):
        super().__init__(parent)
        self._owner = owner
        self.setDragEnabled(True)

    def startDrag(self, supported_actions):
        payload = self._owner.current_drag_payload()
        if payload is None:
            return

        mime_data = QMimeData()
        mime_data.setData(
            DATA_TREE_SELECTION_MIME,
            QByteArray(json.dumps(payload, ensure_ascii=True).encode("utf-8")),
        )
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec(supported_actions, Qt.DropAction.CopyAction)


class ToolTreeView(QTreeView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setHeaderHidden(True)

    def startDrag(self, supported_actions):
        index = self.currentIndex()
        if not index.isValid():
            return

        tool_type = index.data(Qt.UserRole + 1)
        if not isinstance(tool_type, str):
            return

        mime_data = QMimeData()
        mime_data.setData(CLASSIC_TOOL_MIME, QByteArray(tool_type.encode("utf-8")))
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec(supported_actions, Qt.DropAction.CopyAction)


class DataTreeViewer(QWidget):
    tool_requested = Signal(str)

    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tree_view = None
        self.status_text = None
        self.summary_stack = None
        self.info_group = None
        self.tool_group = None
        self.tool_tree = None
        self.tool_model = None
        self.data_manager = data_manager
        self._plot_panels: Dict[tuple[int, str], PlotPanel] = {}

        self.activate_id = 0
        self.init_ui()
        self.connect_signal()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ── 数据浏览器组 ──────────────────────────────────────
        browser_group = QGroupBox("数据管理器 / Data Manager", self)
        browser_group.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        browser_layout = QVBoxLayout(browser_group)
        browser_layout.setContentsMargins(4, 8, 4, 4)
        browser_layout.setSpacing(2)

        # ── 台站摘要组 ────────────────────────────────────────
        self.summary_stack = QStackedWidget(self)

        self.info_group = QGroupBox("台站摘要 / Station Summary", self)
        self.info_group.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        info_layout = QVBoxLayout(self.info_group)
        info_layout.setContentsMargins(4, 8, 4, 4)
        info_layout.setSpacing(2)

        spliter = QSplitter(Qt.Orientation.Vertical)
        spliter.setChildrenCollapsible(False)
        self.tree_view = DataSelectionTreeView(self, browser_group)
        self.tree_view.setFont(QFont("Arial", 9))
        self.model = DataTreeModel(self.data_manager, self)
        self.tree_view.setModel(self.model)
        self.tree_view.setUniformRowHeights(True)
        self.tree_view.setHeaderHidden(False)
        self.tree_view.setAlternatingRowColors(True)
        browser_layout.addWidget(self.tree_view, 1)

        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.status_text.setFont(QFont("Courier New", 9))
        self.status_text.setPlainText('请在数据管理器中选择一个台站数据集。')
        info_layout.addWidget(self.status_text, 1)

        self.tool_group = QGroupBox("数据处理工具 / Processing Tools", self)
        self.tool_group.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        tool_layout = QVBoxLayout(self.tool_group)
        tool_layout.setContentsMargins(4, 8, 4, 4)
        tool_layout.setSpacing(2)
        self.tool_tree = ToolTreeView(self.tool_group)
        self.tool_model = create_classic_tool_model()
        self.tool_tree.setModel(self.tool_model)
        self.tool_tree.expandAll()
        tool_layout.addWidget(self.tool_tree, 1)

        self.summary_stack.addWidget(self.info_group)
        self.summary_stack.addWidget(self.tool_group)

        spliter.addWidget(browser_group)
        spliter.addWidget(self.summary_stack)
        spliter.setSizes([430, 220])
        main_layout.addWidget(spliter)
        self.setLayout(main_layout)
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def connect_signal(self):
        self.tree_view.doubleClicked.connect(self.on_item_double_click)
        self.tree_view.customContextMenuRequested.connect(self.on_context_menu)
        self.tree_view.selectionModel().currentChanged.connect(self.on_current_changed)
        self.tool_tree.doubleClicked.connect(self._on_tool_activated)
        self.data_manager.active_changed.connect(self.on_active_changed)
        self.data_manager.data_removed.connect(self.on_data_removed)

    def on_current_changed(self, current: QModelIndex, previous: QModelIndex):
        del previous
        if not current.isValid():
            self.data_manager.set_active(0)
            return

        if self.model.is_dataset_index(current):
            key = self.model.dataset_key(current)
        elif self.model.is_channel_index(current):
            key = self.model.dataset_key(current)
        else:
            key = None

        if key is not None:
            self.data_manager.set_active(key)

    def on_data_removed(self, key: int):
        for panel_key in [panel_key for panel_key in self._plot_panels if panel_key[0] == key]:
            plot_panel = self._plot_panels.pop(panel_key)
            plot_panel.close()

    def on_active_changed(self, key: int):
        self.activate_id = key
        if key == 0:
            self.status_text.setPlainText('请在数据管理器中选择一个台站数据集。')
            self.tree_view.clearSelection()
        else:
            em_data = self.data_manager.get(key)
            if em_data is None:
                return

            self.status_text.setPlainText(self._format_data_summary(em_data))
            current_index = self.tree_view.currentIndex()
            if not self.model.is_channel_index(current_index):
                index = self.model.index_for_key(key)
            else:
                index = current_index
            if index.isValid() and self.tree_view.currentIndex() != index:
                self.tree_view.setCurrentIndex(index)

    def current_dataset_key(self) -> Optional[int]:
        if self.tree_view is None or self.model is None:
            return None
        current = self.tree_view.currentIndex()
        if not current.isValid():
            return None
        return self.model.dataset_key(current)

    def current_dataset(self) -> Optional[EMData]:
        key = self.current_dataset_key()
        if key is None:
            return None
        return self.data_manager.get(key)

    def checked_channel_refs(self) -> list[tuple[int, str]]:
        if self.model is None:
            return []
        return self.model.checked_channel_refs()

    def clear_checked_channels(self, refs: list[tuple[int, str]]) -> None:
        if self.model is None:
            return
        for data_id, channel_name in refs:
            self.model.set_channel_checked(data_id, channel_name, False)

    def current_channel_ref(self) -> Optional[tuple[int, str]]:
        if self.tree_view is None or self.model is None:
            return None

        current = self.tree_view.currentIndex()
        if not self.model.is_channel_index(current):
            return None

        key = self.model.dataset_key(current)
        channel_name = self.model.channel_name(current)
        if key is None or channel_name is None:
            return None
        return key, channel_name

    def current_drag_payload(self) -> Optional[dict]:
        if self.tree_view is None or self.model is None:
            return None

        current = self.tree_view.currentIndex()
        if not current.isValid():
            return None

        if self.model.is_dataset_index(current):
            data_id = self.model.dataset_key(current)
            if data_id is None:
                return None
            return {
                "kind": "dataset",
                "data_id": data_id,
            }

        if not self.model.is_channel_index(current):
            return None

        data_id = self.model.dataset_key(current)
        channel_name = self.model.channel_name(current)
        if data_id is None or channel_name is None:
            return None

        checked_refs = self.checked_channel_refs()
        if (data_id, channel_name) in checked_refs:
            refs = checked_refs
        else:
            refs = [(data_id, channel_name)]

        return {
            "kind": "channels",
            "refs": [
                {"data_id": ref_data_id, "channel_name": ref_channel_name}
                for ref_data_id, ref_channel_name in refs
            ],
        }

    def show_station_summary(self):
        if self.summary_stack is not None and self.info_group is not None:
            self.summary_stack.setCurrentWidget(self.info_group)

    def show_processing_tools(self):
        if self.summary_stack is not None and self.tool_group is not None:
            self.summary_stack.setCurrentWidget(self.tool_group)

    def _on_tool_activated(self, index: QModelIndex):
        if not index.isValid() or self.tool_model is None:
            return

        item = self.tool_model.itemFromIndex(index)
        if item is None:
            return

        tool_type = item.data(Qt.UserRole + 1)
        if isinstance(tool_type, str):
            self.tool_requested.emit(tool_type)

    @staticmethod
    def _format_data_summary(em_data) -> str:
        if getattr(em_data, 'timezone', None):
            import pytz
            station_tz = pytz.timezone(em_data.timezone)
            start_display = em_data.start_time.tz_convert(station_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
            end_display = em_data.end_time.tz_convert(station_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
        else:
            start_display = em_data.start_time.strftime('%Y-%m-%d %H:%M:%S %Z')
            end_display = em_data.end_time.strftime('%Y-%m-%d %H:%M:%S %Z')

        return (
            f"数据名称：{em_data.name}\n"
            f"开始时间：{start_display}\n"
            f"结束时间：{end_display}\n"
            f"采样周期：{em_data.dt.total_seconds()}s\n"
            f"采集点数：{em_data.npts}\n"
            f"采集站纬度：{em_data.latitude}\n"
            f"采集站经度：{em_data.longitude}\n"
            f"电场单位：{em_data.e_units}\n"
            f"磁场单位：{em_data.m_units}\n"
            f"数据路径：{em_data.path}"
        )

    def on_item_double_click(self, index: QModelIndex):
        if not self.model.is_channel_index(index):
            return

        key = self.model.dataset_key(index)
        ch_name = self.model.channel_name(index)
        if key is None or ch_name is None:
            return

        plot_key = (key, ch_name)
        plot_panel = self._plot_panels.get(plot_key)
        if plot_panel is not None:
            plot_panel.show()
            plot_panel.activateWindow()
            plot_panel.raise_()
            self.logger.info("Reused plot panel for dataset #%s channel %s", key, ch_name)
            return

        em_data = self.data_manager.get(key)
        if em_data is None:
            return
        data_name = em_data.name
        title = f"{data_name} - {ch_name}"

        x_values = em_data.datetime_index
        y_values = em_data.get_channel_series(ch_name)
        plot_panel = PlotPanel(
            x_data=x_values,
            y_data=y_values,
            label=f"{data_name} - {ch_name}",
            parent=self,
        )
        plot_panel.setWindowTitle(title)

        if ch_name.startswith('E'):
            e_units = em_data.e_units
            if e_units is not None:
                plot_panel.set_label('Time', e_units)
            else:
                plot_panel.set_label('Time', None)

        if ch_name.startswith('H') or ch_name.startswith('B'):
            m_units = em_data.m_units
            if m_units is not None:
                plot_panel.set_label('Time', m_units)
            else:
                plot_panel.set_label('Time', None)

        plot_panel.destroyed.connect(
            lambda *_, panel_key=plot_key: self._plot_panels.pop(panel_key, None)
        )
        plot_panel.show()
        self._plot_panels[plot_key] = plot_panel
        self.logger.info("Opened plot panel for dataset #%s channel %s", key, ch_name)

    def on_context_menu(self, pos):
        index = self.tree_view.indexAt(pos)

        if not index.isValid():
            return

        if self.model.is_dataset_index(index):
            menu = QMenu(self)
            action_save = menu.addAction("保存")
            action_save_as = menu.addAction("另存为")
            action_del = menu.addAction("关闭数据")
            action = menu.exec(self.tree_view.viewport().mapToGlobal(pos))
            data_id = self.model.dataset_key(index)
            if data_id is None:
                return

            main_window = self.window()
            if action == action_save and hasattr(main_window, "save_file_for"):
                main_window.save_file_for(data_id)
            elif action == action_save_as and hasattr(main_window, "save_file_as_for"):
                main_window.save_file_as_for(data_id)
            elif action == action_del:
                self.data_manager.remove(data_id)

        elif self.model.is_channel_index(index):
            menu = QMenu(self)
            action_show = menu.addAction("显示曲线")
            action_compare = menu.addAction("序列对比")
            action = menu.exec(self.tree_view.viewport().mapToGlobal(pos))
            if action == action_show:
                self.on_item_double_click(index)
            elif action == action_compare:
                key = self.model.dataset_key(index)
                ch_name = self.model.channel_name(index)
                if key is None or ch_name is None:
                    return
                channels = self.get_check_state()
                if len(channels) == 0:
                    em_data = self.data_manager.get(key)
                    if em_data is None:
                        return
                    channels = [(em_data, ch_name)]
                self.open_compare_widget(channels)

    def get_check_state(self) -> list[tuple[EMData, str]]:
        refs = self.model.checked_channel_refs()
        result: list[tuple[EMData, str]] = []
        for data_id, channel_name in refs:
            em_data = self.data_manager.get(data_id)
            if em_data is not None and em_data.has_channel(channel_name):
                result.append((em_data, channel_name))
        return result

    def open_compare_widget(self, channels: list[tuple[EMData, str]]):
        if len(channels) == 0:
            return
        compare_widget = CompareWidget(channels, parent=self)
        compare_widget.setWindowFlags(Qt.WindowType.Window)
        compare_widget.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        compare_widget.setWindowTitle(f"Series Compare Widget")
        compare_widget.show()
        compare_widget.raise_()
        self.logger.info("Opened compare widget for %s channels", len(channels))


if __name__ == '__main__':
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    data_manager = DataManager()
    window = DataTreeViewer(data_manager)
    window.show()
    sys.exit(app.exec())
