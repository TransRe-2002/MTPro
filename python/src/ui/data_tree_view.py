from PySide6.QtWidgets import (
    QTreeView, QVBoxLayout, QWidget, QSplitter,
    QMenu, QPlainTextEdit
)
from PySide6.QtCore import Qt, QModelIndex
from typing import Dict, List

from core.em_data import Channel
from ui.data_tree_model import DataTreeModel
from ui.compare_widget import CompareWidget
from ui.plot_panel import PlotPanel
from base.data_manager import DataManager


class DataTreeViewer(QWidget):
    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self.model = None
        self.tree_view = None
        self.status_text = None
        self.data_manager = data_manager
        self._plot_panels: Dict[tuple[int, str], PlotPanel] = {}

        self.activate_id = 0
        self.init_ui()
        self.connect_signal()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        spliter = QSplitter(Qt.Orientation.Vertical)
        self.tree_view = QTreeView()
        self.model = DataTreeModel(self.data_manager, self)
        self.tree_view.setModel(self.model)
        self.tree_view.setUniformRowHeights(True)
        spliter.addWidget(self.tree_view)

        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setPlainText('此处显示当前激活数据的详细信息。')
        spliter.addWidget(self.status_text)
        main_layout.addWidget(spliter)
        self.setLayout(main_layout)
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def connect_signal(self):
        self.tree_view.doubleClicked.connect(self.on_item_double_click)
        self.tree_view.customContextMenuRequested.connect(self.on_context_menu)
        self.tree_view.selectionModel().currentChanged.connect(self.on_current_changed)
        self.data_manager.active_changed.connect(self.on_active_changed)
        self.data_manager.data_removed.connect(self.on_data_removed)

    def on_current_changed(self, current: QModelIndex, previous: QModelIndex):
        del previous
        if not current.isValid():
            self.data_manager.set_active(0)
            return

        if self.model.is_dataset_index(current):
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
            self.status_text.setPlainText('此处显示当前激活数据的详细信息。')
            self.tree_view.clearSelection()
        else:
            em_data = self.data_manager.get(key)
            if em_data is None:
                return

            self.status_text.setPlainText(self._format_data_summary(em_data))
            index = self.model.index_for_key(key)
            if index.isValid() and self.tree_view.currentIndex() != index:
                self.tree_view.setCurrentIndex(index)

    @staticmethod
    def _format_data_summary(em_data) -> str:
        return (
            f"数据名称：{em_data.name}\n"
            f"开始时间：{em_data.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"结束时间：{em_data.end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"持续时间：{em_data.dt.total_seconds()}s\n"
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
            return

        em_data = self.data_manager.get(key)
        if em_data is None:
            return
        data_name = em_data.name
        title = f"{data_name} - {ch_name}"

        x_values = em_data.datetime_index
        y_values = em_data.data[ch_name].cts
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
                    channels = [em_data.data[ch_name]]
                self.open_compare_widget(channels)

    def get_check_state(self) -> List[Channel]:
        return self.model.checked_channels()

    def open_compare_widget(self, channels: List[Channel]):
        if len(channels) == 0:
            return
        compare_widget = CompareWidget(channels, parent=self)
        compare_widget.setWindowFlags(Qt.WindowType.Window)
        compare_widget.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        compare_widget.setWindowTitle(f"Series Compare Widget")
        compare_widget.show()
        compare_widget.raise_()


if __name__ == '__main__':
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    data_manager = DataManager()
    window = DataTreeViewer(data_manager)
    window.show()
    sys.exit(app.exec())
