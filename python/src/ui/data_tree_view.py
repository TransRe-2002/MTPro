from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import (
    QTreeView, QVBoxLayout, QWidget,
    QMenu, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QModelIndex
from typing import Dict, Optional, List

from core.em_data import EMData, Channel
from ui.compare_widget import CompareWidget
from ui.plot_panel import PlotPanel
from base.data_manager import DataManager


class DataTreeViewer(QWidget):
    activate_changed = Signal(int)

    @staticmethod
    def _make_readonly_item(text: str) -> QStandardItem:
        """创建不可编辑的 QStandardItem"""
        item = QStandardItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self.model = None
        self.tree_view = None
        self.activate_info = None
        self.btn_compare = None
        self.em_datas:DataManager = data_manager

        self.activate_id = None
        self.init_ui()
        self.connect_signal()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        activate_layout = QHBoxLayout()
        label = QLabel('当前激活数据：')
        self.activate_info = QLineEdit()
        self.activate_info.setReadOnly(True)
        activate_layout.addWidget(label)
        activate_layout.addWidget(self.activate_info)
        main_layout.addLayout(activate_layout)

        self.tree_view = QTreeView()
        self.model = QStandardItemModel()
        self.tree_view.setModel(self.model)
        self.model.setHorizontalHeaderLabels(
            ['数据名称', '开始时间', '结束时间', '采集间隔', '采集点数',
             '坐标纬度', '坐标经度', '电场单位', '磁场单位', '文件路径']
        )
        main_layout.addWidget(self.tree_view)
        self.setLayout(main_layout)

        self.btn_compare = QPushButton('序列对比')
        main_layout.addWidget(self.btn_compare)

        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def connect_signal(self):
        self.tree_view.doubleClicked.connect(self.on_item_double_click)
        self.tree_view.customContextMenuRequested.connect(self.on_context_menu)
        self.btn_compare.clicked.connect(self.on_button_compare_clicked)

    def on_data_added(self, key: int):
        em_data = self.em_datas.get(key)
        data_name = em_data.name

        parent_item = self._make_readonly_item(data_name)
        parent_item.setData(key, Qt.ItemDataRole.UserRole)

        for ch_name in em_data.data:
            child_item = self._make_readonly_item(ch_name)
            child_item.setCheckable(True)
            child_item.setCheckState(Qt.CheckState.Unchecked)
            child_item.setFlags(child_item.flags()
                & ~Qt.ItemFlag.ItemIsUserTristate
                & ~Qt.ItemFlag.ItemIsAutoTristate
            )
            child_item.setData(key, Qt.ItemDataRole.UserRole)
            child_item.setData(None, Qt.ItemDataRole.UserRole + 1)
            # 之后会加上为其加上槽函数
            parent_item.appendRow([
                child_item,
                self._make_readonly_item(em_data.start_time.strftime('%Y-%m-%d %H:%M:%S')),
                self._make_readonly_item(em_data.end_time.strftime('%Y-%m-%d %H:%M:%S')),
                self._make_readonly_item(str(em_data.dt.total_seconds()) + 's'),
                self._make_readonly_item(str(em_data.npts)),
            ])

        self.model.appendRow([
            parent_item,
            self._make_readonly_item(em_data.start_time.strftime('%Y-%m-%d %H:%M:%S')),
            self._make_readonly_item(em_data.end_time.strftime('%Y-%m-%d %H:%M:%S')),
            self._make_readonly_item(str(em_data.dt.total_seconds()) + 's'),
            self._make_readonly_item(str(em_data.npts)),
            self._make_readonly_item(str(em_data.latitude)),
            self._make_readonly_item(str(em_data.longitude)),
            self._make_readonly_item(em_data.e_units),
            self._make_readonly_item(em_data.m_units),
            self._make_readonly_item(em_data.path)
        ])

    def on_item_double_click(self, index: QModelIndex):
        first_col_index = index.sibling(index.row(), 0)
        item = self.model.itemFromIndex(first_col_index)
        if item is None or not item.isCheckable():
            return

        key = item.data(Qt.ItemDataRole.UserRole)
        ch_name = item.text()
        plot_panel = item.data(Qt.ItemDataRole.UserRole + 1)
        if plot_panel is not None:
            plot_panel.show()
            plot_panel.activateWindow()
            plot_panel.raise_()
            return
        else:
            em_data = self.em_datas.get(key)
            data_name = em_data.name
            title = f"{data_name} - {ch_name}"

            x_values = em_data.datetime_index
            y_values = em_data.data[ch_name].cts
            plot_panel = PlotPanel(x_data=x_values, y_data=y_values, label=f"{data_name} - {ch_name}", parent=self)
            plot_panel.setWindowTitle(title)

            if ch_name.startswith('E'):
                e_units = em_data.e_units
                if e_units is not None:
                    plot_panel.set_label(f'Time', f'{e_units}')
                else:
                    plot_panel.set_label(f'Time', None)

            if ch_name.startswith('H') or ch_name.startswith('B'):
                m_units = em_data.m_units
                if m_units is not None:
                    plot_panel.set_label(f'Time', f'{m_units}')
                else:
                    plot_panel.set_label(f'Time', None)

            plot_panel.show()
            item.setData(plot_panel, Qt.ItemDataRole.UserRole + 1)

    def on_context_menu(self, pos):
        index = self.tree_view.indexAt(pos)

        if not index.isValid():
            return

        is_parent = self.tree_view.model().hasIndex(0, 0, index)

        if is_parent:
            menu = QMenu(self)
            action_activate = menu.addAction("切换激活")
            action_del = menu.addAction("关闭数据")
            action = menu.exec(self.tree_view.viewport().mapToGlobal(pos))

            if action == action_activate:
                self.activate_id = index.data(Qt.ItemDataRole.UserRole)
                name = self.em_datas.get(self.activate_id).name
                activate_str = f"{name}({self.activate_id})"
                self.activate_info.setText(activate_str)
                self.activate_changed.emit(self.activate_id)
            elif action == action_del:
                data_id = index.data(Qt.ItemDataRole.UserRole)
                if data_id == self.activate_id:
                    self.activate_id = 0
                    self.activate_changed.emit(self.activate_id)
                self.model.removeRow(index.row())
                self.em_datas.updated_removed.emit(data_id)

        else:
            menu = QMenu(self)
            action_show = menu.addAction("显示曲线")
            action = menu.exec(self.tree_view.viewport().mapToGlobal(pos))
            if action == action_show:
                self.on_item_double_click(index)

    def get_check_state(self) -> List[Channel]:
        result = []
        for row in range(self.model.rowCount()):
            parent_item = self.model.item(row, 0)
            key = parent_item.data(Qt.ItemDataRole.UserRole)
            for child_row in range(parent_item.rowCount()):
                child_item = parent_item.child(child_row, 0)
                if child_item.checkState() == Qt.CheckState.Checked:
                    chid = child_item.text()
                    result.append(self.em_datas.get(key).data[chid])
        return result

    def on_button_compare_clicked(self):
        list_ch = self.get_check_state()
        if len(list_ch) == 0:
            return
        compare_widget = CompareWidget(list_ch, parent=self)
        compare_widget.setWindowFlags(Qt.WindowType.Window)
        compare_widget.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        compare_widget.setWindowTitle(f"Series Compare Widget")
        compare_widget.show()
        compare_widget.raise_()
