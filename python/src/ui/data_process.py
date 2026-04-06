from __future__ import annotations

import sys
from typing import Optional, Dict

import pandas as pd

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QSplitter, QApplication, QMdiSubWindow,
    QMdiArea, QHBoxLayout, QGroupBox, QTextEdit,
    QRadioButton, QGridLayout, QSizePolicy, QToolBar,
    QDockWidget, QTreeView, QVBoxLayout, QLabel,
    QPushButton, QComboBox
)
from PySide6.QtGui import QStandardItemModel, QStandardItem, QAction, QDrag
from PySide6.QtCore import Qt, QMimeData, QByteArray

from core.em_data import EMData, Channel
from processor.remove_spike import RemoveSpike
from processor.remove_step import RemoveStep

class DraggableTreeView(QTreeView):
    """支持拖拽的自定义树视图"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)  # 启用视图的拖拽功能
        self.setEditTriggers(QTreeView.NoEditTriggers)  # 禁用编辑触发
        self.setSelectionMode(QTreeView.SingleSelection)  # 单选模式更符合拖拽直觉

    def startDrag(self, supportedActions):
        index = self.currentIndex()
        if not index.isValid():
            return

        # 获取被拖拽项的数据
        item = self.model().itemFromIndex(index)
        # 这里我们假设项的数据角色（Qt.UserRole + 1）存储了窗口类型标识符
        widget_type = item.data(Qt.UserRole + 1)

        if widget_type:
            # 创建MIME数据，用于在拖拽过程中传递信息
            mime_data = QMimeData()
            # 使用自定义MIME类型来传递窗口类型
            mime_data.setData("application/x-widgettype", QByteArray(widget_type.encode()))

            # 修复：使用 QDrag 而不是 dragObject()
            drag = QDrag(self)
            drag.setMimeData(mime_data)

            # 开始执行拖拽操作
            drag.exec(supportedActions, Qt.CopyAction)

class CustomMdiArea(QMdiArea):
    """支持接收拖拽放置的自定义MDI区域"""

    def __init__(self, parent: DataProcessWidget=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.setAcceptDrops(True)  # 启用放置接受

    def dragEnterEvent(self, event):
        """当拖拽进入区域时，判断数据是否可接受"""
        if event.mimeData().hasFormat("application/x-widgettype"):
            event.acceptProposedAction()  # 接受此操作
        else:
            event.ignore()  # 忽略不相关的拖拽

    def dropEvent(self, event):
        """当放置发生时，创建对应的子窗口"""
        if event.mimeData().hasFormat("application/x-widgettype"):
            # 获取传递过来的窗口类型标识符
            widget_type = event.mimeData().data("application/x-widgettype").data().decode()
            # 获取鼠标释放的位置，作为新窗口的初始位置
            position = event.position().toPoint()

            # 根据类型创建不同的子窗口
            sub_window:Optional[QWidget] = None
            for key in self.parent_widget.data_group.keys():
                if self.parent_widget.data_group[key].isChecked():
                    ch = self.parent_widget.em_data.data[key]
                    sub_window = self.create_subwindow(widget_type, position, ch)
                    break

            if sub_window:
                self.addSubWindow(sub_window)
                sub_window.show()
                # 将活动窗口设置为新创建的窗口
                self.setActiveSubWindow(sub_window)
                event.acceptProposedAction()

    def create_subwindow(self, widget_type, position, channel: Channel):
        """根据类型标识符创建子窗口和内容控件"""
        sub_window = QMdiSubWindow()
        sub_window.setWindowTitle(f"{widget_type}")
        # 设置子窗口的初始位置和大小
        sub_window.setGeometry(position.x(), position.y(), 800, 600)

        content_widget = None
        if widget_type == "remove spike":
            content_widget = RemoveSpike(channel)
            content_widget.result_signal.connect(self.parent_widget.on_change_data_finished)
        elif widget_type == "remove step":
            content_widget = RemoveStep(channel)
            content_widget.result_signal.connect(self.parent_widget.on_change_data_finished)
        else:
            content_widget = QTextEdit()
            content_widget.setPlainText(f"未知工具类型: {widget_type}")

        if content_widget:
            sub_window.setWidget(content_widget)
            return sub_window
        return None

class DataProcessWidget(QWidget):
    change_finished_signal = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tree_view = None
        self.em_data: Optional[EMData] = None
        self.init_ui()
        self.init_tree_view()

    def init_data(self, data):
        self.em_data = data
        self.enable_fields()

    def init_ui(self):
        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Horizontal)
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)

        radio_button_group = QGroupBox("选择处理的曲线")
        self.radio_button_group_layout = QVBoxLayout(radio_button_group)
        self.data_group: Dict[str, QRadioButton] = {}

        for rb in self.data_group.values():
            rb.setDisabled(True)

        radio_button_group.setSizePolicy(
            QSizePolicy.Policy.Minimum,
            QSizePolicy.Policy.Minimum
        )

        self.tree_view = DraggableTreeView(self)
        self.tree_view.setSizePolicy(
            QSizePolicy.Policy.Minimum,
            QSizePolicy.Policy.Expanding
        )
        left_layout.addWidget(radio_button_group)
        left_layout.addWidget(self.tree_view)

        right_widget = CustomMdiArea(self)
        right_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def init_tree_view(self):
        self.tree_model = QStandardItemModel()
        self.tree_model.setHorizontalHeaderLabels(["数据处理工具"])

        cat_manual_process = QStandardItem("手动数据处理")
        cat_auto_process = QStandardItem("自动数据处理")
        cat_output = QStandardItem("阻抗输出")

        remove_spike_item = QStandardItem("手动框选去噪")
        remove_spike_item.setData("remove spike", Qt.UserRole + 1)
        cat_manual_process.appendRow(remove_spike_item)

        remove_step_item = QStandardItem("手动去除阶跃")
        remove_step_item.setData("remove step", Qt.UserRole + 1)
        cat_manual_process.appendRow(remove_step_item)

        root = self.tree_model.invisibleRootItem()
        root.appendRow(cat_manual_process)
        root.appendRow(cat_auto_process)
        root.appendRow(cat_output)

        # 将模型设置到视图
        self.tree_view.setModel(self.tree_model)
        # 默认展开所有第一级分类
        self.tree_view.expandAll()

    def enable_fields(self):
        if self.em_data is None:
            return
        self.data_group.clear()
        layout = self.radio_button_group_layout

        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            widget.deleteLater()

        for key in self.em_data.data.keys():
            self.data_group[key] = QRadioButton(f"{key}")
            self.radio_button_group_layout.addWidget(self.data_group[key])
            continue
        self.data_group[list(self.data_group.keys())[0]].setChecked(True)

    def on_change_data_finished(self, channel, result):
        self.em_data.data[channel].cts = pd.Series(result)
        self.change_finished_signal.emit(channel)

    def on_radio_button_clicked(self):
        for key, radio_button in self.data_group.items():
            if radio_button.isChecked():
                self.current_data = self.em_data[key]
                break

if __name__ == '__main__':
    from io_utils.mat_io import MatLoader
    app = QApplication(sys.argv)
    window = DataProcessWidget()
    window.init_data(MatLoader.load("/home/transen5/Project/atm_rpc/039BE-20240501-20240515-dt5_struct.mat"))
    window.show()
    sys.exit(app.exec())
