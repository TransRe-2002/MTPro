from PySide6.QtCore import Signal, Qt, QObject
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QFileDialog,
    QTabWidget,
    QSplitter,
    QStackedWidget,
    QWidget,
    QVBoxLayout,
    QLabel,
)
from typing import Dict, Optional

from base.data_manager import DataManager
from core.mat_data import MatEMData
from io_utils.mat_io import MatLoader, MatSaver
from ui.data_process import DataProcessWidget
from ui.data_tree_view import DataTreeViewer
from ui.data_view_widget import DataViewWidget
from ui.welcome import Welcome

FILTER_TYPE = [
    "Matlab数据文件 (*.mat)",
]


class MainWindow(QMainWindow):
    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MTPro")
        self.data_manager = data_manager
        self.data_tree_viewer = DataTreeViewer(self.data_manager, self)
        self.activate_id: int = 0
        self.data_viewers: Dict[int, DataViewWidget] = {}
        self.data_process_widgets: Dict[int, DataProcessWidget] = {}
        self.tab_widget = QTabWidget()
        self.vertical_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.vertical_splitter.addWidget(self.data_tree_viewer)
        self.vertical_splitter.addWidget(self.tab_widget)
        self.data_view_stack = QStackedWidget(self)
        self.data_view_placeholder = self._create_data_view_placeholder()
        self.data_view_stack.addWidget(self.data_view_placeholder)
        self.tab_widget.addTab(self.data_view_stack, "查看数据")
        self.data_process_stack = QStackedWidget(self)
        self.data_process_placeholder = self._create_data_process_placeholder()
        self.data_process_stack.addWidget(self.data_process_placeholder)
        self.tab_widget.addTab(self.data_process_stack, "数据处理")

        self.create_menu_bar()
        self.welcome = Welcome(self)

        self.welcome.open_button.clicked.connect(self.open_file)
        self.data_manager.active_changed.connect(self.on_activate_changed)
        self.data_manager.data_removed.connect(self.on_data_removed)

        self.setCentralWidget(self.welcome)

    def _create_data_view_placeholder(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.addStretch(1)
        label = QLabel("请选择左侧数据以查看波形。", widget)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addStretch(1)
        return widget

    def _create_data_process_placeholder(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.addStretch(1)
        label = QLabel("请选择左侧数据以使用处理工具。", widget)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addStretch(1)
        return widget

    def _ensure_data_viewer(self, key: int) -> Optional[DataViewWidget]:
        viewer = self.data_viewers.get(key)
        if viewer is not None:
            return viewer

        em_data = self.data_manager.get(key)
        if em_data is None:
            return None

        viewer = DataViewWidget(self.data_view_stack)
        viewer.init_data(em_data)
        self.data_viewers[key] = viewer
        self.data_view_stack.addWidget(viewer)
        return viewer

    def _ensure_data_process_widget(self, key: int) -> Optional[DataProcessWidget]:
        widget = self.data_process_widgets.get(key)
        if widget is not None:
            return widget

        em_data = self.data_manager.get(key)
        if em_data is None:
            return None

        widget = DataProcessWidget(self.data_manager, self.data_process_stack)
        widget.init_data(em_data, key)
        widget.change_finished_signal.connect(self.on_data_change_finished)
        self.data_process_widgets[key] = widget
        self.data_process_stack.addWidget(widget)
        return widget

    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件(&F)")
        open_action = QAction("打开", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        save_action = QAction("保存", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_file)
        save_as_action = QAction("另存为", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_file_as)
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Ctrl+Q")
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        file_menu.addAction(save_action)
        file_menu.addAction(save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
    
    def open_file(self):
        filters = ';;'.join(FILTER_TYPE)
        filename, selected_filter = QFileDialog.getOpenFileName(
            self,
            filter=filters,
        )
        if filename and selected_filter == FILTER_TYPE[0]:
            em_data = MatLoader.load(filename)
            key = self.data_manager.add(em_data)
            if self.welcome is not None:
                try:
                    self.welcome.hide()
                except RuntimeError:
                    self.welcome = None
            self.setCentralWidget(self.vertical_splitter)
            self.data_manager.set_active(key)
        else:
            return

    def save_file(self):
        self.save_file_for(self.activate_id)

    def save_file_for(self, data_id: int):
        if data_id == 0:
            QMessageBox.warning(self, "错误", "请设置激活数据")
            return

        if self.activate_id != data_id:
            self.data_manager.set_active(data_id)

        data = self.data_manager.get(data_id)
        if data is not None and isinstance(data, MatEMData):
            MatSaver.save(data, data.path)
        else:
            return

    def save_file_as(self):
        self.save_file_as_for(self.activate_id)

    def save_file_as_for(self, data_id: int):
        if data_id == 0:
            QMessageBox.warning(self, "错误", "请设置激活数据")
            return

        if self.activate_id != data_id:
            self.data_manager.set_active(data_id)

        filters = ';;'.join(FILTER_TYPE)
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            filter=filters,
        )
        data = self.data_manager.get(data_id)
        if filename and selected_filter == FILTER_TYPE[0]:
            MatSaver.save(data, filename)
        else:
            return

    def on_activate_changed(self, activate_id: int):
        self.activate_id = activate_id
        viewer = self._ensure_data_viewer(activate_id)
        if viewer is None:
            self.data_view_stack.setCurrentWidget(self.data_view_placeholder)
        else:
            self.data_view_stack.setCurrentWidget(viewer)

        process_widget = self._ensure_data_process_widget(activate_id)
        if process_widget is None:
            self.data_process_stack.setCurrentWidget(self.data_process_placeholder)
        else:
            self.data_process_stack.setCurrentWidget(process_widget)

    def on_data_change_finished(self, data_id: int, channel: str):
        viewer = self.data_viewers.get(data_id)
        if viewer is not None:
            viewer.on_change_data_finished(channel)

    def on_data_removed(self, key: int):
        viewer = self.data_viewers.pop(key, None)
        if viewer is not None:
            self.data_view_stack.removeWidget(viewer)
            viewer.close()
            viewer.deleteLater()

        process_widget = self.data_process_widgets.pop(key, None)
        if process_widget is not None:
            self.data_process_stack.removeWidget(process_widget)
            process_widget.close()
            process_widget.deleteLater()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "确认退出", "退出程序吗？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            for viewer in self.data_viewers.values():
                viewer.close()
            for widget in self.data_process_widgets.values():
                widget.close()
            event.accept()
        else:
            event.ignore()
