from PySide6.QtCore import Signal, Qt, QObject
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QFileDialog,
    QTabWidget,
    QSplitter,
)

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
        self.tab_widget = QTabWidget()
        self.vertical_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.vertical_splitter.addWidget(self.data_tree_viewer)
        self.vertical_splitter.addWidget(self.tab_widget)
        self.data_viewer = DataViewWidget(self)
        self.tab_widget.addTab(self.data_viewer, "查看数据")
        self.data_process = DataProcessWidget(self)
        self.tab_widget.addTab(self.data_process, "数据处理")

        self.create_menu_bar()
        self.welcome = Welcome(self)

        self.welcome.open_button.clicked.connect(self.open_file)
        self.data_process.change_finished_signal.connect(
            self.data_viewer.on_change_data_finished
        )
        self.data_tree_viewer.activate_changed.connect(self.on_activate_changed)
        self.data_manager.updated_added.connect(self.data_tree_viewer.on_data_added)

        self.setCentralWidget(self.welcome)

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
            self.data_manager.add(em_data)
            if self.welcome is not None:
                try:
                    self.welcome.hide()
                except RuntimeError:
                    self.welcome = None
            self.setCentralWidget(self.vertical_splitter)
        else:
            return

    def save_file(self):
        data = self.data_manager.get(self.activate_id)
        if data is not None and isinstance(data, MatEMData):
            MatSaver.save(data, data.path)
        else:
            return

    def save_file_as(self):
        filters = ';;'.join(FILTER_TYPE)
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            filter=filters,
        )
        if filename and selected_filter == FILTER_TYPE[0]:
            MatSaver.save(self.data_manager.get(self.activate_id), filename)
        else:
            return

    def on_activate_changed(self, activate_id: int):
        self.activate_id = activate_id
        self.data_viewer.init_data(self.data_manager.get(activate_id))
        self.data_process.init_data(self.data_manager.get(activate_id))

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "确认退出", "退出程序吗？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if self.data_viewer is not None:
                self.data_viewer.close()
            event.accept()
        else:
            event.ignore()
