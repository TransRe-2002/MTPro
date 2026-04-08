import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QFrame,
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
from ui.data_process import DataProcessWidget
from ui.data_tree_view import DataTreeViewer
from ui.data_view_widget import DataViewWidget
from ui.log_console import LogConsoleWidget, QtTextEditHandler, configure_application_logging, log_session_banner
from ui.welcome import Welcome

FILTER_TYPE = [
    "Matlab数据文件 (*.mat)",
]


class MainWindow(QMainWindow):
    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MTPro")
        self.setObjectName("MainWindow")
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        self.data_tree_viewer = DataTreeViewer(self.data_manager, self)
        self.activate_id: int = 0
        self.data_viewers: Dict[int, DataViewWidget] = {}
        self.data_process_widgets: Dict[int, DataProcessWidget] = {}
        self._log_last_nonzero_height = 120
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter.setOpaqueResize(False)
        self.workspace_splitter.addWidget(self.data_tree_viewer)
        self.workspace_splitter.addWidget(self.tab_widget)
        self.workspace_splitter.setSizes([340, 1180])
        self.data_view_stack = QStackedWidget(self)
        self.data_view_placeholder = self._create_placeholder(
            "查看数据",
            "在左侧选择一个台站数据集后，这里会显示分量波形、Kp 和时间导航。",
        )
        self.data_view_stack.addWidget(self.data_view_placeholder)
        self.tab_widget.addTab(self.data_view_stack, "查看数据")
        self.data_process_stack = QStackedWidget(self)
        self.data_process_placeholder = self._create_placeholder(
            "数据处理",
            "在左侧选中数据后，这里可进入去尖峰、去阶跃、对比与后续处理流程。",
        )
        self.data_process_stack.addWidget(self.data_process_placeholder)
        self.tab_widget.addTab(self.data_process_stack, "数据处理")

        self.log_console = LogConsoleWidget(self)
        self.log_console_handler = QtTextEditHandler()
        self.log_console_handler.emitter.message_ready.connect(self.log_console.append_log)
        self.log_path = configure_application_logging(self.log_console_handler)

        self.workspace_with_log_splitter = QSplitter(Qt.Orientation.Vertical)
        self.workspace_with_log_splitter.setOpaqueResize(False)
        self.workspace_with_log_splitter.setChildrenCollapsible(True)
        self.workspace_with_log_splitter.addWidget(self.workspace_splitter)
        self.workspace_with_log_splitter.addWidget(self.log_console)
        self.workspace_with_log_splitter.setCollapsible(0, False)
        self.workspace_with_log_splitter.setCollapsible(1, True)
        self.workspace_with_log_splitter.setStretchFactor(0, 10)
        self.workspace_with_log_splitter.setStretchFactor(1, 1)
        self.workspace_with_log_splitter.setSizes([760, self.log_console.preferred_height()])
        self.workspace_with_log_splitter.splitterMoved.connect(self._on_workspace_splitter_moved)

        self.create_menu_bar()
        self._create_status_bar()
        self.welcome = Welcome(self)

        self.welcome.open_button.clicked.connect(self.open_file)
        self.welcome.continue_button.clicked.connect(self._continue_to_workspace)
        self.data_manager.data_added.connect(self.on_data_added)
        self.data_manager.active_changed.connect(self.on_activate_changed)
        self.data_manager.data_removed.connect(self.on_data_removed)

        self.setCentralWidget(self.welcome)
        self.resize(1480, 920)
        log_session_banner(self.logger, self.log_path)
        self.logger.info("Main window initialized")

    def _create_placeholder(self, title: str, body: str) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.addStretch(1)
        panel = QFrame(widget)
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(24, 24, 24, 24)
        panel_layout.setSpacing(12)

        title_label = QLabel(title, panel)
        title_label.setObjectName("PlaceholderTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFixedHeight(40)
        title_label.setFont(QFont("Arial", 25))
        body_label = QLabel(body, panel)
        body_label.setObjectName("PlaceholderBody")
        body_label.setFixedHeight(40)
        body_label.setWordWrap(True)
        body_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        panel_layout.addWidget(title_label)
        panel_layout.addWidget(body_label)
        layout.addWidget(panel, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)
        return widget

    def _create_status_bar(self) -> None:
        self.dataset_status_label = QLabel("未加载数据", self)
        self.path_status_label = QLabel(f"日志文件: {self.log_path}", self)
        self.statusBar().addWidget(self.dataset_status_label, 1)
        self.statusBar().addPermanentWidget(self.path_status_label)

    def _continue_to_workspace(self) -> None:
        self.setCentralWidget(self.workspace_with_log_splitter)
        self.logger.info("Switched from welcome screen to workspace")

    def _on_workspace_splitter_moved(self, _pos: int, _index: int) -> None:
        sizes = self.workspace_with_log_splitter.sizes()
        if len(sizes) >= 2:
            log_visible = sizes[1] > 0
            if log_visible:
                self._log_last_nonzero_height = sizes[1]
            if hasattr(self, "toggle_log_action"):
                self.toggle_log_action.blockSignals(True)
                self.toggle_log_action.setChecked(log_visible)
                self.toggle_log_action.blockSignals(False)

    def _toggle_log_console(self, checked: bool) -> None:
        total = max(1, self.workspace_with_log_splitter.height())
        if checked:
            log_height = max(80, self._log_last_nonzero_height)
            self.workspace_with_log_splitter.setSizes([max(1, total - log_height), log_height])
        else:
            self.workspace_with_log_splitter.setSizes([total, 0])

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
        self.logger.info("Created data viewer for dataset #%s (%s)", key, em_data.name)
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
        self.logger.info("Created processing workspace for dataset #%s (%s)", key, em_data.name)
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

        view_menu = menubar.addMenu("视图(&V)")
        self.toggle_log_action = QAction("显示日志", self)
        self.toggle_log_action.setCheckable(True)
        self.toggle_log_action.setChecked(True)
        self.toggle_log_action.triggered.connect(self._toggle_log_console)
        view_menu.addAction(self.toggle_log_action)
        show_workspace_action = QAction("显示工作区", self)
        show_workspace_action.triggered.connect(self._continue_to_workspace)
        view_menu.addAction(show_workspace_action)
    
    def open_file(self):
        filters = ';;'.join(FILTER_TYPE)
        filename, selected_filter = QFileDialog.getOpenFileName(
            self,
            filter=filters,
        )
        if filename and selected_filter == FILTER_TYPE[0]:
            try:
                from io_utils.mat_io import MatLoader

                em_data = MatLoader.load(filename)
                key = self.data_manager.add(em_data)
                if self.welcome is not None:
                    try:
                        self.welcome.hide()
                    except RuntimeError:
                        self.welcome = None
                self._continue_to_workspace()
                self.data_manager.set_active(key)
                self.logger.info("Opened file: %s", filename)
            except Exception:
                self.logger.exception("Failed to open file: %s", filename)
                QMessageBox.critical(self, "打开失败", f"无法打开文件:\n{filename}")
        else:
            self.logger.info("Open file dialog cancelled")
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
        if data is not None:
            try:
                from io_utils.mat_io import MatSaver

                MatSaver.save(data, data.path)
                self.logger.info("Saved dataset #%s to %s", data_id, data.path)
            except Exception:
                self.logger.exception("Failed to save dataset #%s to %s", data_id, data.path)
                QMessageBox.critical(self, "保存失败", f"无法保存文件:\n{data.path}")
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
            try:
                from io_utils.mat_io import MatSaver

                MatSaver.save(data, filename)
                self.logger.info("Saved dataset #%s as %s", data_id, filename)
            except Exception:
                self.logger.exception("Failed to save dataset #%s as %s", data_id, filename)
                QMessageBox.critical(self, "另存为失败", f"无法保存到:\n{filename}")
        else:
            return

    def on_data_added(self, key: int):
        data = self.data_manager.get(key)
        if data is None:
            return
        self.logger.info(
            "Dataset added: #%s | %s | samples=%s | path=%s",
            key,
            data.name,
            data.npts,
            data.path,
        )

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

        if activate_id == 0:
            self.dataset_status_label.setText("未加载数据")
            self.logger.info("Active dataset cleared")
        else:
            data = self.data_manager.get(activate_id)
            if data is not None:
                self.dataset_status_label.setText(
                    f"当前数据: #{activate_id} | {data.name} | 样点: {data.npts}"
                )
                self.logger.info("Activated dataset #%s (%s)", activate_id, data.name)

    def on_data_change_finished(self, data_id: int, channel: str):
        viewer = self.data_viewers.get(data_id)
        if viewer is not None:
            viewer.on_change_data_finished(channel)
        self.logger.info("Data updated: dataset #%s, channel %s", data_id, channel)

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
        self.logger.info("Removed dataset #%s", key)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "确认退出", "退出程序吗？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.logger.info("Application close requested by user")
            for viewer in self.data_viewers.values():
                viewer.close()
            for widget in self.data_process_widgets.values():
                widget.close()
            event.accept()
        else:
            self.logger.info("Application close cancelled")
            event.ignore()
