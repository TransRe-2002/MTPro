import logging
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFont, QIcon
from collections.abc import Callable

from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QMainWindow,
    QMessageBox,
    QFileDialog,
    QInputDialog,
    QProgressDialog,
    QTabWidget,
    QSplitter,
    QStackedWidget,
    QWizard,
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QToolBar,
    QSizePolicy, QHBoxLayout,
)
from typing import Dict, Optional

from base.data_manager import DataManager
from ui.data_process import DataProcessWidget
from ui.data_tree_view import DataTreeViewer
from ui.data_view_widget import DataViewWidget
from ui.log_console import LogConsoleWidget, QtTextEditHandler, configure_application_logging, log_session_banner
from ui.welcome import Welcome
from utils.app_state import AppState, load_app_state, remember_recent_file, remove_recent_file, save_app_state

FILTER_TYPE = [
    "Matlab数据文件 (*.mat)",
]

DATA_MANAGER_INITIAL_WIDTH = 220
WORKSPACE_INITIAL_WIDTH = 1260


class MainWindow(QMainWindow):
    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MTPro - 大地电磁时间序列处理工作台")
        self.setObjectName("MainWindow")
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        self.data_tree_viewer = DataTreeViewer(self.data_manager, self)
        self.activate_id: int = 0
        self.data_viewers: Dict[int, DataViewWidget] = {}
        self.app_state: AppState = load_app_state()
        self._log_last_nonzero_height = 120
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(False)
        self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter.setOpaqueResize(False)
        self.workspace_splitter.addWidget(self.data_tree_viewer)
        self.workspace_splitter.addWidget(self.tab_widget)
        self.workspace_splitter.setSizes(
            self.app_state.workspace_splitter_sizes
            if self.app_state.workspace_splitter_sizes
            else [DATA_MANAGER_INITIAL_WIDTH, WORKSPACE_INITIAL_WIDTH]
        )
        self.data_view_stack = QStackedWidget(self)
        self.data_view_placeholder = self._create_placeholder(
            "波形浏览 / Waveform View",
            "在左侧数据管理器中选择一个台站数据集，此处将显示各分量波形、Kp 指数及时间导航控制。",
        )
        self.data_view_stack.addWidget(self.data_view_placeholder)
        self.tab_widget.addTab(self.data_view_stack, "波形浏览")
        self.data_process_widget = DataProcessWidget(self.data_manager, self)
        self.data_process_widget.set_context_resolver(self._resolve_processing_target)
        self.data_process_widget.set_selection_payload_resolver(self.data_tree_viewer.current_drag_payload)
        self.data_process_widget.set_clear_checked_channels(self.data_tree_viewer.clear_checked_channels)
        self.tab_widget.addTab(self.data_process_widget, "数据处理")

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
        self.workspace_with_log_splitter.setSizes(
            self.app_state.workspace_with_log_splitter_sizes
            if self.app_state.workspace_with_log_splitter_sizes
            else [760, self.log_console.preferred_height()]
        )
        if not self.app_state.log_visible:
            total = sum(self.workspace_with_log_splitter.sizes()) or 1
            self.workspace_with_log_splitter.setSizes([total, 0])
        self.workspace_with_log_splitter.splitterMoved.connect(self._on_workspace_splitter_moved)
        self.workspace_splitter.splitterMoved.connect(self._on_main_splitter_moved)

        self.create_menu_bar()
        self._create_main_toolbar()
        self._create_status_bar()
        self.welcome = Welcome(self)
        self.welcome.set_recent_files(self.app_state.recent_files)
        self.central_stack = QStackedWidget(self)
        self.central_stack.addWidget(self.welcome)
        self.central_stack.addWidget(self.workspace_with_log_splitter)

        self.welcome.open_button.clicked.connect(self.open_file)
        self.welcome.continue_button.clicked.connect(self._continue_to_workspace)
        self.welcome.recent_file_requested.connect(self._open_recent_file)
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        self.data_tree_viewer.tool_requested.connect(self._open_processing_tool_from_sidebar)
        self.data_process_widget.change_finished_signal.connect(self.on_data_change_finished)
        self.data_manager.data_added.connect(self.on_data_added)
        self.data_manager.active_changed.connect(self.on_activate_changed)
        self.data_manager.data_removed.connect(self.on_data_removed)

        self.setCentralWidget(self.central_stack)
        self.central_stack.setCurrentWidget(self.welcome)
        self.resize(1480, 920)
        self._restore_active_tab()
        self._on_tab_changed(self.tab_widget.currentIndex())
        log_session_banner(self.logger, self.log_path)
        self.logger.info("Main window initialized")

    def _create_placeholder(self, title: str, body: str) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.addStretch(1)

        panel = QFrame(widget)
        panel.setFrameShape(QFrame.Shape.Box)
        panel.setFrameShadow(QFrame.Shadow.Sunken)

        # 让 panel 水平方向可拉伸，但不超过一定比例
        panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(32, 28, 32, 28)
        panel_layout.setSpacing(10)

        title_label = QLabel(title, panel)
        title_label.setObjectName("PlaceholderTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))

        body_label = QLabel(body, panel)
        body_label.setObjectName("PlaceholderBody")
        body_label.setWordWrap(True)
        body_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body_label.setFont(QFont("Arial", 10))

        sep = QFrame(panel)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)

        panel_layout.addWidget(title_label)
        panel_layout.addWidget(sep)
        panel_layout.addWidget(body_label)

        # 用一个水平布局包裹 panel，两侧加弹性空间
        h_layout = QHBoxLayout()
        h_layout.addStretch(1)
        h_layout.addWidget(panel, stretch=2)  # panel 占 2/4 的宽度
        h_layout.addStretch(1)

        layout.addLayout(h_layout)
        layout.addStretch(1)
        return widget

    def _create_main_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏", self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        open_tb_action = QAction(QIcon.fromTheme("document-open"), "打开", self)
        open_tb_action.setToolTip("打开 MAT 数据文件 (Ctrl+O) — 可多选")
        open_tb_action.triggered.connect(self.open_file)
        toolbar.addAction(open_tb_action)

        open_folder_tb_action = QAction(QIcon.fromTheme("folder-open"), "批量", self)
        open_folder_tb_action.setToolTip("批量打开文件夹下所有 MAT 文件 (Ctrl+Shift+O)")
        open_folder_tb_action.triggered.connect(self.open_folder)
        toolbar.addAction(open_folder_tb_action)

        save_tb_action = QAction(QIcon.fromTheme("document-save"), "保存", self)
        save_tb_action.setToolTip("保存当前数据集 (Ctrl+S)")
        save_tb_action.triggered.connect(self.save_file)
        toolbar.addAction(save_tb_action)

        toolbar.addSeparator()

        workspace_tb_action = QAction(QIcon.fromTheme("go-home"), "工作区", self)
        workspace_tb_action.setToolTip("切换到工作区视图")
        workspace_tb_action.triggered.connect(self._continue_to_workspace)
        toolbar.addAction(workspace_tb_action)

        toolbar.addSeparator()

        # 日志开关按钮（右对齐填充）
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.toggle_log_tb_action = QAction(QIcon.fromTheme("utilities-terminal"), "日志", self)
        self.toggle_log_tb_action.setCheckable(True)
        self.toggle_log_tb_action.setChecked(True)
        self.toggle_log_tb_action.setToolTip("显示/隐藏运行日志面板")
        self.toggle_log_tb_action.triggered.connect(self._toggle_log_console)
        toolbar.addAction(self.toggle_log_tb_action)

        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

    def _create_status_bar(self) -> None:
        sb = self.statusBar()
        sb.setSizeGripEnabled(True)
        self.dataset_status_label = QLabel("未加载数据", self)
        self.dataset_status_label.setFont(QFont("Arial", 9))
        sep_label = QLabel(" | ", self)
        self.path_status_label = QLabel(f"日志: {self.log_path}", self)
        self.path_status_label.setFont(QFont("Arial", 9))
        sb.addWidget(self.dataset_status_label, 1)
        sb.addPermanentWidget(sep_label)
        sb.addPermanentWidget(self.path_status_label)

    def _continue_to_workspace(self) -> None:
        self.central_stack.setCurrentWidget(self.workspace_with_log_splitter)
        self.logger.info("Switched from welcome screen to workspace")

    def _on_main_splitter_moved(self, _pos: int, _index: int) -> None:
        self._persist_app_state()

    def _on_workspace_splitter_moved(self, _pos: int, _index: int) -> None:
        sizes = self.workspace_with_log_splitter.sizes()
        if len(sizes) >= 2:
            log_visible = sizes[1] > 0
            if log_visible:
                self._log_last_nonzero_height = sizes[1]
            for action_name in ("toggle_log_action", "toggle_log_tb_action"):
                action = getattr(self, action_name, None)
                if action is not None:
                    action.blockSignals(True)
                    action.setChecked(log_visible)
                    action.blockSignals(False)
        self._persist_app_state()

    def _toggle_log_console(self, checked: bool) -> None:
        total = max(1, self.workspace_with_log_splitter.height())
        if checked:
            log_height = max(80, self._log_last_nonzero_height)
            self.workspace_with_log_splitter.setSizes([max(1, total - log_height), log_height])
        else:
            self.workspace_with_log_splitter.setSizes([total, 0])
        # 同步工具栏按钮状态
        if hasattr(self, "toggle_log_tb_action"):
            self.toggle_log_tb_action.blockSignals(True)
            self.toggle_log_tb_action.setChecked(checked)
            self.toggle_log_tb_action.blockSignals(False)
        self._persist_app_state()

    def _ensure_data_viewer(self, key: int) -> Optional[DataViewWidget]:
        viewer = self.data_viewers.get(key)
        if viewer is not None:
            return viewer

        em_data = self.data_manager.get(key)
        if em_data is None:
            return None

        callback = getattr(self, '_load_progress_callback', None)
        viewer = DataViewWidget(self.data_view_stack)
        viewer.init_data(em_data, callback)
        self.data_viewers[key] = viewer
        self.data_view_stack.addWidget(viewer)
        self.logger.info("Created data viewer for dataset #%s (%s)", key, em_data.name)
        return viewer

    def _persist_app_state(self) -> None:
        self.app_state.workspace_splitter_sizes = [int(size) for size in self.workspace_splitter.sizes()]
        self.app_state.workspace_with_log_splitter_sizes = [
            int(size) for size in self.workspace_with_log_splitter.sizes()
        ]
        log_sizes = self.workspace_with_log_splitter.sizes()
        self.app_state.log_visible = len(log_sizes) >= 2 and log_sizes[1] > 0
        current_index = self.tab_widget.currentIndex()
        if current_index >= 0:
            self.app_state.active_tab = self.tab_widget.tabText(current_index)
        save_app_state(self.app_state)

    def _restore_active_tab(self) -> None:
        target = self.app_state.active_tab
        if not target:
            return
        for index in range(self.tab_widget.count()):
            if self.tab_widget.tabText(index) == target:
                self.tab_widget.setCurrentIndex(index)
                return

    def _refresh_recent_file_views(self) -> None:
        if self.welcome is not None:
            self.welcome.set_recent_files(self.app_state.recent_files)
        self._refresh_recent_file_menu()

    def _refresh_recent_file_menu(self) -> None:
        if not hasattr(self, "recent_files_menu"):
            return

        self.recent_files_menu.clear()
        if not self.app_state.recent_files:
            placeholder = self.recent_files_menu.addAction("（暂无最近文件记录）")
            placeholder.setEnabled(False)
            return

        for path in self.app_state.recent_files:
            action = self.recent_files_menu.addAction(path)
            action.setToolTip(path)
            action.triggered.connect(
                lambda checked=False, current=path: self._open_recent_file(current)
            )

        self.recent_files_menu.addSeparator()
        clear_action = self.recent_files_menu.addAction("清空最近记录")
        clear_action.triggered.connect(self._clear_recent_files)

    def create_menu_bar(self):
        menubar = self.menuBar()
        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        open_action = QAction("打开(&O)...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("打开 MAT 格式数据文件（可多选）")
        open_action.triggered.connect(self.open_file)
        import_bin_action = QAction("从二进制文件导入(&I)...", self)
        import_bin_action.setShortcut("Ctrl+I")
        import_bin_action.setStatusTip("以向导方式从二进制文件 (.dat/.inf) 导入数据")
        import_bin_action.triggered.connect(self.import_binary)
        open_folder_action = QAction("批量打开文件夹(&B)...", self)
        open_folder_action.setShortcut("Ctrl+Shift+O")
        open_folder_action.setStatusTip("打开指定文件夹下所有 MAT 文件")
        open_folder_action.triggered.connect(self.open_folder)
        save_action = QAction("保存(&S)", self)
        save_action.setShortcut("Ctrl+S")
        save_action.setStatusTip("保存当前激活的数据集")
        save_action.triggered.connect(self.save_file)
        save_as_action = QAction("另存为(&A)...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.setStatusTip("将当前数据集另存为新文件")
        save_as_action.triggered.connect(self.save_file_as)
        batch_save_action = QAction("批量保存(&M)...", self)
        batch_save_action.setShortcut("Ctrl+Shift+M")
        batch_save_action.setStatusTip("将所有已打开数据集批量保存到指定文件夹")
        batch_save_action.triggered.connect(self.batch_save)
        exit_action = QAction("退出(&X)", self)
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Ctrl+Q")
        file_menu.addAction(open_action)
        file_menu.addAction(import_bin_action)
        file_menu.addSeparator()
        file_menu.addAction(open_folder_action)
        self.recent_files_menu = file_menu.addMenu("最近打开")
        file_menu.addSeparator()
        file_menu.addAction(save_action)
        file_menu.addAction(save_as_action)
        file_menu.addAction(batch_save_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        self._refresh_recent_file_menu()

        # 视图菜单
        view_menu = menubar.addMenu("视图(&V)")
        self.toggle_log_action = QAction("显示日志面板(&L)", self)
        self.toggle_log_action.setCheckable(True)
        self.toggle_log_action.setChecked(True)
        self.toggle_log_action.setShortcut("Ctrl+Shift+L")
        self.toggle_log_action.setStatusTip("切换底部运行日志面板的显示状态")
        self.toggle_log_action.triggered.connect(self._toggle_log_console)
        view_menu.addAction(self.toggle_log_action)
        show_workspace_action = QAction("切换到工作区(&W)", self)
        show_workspace_action.setStatusTip("从欢迎界面切换到主工作区")
        show_workspace_action.triggered.connect(self._continue_to_workspace)
        view_menu.addAction(show_workspace_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        about_action = QAction("关于 MTPro(&A)", self)
        about_action.setStatusTip("查看软件版本与版权信息")
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _show_about(self):
        QMessageBox.about(
            self,
            "关于 MTPro",
            "<b>MTPro</b> — 大地电磁时间序列处理工作台<br><br>"
            "版本：0.1.0<br>"
            "支持 MAT 格式数据的波形浏览、人工清洗与处理流程。<br><br>"
            "本软件基于 PySide6 / PyQtGraph 构建。",
        )
    
    def open_file(self):
        """打开一个或多个 MAT 文件（支持多选）。"""
        filters = ';;'.join(FILTER_TYPE)
        initial_dir = self.app_state.last_open_dir if self.app_state.last_open_dir else ""
        filenames, selected_filter = QFileDialog.getOpenFileNames(
            self,
            dir=initial_dir,
            filter=filters,
        )
        if not filenames or selected_filter != FILTER_TYPE[0]:
            self.logger.info("Open file dialog cancelled")
            return

        if not self._prompt_timezone_for_files(filenames):
            return
        self._open_files_batch(filenames, self._batch_tz)

    def import_binary(self):
        """以向导方式从二进制文件 (.dat/.inf) 导入数据。"""
        from ui.import_bin_wizard import ImportBinWizard

        wizard = ImportBinWizard(self.data_manager, self)
        if wizard.exec() != QWizard.DialogCode.Accepted:
            self.logger.info("Binary import wizard cancelled")
            return

        key = wizard.imported_key()
        if key is not None:
            # 记录到最近文件
            dat_path = wizard.dat_path()
            if dat_path:
                self.app_state = remember_recent_file(self.app_state, dat_path)
                self._persist_app_state()
                self._refresh_recent_file_views()
            self._continue_to_workspace()
            self.data_manager.set_active(key)
            self.logger.info("Binary import completed: key=%d, path=%s", key, dat_path)

    def open_folder(self):
        """批量打开文件夹下所有 MAT 文件。"""
        initial_dir = self.app_state.last_open_dir if self.app_state.last_open_dir else ""
        folder = QFileDialog.getExistingDirectory(
            self,
            caption="选择包含 MAT 文件的文件夹",
            dir=initial_dir,
        )
        if not folder:
            self.logger.info("Batch open folder cancelled")
            return

        mat_files = sorted(Path(folder).glob("*.mat"))
        if not mat_files:
            QMessageBox.information(self, "未找到文件", f"该文件夹下没有 .mat 文件：\n{folder}")
            return

        filenames = [str(f) for f in mat_files]
        if not self._prompt_timezone_for_files(filenames):
            return
        self._open_files_batch(filenames, self._batch_tz)

    # ------------------------------------------------------------------
    # 批量打开辅助
    # ------------------------------------------------------------------

    def _prompt_timezone_for_files(self, filenames: list[str]) -> bool:
        """弹出一次时区选择对话框（批量打开共用）。返回 True 表示用户已确认。"""
        from ui.timezone_dialog import TimezoneDialog

        tz_dialog = TimezoneDialog(filenames[0], self)
        if tz_dialog.exec() != QDialog.DialogCode.Accepted:
            self.logger.info("Batch open cancelled at timezone selection")
            return False
        self._batch_tz = tz_dialog.selected_timezone()
        self.logger.info("Batch open — selected timezone %s for %d files",
                          self._batch_tz, len(filenames))
        return True

    def _open_files_batch(self, filenames: list[str], tz: str) -> None:
        """批量打开文件：单个进度条追踪已打开个数，遇错跳过继续。"""
        from PySide6.QtWidgets import QApplication

        total = len(filenames)
        progress = QProgressDialog(
            f"正在打开 0/{total} 个文件...", "", 0, total, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(300)
        progress.setWindowTitle("批量加载数据")
        progress.setCancelButton(None)
        progress.setValue(0)

        opened = 0
        for filename in filenames:
            progress.setLabelText(f"正在打开 {opened}/{total} 个文件...\n{Path(filename).name}")
            QApplication.processEvents()

            try:
                self._open_file_path_with_tz(filename, tz, progress=progress)
                opened += 1
                progress.setValue(opened)
            except Exception:
                self.logger.exception("Batch open — skipped: %s", filename)

        progress.setValue(total)
        progress.close()

    def _open_file_path_with_tz(self, filename: str, tz: str,
                                 from_recent: bool = False,
                                 progress: Optional[QProgressDialog] = None) -> None:
        """使用指定时区加载单个文件（不再弹时区选择窗）。"""
        normalized = str(Path(filename).expanduser().resolve(strict=False))
        if not Path(normalized).exists():
            if from_recent:
                self.app_state = remove_recent_file(self.app_state, normalized)
                self._persist_app_state()
                self._refresh_recent_file_views()
            QMessageBox.warning(self, "文件不存在", f"无法找到文件:\n{normalized}")
            self.logger.warning("File not found: %s", normalized)
            return

        own_progress = None
        is_batch = progress is not None
        try:
            from io_utils.mat_io import MatLoader
            from PySide6.QtWidgets import QApplication

            if is_batch:
                # 批量模式：复用外部进度条，仅更新 label
                if progress is not None:
                    progress.setLabelText(
                        f"正在读取 {Path(normalized).name} ...")
                    QApplication.processEvents()
            else:
                # 单个文件模式：创建自己的进度条
                own_progress = QProgressDialog(
                    f"正在读取 {Path(normalized).name} ...", "", 0, 3, self)
                own_progress.setWindowModality(Qt.WindowModality.WindowModal)
                own_progress.setMinimumDuration(500)
                own_progress.setWindowTitle("加载数据")
                own_progress.setCancelButton(None)
                own_progress.setValue(0)
                progress = own_progress

            em_data = MatLoader.load(normalized, tz=tz)

            if not is_batch:
                total_steps = 3 + em_data.NCh + (1 if em_data.kp_data is not None else 0)
                progress.setMaximum(total_steps)
                progress.setValue(1)
                progress.setLabelText("正在读取通道数据...")
                QApplication.processEvents()

            key = self.data_manager.add(em_data)
            self.app_state = remember_recent_file(self.app_state, normalized)
            self._persist_app_state()
            self._refresh_recent_file_views()
            self._continue_to_workspace()

            if not is_batch:
                progress.setValue(2)
                progress.setLabelText("正在初始化通道曲线...")
                QApplication.processEvents()

                step = [2]
                def _on_progress(msg: str):
                    step[0] += 1
                    progress.setValue(step[0])
                    progress.setLabelText(msg)
                    QApplication.processEvents()

                self._load_progress_callback = _on_progress
                self.data_manager.set_active(key)
                self._load_progress_callback = None

                progress.setValue(total_steps)
            else:
                self.data_manager.set_active(key)

            self.logger.info("Opened file: %s", normalized)
        except Exception:
            self.logger.exception("Failed to open file: %s", normalized)
            if not is_batch:
                QMessageBox.critical(self, "打开失败", f"无法打开文件:\n{normalized}")
            else:
                raise  # 抛给 _open_files_batch 处理
        finally:
            if own_progress is not None:
                own_progress.close()

    def _open_file_path(self, filename: str, from_recent: bool = False) -> None:
        """单个文件打开（含时区弹窗）。用于最近文件等逐文件打开场景。"""
        normalized = str(Path(filename).expanduser().resolve(strict=False))
        if not Path(normalized).exists():
            if from_recent:
                self.app_state = remove_recent_file(self.app_state, normalized)
                self._persist_app_state()
                self._refresh_recent_file_views()
            QMessageBox.warning(self, "文件不存在", f"无法找到文件:\n{normalized}")
            self.logger.warning("File not found: %s", normalized)
            return

        from ui.timezone_dialog import TimezoneDialog

        tz_dialog = TimezoneDialog(normalized, self)
        if tz_dialog.exec() != QDialog.DialogCode.Accepted:
            self.logger.info("File open cancelled at timezone selection: %s", normalized)
            return
        tz = tz_dialog.selected_timezone()

        self._open_file_path_with_tz(filename, tz, from_recent=from_recent)

    def _open_recent_file(self, filename: str) -> None:
        self._open_file_path(filename, from_recent=True)

    def _clear_recent_files(self) -> None:
        self.app_state.recent_files = []
        self._persist_app_state()
        self._refresh_recent_file_views()

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
            # 二次确认：提示将覆盖原文件
            reply = QMessageBox.question(
                self, "确认保存",
                f"是否保存到：\n{data.path}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

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
        data = self.data_manager.get(data_id)
        default_name = data.name if data else ""

        dialog = QFileDialog(self)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilters(FILTER_TYPE)
        dialog.selectFile(default_name)
        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return
        filenames = dialog.selectedFiles()
        if not filenames:
            return
        filename = filenames[0]
        selected_filter = dialog.selectedNameFilter()
        # 根据选择的过滤器自动补扩展名
        import re
        m = re.search(r'\*\.(\w+)', selected_filter)
        if m and not Path(filename).suffix:
            filename += "." + m.group(1)
        if selected_filter == FILTER_TYPE[0]:
            try:
                from io_utils.mat_io import MatSaver

                MatSaver.save(data, filename)
                self.logger.info("Saved dataset #%s as %s", data_id, filename)
            except Exception:
                self.logger.exception("Failed to save dataset #%s as %s", data_id, filename)
                QMessageBox.critical(self, "另存为失败", f"无法保存到:\n{filename}")
        else:
            return

    def batch_save(self):
        """批量保存 DataManager 中所有数据集到指定文件夹。"""
        from io_utils.mat_io import MatSaver

        items = list(self.data_manager.items())
        if not items:
            QMessageBox.information(self, "提示", "没有已加载的数据集。")
            return

        dlg = _BatchSaveDialog(items, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        folder = dlg.folder_path.text()
        suffix = dlg.suffix_edit.text().strip()

        def _unique_path(folder: str, name: str, suffix: str) -> str:
            base = f"{name}{suffix}"
            path = os.path.join(folder, f"{base}.mat")
            if not os.path.exists(path):
                return path
            n = 1
            while True:
                path = os.path.join(folder, f"{base}({n}).mat")
                if not os.path.exists(path):
                    return path
                n += 1

        saved, failed, skipped = 0, 0, 0
        for data_id, data in items:
            name = data.name if data.name else f"dataset_{data_id}"
            path = _unique_path(folder, name, suffix)
            try:
                MatSaver.save(data, path)
                self.logger.info("Batch saved #%s to %s", data_id, path)
                saved += 1
            except Exception:
                self.logger.exception("Batch save failed for #%s", data_id)
                failed += 1

        msg = f"成功: {saved} 个\n失败: {failed} 个"
        if skipped:
            msg += f"\n跳过: {skipped} 个"
        QMessageBox.information(self, "批量保存完成", msg)


class _BatchSaveDialog(QDialog):
    """批量保存对话框：文件夹选择 + 后缀 + 预览。"""
    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量保存")
        self.resize(550, 350)

        layout = QVBoxLayout(self)

        # 文件夹选择
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("目标文件夹:"))
        self.folder_path = QLineEdit()
        self.folder_path.setReadOnly(True)
        folder_row.addWidget(self.folder_path, 1)
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse)
        folder_row.addWidget(browse_btn)
        layout.addLayout(folder_row)

        # 后缀
        suffix_row = QHBoxLayout()
        suffix_row.addWidget(QLabel("文件名后缀:"))
        self.suffix_edit = QLineEdit()
        self.suffix_edit.setPlaceholderText("如 _fixed")
        suffix_row.addWidget(self.suffix_edit, 1)
        layout.addLayout(suffix_row)

        # 预览
        layout.addWidget(QLabel("保存预览（最多显示前 5 个）:"))
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(120)
        layout.addWidget(self.preview)

        # 按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        save_btn = QPushButton("批量保存")
        save_btn.clicked.connect(self.accept)
        save_btn.setDefault(True)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        self._items = items
        self._update_preview()

        self.suffix_edit.textChanged.connect(self._update_preview)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(
            self, "选择批量保存的目标文件夹",
            options=QFileDialog.Option.ShowDirsOnly,
        )
        if folder:
            self.folder_path.setText(folder)
            self._update_preview()

    def _update_preview(self):
        folder = self.folder_path.text() or "<未选择>"
        suffix = self.suffix_edit.text().strip()
        lines = []
        for i, (_, data) in enumerate(self._items):
            if i >= 5:
                lines.append(f"  ... 共 {len(self._items)} 个文件")
                break
            name = data.name if data.name else f"dataset_{i}"
            lines.append(f"  {name}{suffix}.mat")
        self.preview.setPlainText(f"目标: {folder}\n\n" + "\n".join(lines))

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

        if activate_id != 0:
            data = self.data_manager.get(activate_id)
            if data is not None:
                self.data_process_widget.init_data(data, activate_id)

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
        self.data_process_widget.close_windows_for_data(key)
        self.logger.info("Removed dataset #%s", key)

    def closeEvent(self, event):
        if not self.isVisible():
            for viewer in self.data_viewers.values():
                viewer.close()
            self.data_process_widget.close()
            event.accept()
            return

        reply = QMessageBox.question(self, "确认退出", "退出程序吗？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.logger.info("Application close requested by user")
            for viewer in self.data_viewers.values():
                viewer.close()
            self.data_process_widget.close()
            event.accept()
        else:
            self.logger.info("Application close cancelled")
            event.ignore()

    def _on_tab_changed(self, index: int):
        self._persist_app_state()
        if self.tab_widget.tabText(index) == "数据处理":
            self.data_tree_viewer.show_processing_tools()
        else:
            self.data_tree_viewer.show_station_summary()

    def _resolve_processing_target(self, tool_type: str = "") -> Optional[tuple[int, str]]:
        # Station-level tools (robust estimate, remote reference) only need a
        # dataset, not a specific channel.
        station_tools = {"robust estimate", "remote reference"}

        current_channel = self.data_tree_viewer.current_channel_ref()
        if current_channel is not None:
            data_id, ch = current_channel
            if tool_type in station_tools:
                return data_id, ch
            return current_channel

        data = self.data_tree_viewer.current_dataset()
        data_id = self.data_tree_viewer.current_dataset_key()
        if data is None or data_id is None:
            return None

        if tool_type in station_tools:
            return data_id, data.first_channel_name() or ""

        channel_names = list(data.data.keys())
        if not channel_names:
            return None

        channel_name, accepted = QInputDialog.getItem(
            self,
            "选择处理通道",
            f"为数据集 #{data_id} {data.name} 选择一个通道：",
            channel_names,
            0,
            False,
        )
        if not accepted or not channel_name:
            return None
        return data_id, channel_name

    def _open_processing_tool_from_sidebar(self, tool_type: str):
        target = self._resolve_processing_target(tool_type)
        if target is None:
            QMessageBox.information(self, "无法打开工具", "请先在左侧数据管理器中选择一个数据集或通道。")
            return

        data_id, channel_name = target
        data = self.data_manager.get(data_id)
        if data is None:
            QMessageBox.warning(self, "数据不存在", "当前选择的数据集已不可用。")
            return

        self._continue_to_workspace()
        self.data_manager.set_active(data_id)
        self.data_process_widget.init_data(data, data_id)
        self.tab_widget.setCurrentWidget(self.data_process_widget)
        self.data_process_widget.open_tool_for(data_id, channel_name, tool_type)
