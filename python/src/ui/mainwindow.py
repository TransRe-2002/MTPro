from __future__ import annotations

import logging
import os
from concurrent.futures import as_completed
from pathlib import Path

from PySide6.QtCore import Qt, QEventLoop, QObject, Signal
from PySide6.QtGui import QAction, QFont, QIcon
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QWizard,
)

from base.data_manager import DataManager
from base.plugin_loader import LoadReport, load_all_plugins
from ui.data_process import DataProcessWidget
from ui.data_process_pipeline import refresh_tool_meta
from ui.data_tree_view import DataTreeViewer
from ui.data_view_widget import DataViewWidget
from base.logging_setup import configure_application_logging, log_session_banner
from ui.widgets.log_console import LogConsoleWidget, QtTextEditHandler
from ui.plugin_manager_dialog import PluginManagerDialog
from ui.impedance_output import ImpedanceOutputWidget
from ui.impedance_preview import ImpedancePreviewWidget, ImpedanceResultPool
from ui.welcome import Welcome
from ui.dialogs.batch_save_dialog import BatchSaveDialog
from ui.tools import registry as tool_registry
from io_utils import file_ops
from base.concurrency import io_pool
from base.qt_bridge import run as run_in_pool
from base.app_state import (
    AppState,
    load_app_state,
    remember_recent_file,
    remove_recent_file,
    save_app_state,
)

FILTER_TYPE = [
    "Matlab数据文件 (*.mat)",
]

DATA_MANAGER_INITIAL_WIDTH = 220
WORKSPACE_INITIAL_WIDTH = 1260


# ---------------------------------------------------------------------------
# 后台任务（跑在共享线程池上，禁止碰 Qt 控件与 DataManager）
# ---------------------------------------------------------------------------

def _load_mat_task(filename: str, tz: str) -> tuple[str, object]:
    """解析单个 MAT（含 Kp 获取），返回 (规范化路径, EMData)。"""
    normalized = str(Path(filename).expanduser().resolve(strict=False))
    if not Path(normalized).exists():
        raise FileNotFoundError(normalized)
    return normalized, file_ops.load_mat(normalized, tz=tz)


def _save_mat_task(data: object, path: str) -> str:
    """保存单个数据集到指定路径，返回路径。

    注：后台化后保存理论上可与用户编辑并发（实际罕见：保存确认
    弹窗刚关闭，用户注意力在保存上）。
    """
    file_ops.save_mat(data, path)
    return path


class _BatchOpenSignals(QObject):
    """批量打开的信号总线：任务跑在 compute 池，编排 io 池逐文件加载。"""

    file_loaded = Signal(str, object)  # (规范化路径, EMData)
    failed = Signal(str, str)          # (文件名, 错误信息)
    advanced = Signal(int, int, str)   # (已完成数, 总数, 当前文件名)


def _estimate_open_memory_gb(filenames: list[str]) -> tuple[float, float | None]:
    """估算批量打开的内存开销。

    返回 (预估占用 GB, 系统可用 GB)。MAT v7.3 带压缩，实测内存
    占用约为磁盘大小的 2 倍；可用内存读 /proc/meminfo（非 Linux
    平台返回 None，跳过预警）。
    """
    total_disk = 0
    for name in filenames:
        try:
            total_disk += Path(name).stat().st_size
        except OSError:
            continue
    est_gb = total_disk * 2 / 1e9
    avail_gb: float | None = None
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    avail_gb = int(line.split()[1]) / (1024 * 1024)
                    break
    except (OSError, ValueError):
        pass
    return est_gb, avail_gb


class MainWindow(QMainWindow):
    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MTPro - 大地电磁时间序列处理工作台")
        self.setObjectName("MainWindow")
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        self.data_tree_viewer = DataTreeViewer(self.data_manager, self)
        self.activate_id: int = 0
        self.data_viewers: dict[int, DataViewWidget] = {}
        self.app_state: AppState = load_app_state()
        self.plugin_report: LoadReport = load_all_plugins(state=self.app_state)
        tool_registry.register_loaded_plugins(self.plugin_report.loaded)
        refresh_tool_meta()
        self._log_last_nonzero_height = 120
        # 后台打开/保存的临时状态（供主线程槽函数读写）
        self._single_open_progress: QProgressDialog | None = None
        self._single_open_ctx: dict | None = None
        self._batch_progress: QProgressDialog | None = None
        self._batch_loop: QEventLoop | None = None
        self._batch_failed: list[tuple[str, str]] = []
        self._batch_save_progress: QProgressDialog | None = None
        self._save_ctx: dict | None = None
        self._viewer_build_progress: QProgressDialog | None = None
        # 波形视图"现场"状态（视口/通道顺序/隐藏通道/行高）：
        # viewer 控件是 active-only 的（切走即销毁），但现场按 key
        # 永久保存，再次激活时原样恢复
        self.viewer_state: dict[int, dict] = {}
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
        self.impedance_pool = ImpedanceResultPool(self)
        self.impedance_output_widget = ImpedanceOutputWidget(self.data_manager, self.impedance_pool, self)
        self.tab_widget.addTab(self.impedance_output_widget, "阻抗输出")
        self.impedance_preview_widget = ImpedancePreviewWidget(self.impedance_pool, self)
        self.tab_widget.addTab(self.impedance_preview_widget, "阻抗预览")
        self.impedance_output_widget.preview_requested.connect(self._show_impedance_preview)

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
        self.welcome.recent_files_requested.connect(self._open_recent_files)
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        self.data_tree_viewer.tool_requested.connect(self._open_processing_tool_from_sidebar)
        self.data_process_widget.change_finished_signal.connect(self.on_data_change_finished)
        self.data_manager.data_added.connect(self.on_data_added)
        self.data_manager.active_changed.connect(self.on_activate_changed)
        self.data_manager.data_removed.connect(self.on_data_removed)
        self.data_manager.data_changed.connect(self.on_data_changed)
        self.data_manager.data_restored.connect(self.on_data_restored)

        # DataManager 可能在 MainWindow 创建前就已有活动数据集；此时不会
        # 再发一次 active_changed，需主动同步数据处理页和阻抗输出页。
        initial_active_id = self.data_manager.active_id()
        if initial_active_id != 0:
            initial_data = self.data_manager.get(initial_active_id)
            if initial_data is not None:
                self.activate_id = initial_active_id
                self.data_process_widget.init_data(initial_data, initial_active_id)
                self.impedance_output_widget.init_data(initial_data, initial_active_id)

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

    def _ensure_data_viewer(self, key: int) -> DataViewWidget | None:
        viewer = self.data_viewers.get(key)
        if viewer is not None:
            return viewer

        em_data = self.data_manager.get(key)
        if em_data is None:
            return None

        viewer = DataViewWidget(self.data_view_stack)
        viewer.init_data(
            em_data, view_state=self.viewer_state.get(key), build_plots=False,
        )
        self.data_viewers[key] = viewer
        self.data_view_stack.addWidget(viewer)

        # 金字塔在 io 池异步构建（5s 采样年度文件需数秒）；进度框设了
        # 250ms 最小显示阈值——小文件构建快到根本不会闪现对话框
        progress = QProgressDialog("正在构建波形显示数据...", "", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(250)
        progress.setWindowTitle("加载数据")
        progress.setCancelButton(None)
        progress.show()
        self._viewer_build_progress = progress
        viewer.display_ready.connect(self._on_viewer_display_ready)
        viewer.start_display_build()

        self.logger.info("Created data viewer for dataset #%s (%s)", key, em_data.name)
        return viewer

    def _on_viewer_display_ready(self) -> None:
        if self._viewer_build_progress is not None:
            self._viewer_build_progress.close()
            self._viewer_build_progress = None

    def _destroy_data_viewer(self, key: int) -> None:
        """销毁指定数据集的 viewer 控件（active-only 策略的拆离路径）。"""
        viewer = self.data_viewers.pop(key, None)
        if viewer is not None:
            self.data_view_stack.removeWidget(viewer)
            viewer.close()
            viewer.deleteLater()

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
        import_xlsx_action = QAction("从 xlsx 时序文件导入(&X)...", self)
        import_xlsx_action.setShortcut("Ctrl+Shift+I")
        import_xlsx_action.setStatusTip("以向导方式批量导入 xlsx 时序数据")
        import_xlsx_action.triggered.connect(self.import_xlsx)
        import_txt_action = QAction("从 MT TXT 文件导入(&T)...", self)
        import_txt_action.setStatusTip(
            "读取 MT Portable 格式的空白分隔 TXT 时序文件"
        )
        import_txt_action.triggered.connect(self.import_txt)
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
        file_menu.addAction(import_xlsx_action)
        file_menu.addAction(import_txt_action)
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

        tools_menu = menubar.addMenu("工具(&T)")
        plugin_manager_action = QAction("插件管理器(&P)…", self)
        plugin_manager_action.setStatusTip("查看插件、管理扫描路径并安装本地插件")
        plugin_manager_action.triggered.connect(self._show_plugin_manager)
        tools_menu.addAction(plugin_manager_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        about_action = QAction("关于 MTPro(&A)", self)
        about_action.setStatusTip("查看软件版本与版权信息")
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _show_plugin_manager(self) -> None:
        dialog = PluginManagerDialog(self.app_state, self.plugin_report, self)
        dialog.exec()
        self.plugin_report = dialog.report

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

    def import_xlsx(self):
        """以向导方式从 xlsx 时序文件批量导入数据。"""
        from ui.import_xlsx_wizard import ImportXlsxWizard

        wizard = ImportXlsxWizard(self.data_manager, self)
        if wizard.exec() != QWizard.DialogCode.Accepted:
            self.logger.info("Xlsx import wizard cancelled")
            return

        keys = wizard.imported_keys()
        if not keys:
            key = wizard.imported_key()
            keys = [key] if key is not None else []
        if not keys:
            self.logger.info("Xlsx import finished without imported datasets")
            return

        self._continue_to_workspace()
        self.data_manager.set_active(keys[0])
        self.logger.info(
            "Xlsx import completed: keys=%s, mode=%s, files=%d",
            keys,
            wizard.import_mode(),
            len(wizard.selected_files()),
        )

    def import_txt(self):
        """通过向导一次性选择并导入一组 TXT 时序文件。"""
        from ui.import_txt_wizard import ImportTxtWizard

        wizard = ImportTxtWizard(self.data_manager, self)
        if wizard.exec() != QWizard.DialogCode.Accepted:
            self.logger.info("TXT import wizard cancelled")
            return

        key = wizard.imported_key()
        if key is None:
            self.logger.info("TXT import wizard completed without a registered dataset")
            return

        self._continue_to_workspace()
        self.data_manager.set_active(key)
        self.logger.info(
            "TXT group import completed: key=%d, files=%d",
            key,
            len(wizard.selected_files()),
        )

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
        from ui.dialogs.timezone_dialog import TimezoneDialog

        tz_dialog = TimezoneDialog(filenames[0], self)
        if tz_dialog.exec() != QDialog.DialogCode.Accepted:
            self.logger.info("Batch open cancelled at timezone selection")
            return False
        self._batch_tz = tz_dialog.selected_timezone()
        self.logger.info("Batch open — selected timezone %s for %d files",
                          self._batch_tz, len(filenames))
        return True

    def _open_files_batch(self, filenames: list[str], tz: str) -> None:
        """批量后台打开：io 池逐文件并行加载，完成后在主线程逐个入册。

        失败文件跳过并列清单（与 xlsx 导入策略一致）；取消按钮停止
        后续任务，已加载完成的数据集保留。
        """
        # 内存预警：几十个 5s 采样年度文件的原始数据就可达 ~10 GB 级，
        # 超过可用内存 70% 时让用户明确确认
        est_gb, avail_gb = _estimate_open_memory_gb(filenames)
        if avail_gb is not None and est_gb > max(1.0, avail_gb * 0.7):
            reply = QMessageBox.question(
                self,
                "内存预警",
                f"打开这 {len(filenames)} 个文件预计占用约 {est_gb:.1f} GB 内存，\n"
                f"当前机器可用约 {avail_gb:.1f} GB。全量加载可能导致内存不足。\n\n"
                f"仍要继续打开吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self.logger.info("Batch open declined at memory warning")
                return

        total = len(filenames)
        progress = QProgressDialog(
            f"正在打开 0/{total} 个文件...", "取消", 0, total, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setWindowTitle("批量加载数据")
        progress.setValue(0)

        cancel_flag = [False]
        progress.canceled.connect(lambda: cancel_flag.__setitem__(0, True))

        self._batch_progress = progress
        self._batch_failed = []
        signals = _BatchOpenSignals(self)
        signals.file_loaded.connect(self._on_batch_file_loaded)
        signals.failed.connect(self._on_batch_file_failed)
        signals.advanced.connect(self._on_batch_advanced)

        def task(sigs: _BatchOpenSignals) -> None:
            futures = {
                io_pool.submit(_load_mat_task, filename, tz): filename
                for filename in filenames
            }
            done = 0
            for future in as_completed(futures):
                filename = futures[future]
                try:
                    normalized, em_data = future.result()
                    sigs.file_loaded.emit(normalized, em_data)
                except Exception as exc:
                    sigs.failed.emit(filename, str(exc))
                done += 1
                sigs.advanced.emit(done, total, Path(filename).name)
                if cancel_flag[0]:
                    for pending in futures:
                        pending.cancel()
                    break

        loop = QEventLoop(self)
        self._batch_loop = loop
        # 编排任务跑在 compute 池（等 io 池的 future 属跨池等待，允许）；
        # on_done 是 QObject 槽，经排队连接回主线程。
        run_in_pool(
            "compute", task, signals,
            on_done=self._on_batch_finished, owner=self,
        )
        loop.exec()   # UI 保持响应：进度更新与取消都走事件循环
        progress.close()
        self._batch_progress = None
        self._batch_loop = None

        failed = self._batch_failed
        self._batch_failed = []
        if cancel_flag[0]:
            self.logger.info(
                "Batch open cancelled by user (%d/%d files loaded)",
                progress.value(), total,
            )
        if failed:
            QMessageBox.warning(
                self,
                "部分文件被跳过",
                "以下文件无法加载，已跳过:\n\n"
                + "\n".join(f"{Path(name).name}: {error}" for name, error in failed),
            )

    # -- 批量打开信号槽（排队连接，主线程执行） -------------------------

    def _on_batch_advanced(self, done: int, total: int, name: str) -> None:
        if self._batch_progress is not None:
            self._batch_progress.setValue(done)
            self._batch_progress.setLabelText(
                f"正在打开 {done}/{total} 个文件...\n{name}")

    def _on_batch_file_loaded(self, normalized: str, em_data) -> None:
        key = self.data_manager.add(em_data)
        self.app_state = remember_recent_file(self.app_state, normalized)
        self._persist_app_state()
        self._refresh_recent_file_views()
        self._continue_to_workspace()
        self.data_manager.set_active(key)
        self.logger.info("Opened file: %s", normalized)

    def _on_batch_file_failed(self, filename: str, error: str) -> None:
        self.logger.error("Batch open — skipped: %s (%s)", filename, error)
        self._batch_failed.append((filename, error))

    def _on_batch_finished(self, _result=None) -> None:
        if self._batch_loop is not None:
            self._batch_loop.quit()

    def _open_file_path_with_tz(self, filename: str, tz: str,
                                 from_recent: bool = False) -> None:
        """使用指定时区后台加载单个文件（不再弹时区选择窗）。"""
        normalized = str(Path(filename).expanduser().resolve(strict=False))
        if not Path(normalized).exists():
            if from_recent:
                self.app_state = remove_recent_file(self.app_state, normalized)
                self._persist_app_state()
                self._refresh_recent_file_views()
            QMessageBox.warning(self, "文件不存在", f"无法找到文件:\n{normalized}")
            self.logger.warning("File not found: %s", normalized)
            return

        # 不定量进度条：加载期间父窗口模态阻塞，但事件循环保持运转
        progress = QProgressDialog(
            f"正在读取 {Path(normalized).name} ...", "", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(200)
        progress.setWindowTitle("加载数据")
        progress.setCancelButton(None)
        progress.show()

        self._single_open_progress = progress
        self._single_open_ctx = {
            "normalized": normalized, "from_recent": from_recent,
        }
        run_in_pool(
            "io", _load_mat_task, normalized, tz,
            on_done=self._on_single_open_done,
            on_error=self._on_single_open_failed,
            owner=self,
        )

    def _on_single_open_done(self, result) -> None:
        normalized, em_data = result
        if self._single_open_progress is not None:
            self._single_open_progress.close()
            self._single_open_progress = None
        key = self.data_manager.add(em_data)
        self.app_state = remember_recent_file(self.app_state, normalized)
        self._persist_app_state()
        self._refresh_recent_file_views()
        self._continue_to_workspace()
        self.data_manager.set_active(key)
        self.logger.info("Opened file: %s", normalized)

    def _on_single_open_failed(self, exc: BaseException) -> None:
        if self._single_open_progress is not None:
            self._single_open_progress.close()
            self._single_open_progress = None
        ctx = self._single_open_ctx or {}
        normalized = ctx.get("normalized", "")
        if isinstance(exc, FileNotFoundError) and ctx.get("from_recent"):
            self.app_state = remove_recent_file(self.app_state, normalized)
            self._persist_app_state()
            self._refresh_recent_file_views()
        self.logger.error(
            "Failed to open file: %s: %r", normalized, exc, exc_info=exc)
        QMessageBox.critical(
            self, "打开失败", f"无法打开文件:\n{normalized}\n\n错误: {exc}")

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

        from ui.dialogs.timezone_dialog import TimezoneDialog

        tz_dialog = TimezoneDialog(normalized, self)
        if tz_dialog.exec() != QDialog.DialogCode.Accepted:
            self.logger.info("File open cancelled at timezone selection: %s", normalized)
            return
        tz = tz_dialog.selected_timezone()

        self._open_file_path_with_tz(filename, tz, from_recent=from_recent)

    def _open_recent_file(self, filename: str) -> None:
        self._open_recent_files([filename])

    def _open_recent_files(self, filenames: list[str]) -> None:
        """Open selected recent MAT files, using one timezone for a batch."""
        normalized_files = list(dict.fromkeys(
            str(Path(filename).expanduser().resolve(strict=False))
            for filename in filenames
            if filename
        ))
        if not normalized_files:
            return

        missing: list[str] = []
        unsupported: list[str] = []
        supported: list[str] = []
        for filename in normalized_files:
            if not Path(filename).exists():
                missing.append(filename)
            elif Path(filename).suffix.lower() != ".mat":
                unsupported.append(filename)
            else:
                supported.append(filename)

        if missing:
            for filename in missing:
                self.app_state = remove_recent_file(self.app_state, filename)
            self._persist_app_state()
            self._refresh_recent_file_views()

        skipped_messages: list[str] = []
        if missing:
            skipped_messages.append(
                "找不到以下文件，已从最近记录中移除：\n"
                + "\n".join(missing)
            )
        if unsupported:
            skipped_messages.append(
                "以下文件不能从最近列表直接打开，请使用对应的导入向导"
                "（TXT 请使用“从 MT TXT 文件导入”）：\n"
                + "\n".join(unsupported)
            )
        if skipped_messages:
            QMessageBox.warning(
                self,
                "部分文件未打开" if supported else "无法打开所选文件",
                "\n\n".join(skipped_messages),
            )

        if len(supported) == 1:
            self._open_file_path(supported[0], from_recent=True)
            return
        if not supported or not self._prompt_timezone_for_files(supported):
            return
        self._open_files_batch(supported, self._batch_tz)

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

            self._save_ctx = {"path": data.path, "data_id": data_id, "kind": "save"}
            run_in_pool(
                "io", _save_mat_task, data, data.path,
                on_done=self._on_save_done,
                on_error=self._on_save_failed,
                owner=self,
            )
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
            self._save_ctx = {"path": filename, "data_id": data_id, "kind": "save_as"}
            run_in_pool(
                "io", _save_mat_task, data, filename,
                on_done=self._on_save_done,
                on_error=self._on_save_failed,
                owner=self,
            )
        else:
            return

    def _on_save_done(self, path: str) -> None:
        ctx = self._save_ctx or {}
        verb = "as" if ctx.get("kind") == "save_as" else "to"
        self.logger.info("Saved dataset #%s %s %s", ctx.get("data_id"), verb, path)

    def _on_save_failed(self, exc: BaseException) -> None:
        ctx = self._save_ctx or {}
        path = ctx.get("path", "")
        self.logger.error("Failed to save %s: %r", path, exc, exc_info=exc)
        title = "另存为失败" if ctx.get("kind") == "save_as" else "保存失败"
        QMessageBox.critical(self, title, f"无法保存到:\n{path}\n\n错误: {exc}")

    def batch_save(self):
        """批量保存 DataManager 中所有数据集到指定文件夹。"""
        items = list(self.data_manager.items())
        if not items:
            QMessageBox.information(self, "提示", "没有已加载的数据集。")
            return

        dlg = BatchSaveDialog(items, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        folder = dlg.folder_path.text()
        suffix = dlg.suffix_edit.text().strip()

        progress = QProgressDialog("正在保存所有数据集...", "", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(200)
        progress.setWindowTitle("批量保存")
        progress.setCancelButton(None)
        progress.show()
        self._batch_save_progress = progress
        run_in_pool(
            "io", file_ops.batch_save_mat, items, folder, suffix,
            on_done=self._on_batch_save_done,
            on_error=self._on_batch_save_failed,
            owner=self,
        )

    def _on_batch_save_done(self, result) -> None:
        if self._batch_save_progress is not None:
            self._batch_save_progress.close()
            self._batch_save_progress = None
        saved, failed = result
        QMessageBox.information(
            self, "批量保存完成", f"成功: {saved} 个\n失败: {failed} 个")

    def _on_batch_save_failed(self, exc: BaseException) -> None:
        if self._batch_save_progress is not None:
            self._batch_save_progress.close()
            self._batch_save_progress = None
        self.logger.error("Batch save failed: %r", exc, exc_info=exc)
        QMessageBox.critical(self, "批量保存失败", f"错误: {exc}")

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

        # active-only：除即将激活的数据集外，其余 viewer 快照现场后销毁。
        # 金字塔/控件内存因此只保留一份，数据集个数不再放大显示开销。
        if self._viewer_build_progress is not None:
            self._viewer_build_progress.close()
            self._viewer_build_progress = None
        for old_key in [k for k in self.data_viewers if k != activate_id]:
            old_viewer = self.data_viewers.get(old_key)
            if old_viewer is not None:
                self.viewer_state[old_key] = old_viewer.capture_view_state()
            self._destroy_data_viewer(old_key)

        viewer = self._ensure_data_viewer(activate_id)
        if viewer is None:
            self.data_view_stack.setCurrentWidget(self.data_view_placeholder)
        else:
            self.data_view_stack.setCurrentWidget(viewer)

        if activate_id != 0:
            data = self.data_manager.get(activate_id)
            if data is not None:
                self.data_process_widget.init_data(data, activate_id)
                self.impedance_output_widget.init_data(data, activate_id)

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

    def on_data_restored(self, data_id: int, channel: str | None) -> None:
        if channel is not None:
            self.on_data_change_finished(data_id, channel)

    def on_data_changed(self, data_id: int, structural: bool) -> None:
        """就地预处理结果路由。

        structural=False：通道数值或元数据变化 → 无需完整重建视图。
        structural=True ：npts/dt/通道集变化 → 活跃数据集走
        「快照现场 → 销毁 → 重建」的 active-only 重建路径；
        非活跃数据集丢弃旧现场快照，下次激活时干净重建。
        （DataTreeModel 自身已监听 data_changed 刷新行。）
        """
        em_data = self.data_manager.get(data_id)
        if em_data is None:
            return

        if data_id == self.activate_id:
            self.data_tree_viewer.on_active_changed(data_id)
            self.dataset_status_label.setText(
                f"当前数据: #{data_id} | {em_data.name} | 样点: {em_data.npts}"
            )

        if not structural:
            self.logger.info("Dataset #%s content or metadata updated", data_id)
            return

        viewer = self.data_viewers.get(data_id)
        if viewer is None:
            self.viewer_state.pop(data_id, None)
            self.logger.info(
                "Dataset #%s restructured; viewer rebuild deferred", data_id,
            )
            return

        if self._viewer_build_progress is not None:
            self._viewer_build_progress.close()
            self._viewer_build_progress = None

        self.viewer_state[data_id] = viewer.capture_view_state()
        self._destroy_data_viewer(data_id)
        new_viewer = self._ensure_data_viewer(data_id)
        if new_viewer is not None:
            self.data_view_stack.setCurrentWidget(new_viewer)

        # 通道集可能已变：同步数据处理页上下文
        self.data_process_widget.init_data(em_data, data_id)
        self.impedance_output_widget.init_data(em_data, data_id)
        self.logger.info("Dataset #%s restructured; viewer rebuilt", data_id)

    def on_data_removed(self, key: int):
        viewer = self.data_viewers.pop(key, None)
        if viewer is not None:
            self.data_view_stack.removeWidget(viewer)
            viewer.close()
            viewer.deleteLater()
        self.viewer_state.pop(key, None)
        self.data_process_widget.close_windows_for_data(key)
        self.impedance_output_widget.close_windows_for_data(key)
        self.logger.info("Removed dataset #%s", key)

    def closeEvent(self, event):
        if not self.isVisible():
            for viewer in self.data_viewers.values():
                viewer.close()
            self.data_process_widget.close()
            self.impedance_output_widget.discard_and_close_all()
            event.accept()
            return

        unexported = self.impedance_pool.unexported_count()
        pending = self.impedance_output_widget.pending_calculation_count()
        running = self.impedance_output_widget.running_calculation_count()
        question = "退出程序吗？"
        warnings = []
        if unexported:
            warnings.append(f"阻抗预览中有 {unexported} 个临时结果尚未输出为 EDI、Z 或 CSV")
        if pending:
            warnings.append(f"有 {pending} 个计算结果尚未保存到阻抗预览")
        if running:
            warnings.append(f"有 {running} 个阻抗计算仍在运行")
        if warnings:
            question = "；\n".join(warnings) + "。\n退出后这些内容将丢失，仍要退出吗？"
        reply = QMessageBox.question(self, "确认退出", question,
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.logger.info("Application close requested by user")
            for viewer in self.data_viewers.values():
                viewer.close()
            self.data_process_widget.close()
            self.impedance_output_widget.discard_and_close_all()
            event.accept()
        else:
            self.logger.info("Application close cancelled")
            event.ignore()

    def _show_impedance_preview(self, snapshot_id: str = "") -> None:
        self.tab_widget.setCurrentWidget(self.impedance_preview_widget)
        if snapshot_id:
            self.impedance_preview_widget.select_snapshot(snapshot_id)

    def _on_tab_changed(self, index: int):
        self._persist_app_state()
        if self.tab_widget.tabText(index) == "数据处理":
            self.data_tree_viewer.show_processing_tools()
        else:
            self.data_tree_viewer.show_station_summary()

    def _resolve_processing_target(self, tool_type: str = "") -> tuple[int, str] | None:
        # Station-level tools (robust estimate, remote reference) only need a
        # dataset, not a specific channel.
        is_station = tool_registry.is_station_tool(tool_type)

        current_channel = self.data_tree_viewer.current_channel_ref()
        if current_channel is not None:
            data_id, ch = current_channel
            if is_station:
                return data_id, ch
            return current_channel

        data = self.data_tree_viewer.current_dataset()
        data_id = self.data_tree_viewer.current_dataset_key()
        if data is None or data_id is None:
            return None

        if is_station:
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
        self.impedance_output_widget.init_data(data, data_id)
        self.tab_widget.setCurrentWidget(self.data_process_widget)
        self.data_process_widget.open_tool_for(data_id, channel_name, tool_type)
