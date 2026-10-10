from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)


class Welcome(QWidget):
    recent_file_requested = Signal(str)
    recent_files_requested = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MTPro - 欢迎")
        self.setObjectName("WelcomeRoot")

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── 左侧：程序标识区 ──────────────────────────────────────
        left_panel = QFrame(self)
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_panel.setFrameShadow(QFrame.Shadow.Plain)
        left_panel.setFixedWidth(240)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 30, 20, 20)
        left_layout.setSpacing(6)

        app_title = QLabel("MTPro", left_panel)
        app_title.setObjectName("WelcomeTitle")
        app_title.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        app_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        app_subtitle = QLabel("大地电磁时间序列\n处理工作台", left_panel)
        app_subtitle.setObjectName("WelcomeSubtitle")
        app_subtitle.setFont(QFont("Arial", 10))
        app_subtitle.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        app_subtitle.setWordWrap(True)

        version_label = QLabel("Version 0.1.0", left_panel)
        version_label.setFont(QFont("Arial", 8))
        version_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        left_layout.addWidget(app_title)
        left_layout.addWidget(app_subtitle)
        left_layout.addSpacing(12)

        sep = QFrame(left_panel)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        left_layout.addWidget(sep)
        left_layout.addSpacing(6)

        info_label = QLabel(
            "支持格式：\n  • MAT 结构体（Matlab）\n\n"
            "主要功能：\n  • 波形浏览与导航\n  • 去尖峰 / 去阶跃\n  • 多序列对比\n  • 处理流程编排",
            left_panel
        )
        info_label.setFont(QFont("Arial", 9))
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        left_layout.addWidget(info_label)
        left_layout.addStretch(1)
        left_layout.addWidget(version_label)

        # ── 右侧：操作区 ─────────────────────────────────────────
        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(32, 32, 32, 32)
        right_layout.setSpacing(16)

        # 新建/打开 分组
        action_group = QGroupBox("开始", right_panel)
        action_layout = QVBoxLayout(action_group)
        action_layout.setContentsMargins(16, 16, 16, 16)
        action_layout.setSpacing(10)

        self.open_button = QPushButton("打开数据文件 (MAT)...", action_group)
        self.open_button.setMinimumHeight(28)
        self.open_button.setFont(QFont("Arial", 10))
        self.open_button.setToolTip("打开 MAT 格式的大地电磁时间序列数据文件 (Ctrl+O)")

        self.continue_button = QPushButton("进入工作区（无需加载数据）", action_group)
        self.continue_button.setMinimumHeight(28)
        self.continue_button.setFont(QFont("Arial", 10))
        self.continue_button.setToolTip("直接进入工作区，稍后再加载数据")

        action_layout.addWidget(self.open_button)
        action_layout.addWidget(self.continue_button)

        # 最近文件
        recent_group = QGroupBox("最近使用的文件", right_panel)
        self.recent_layout = QVBoxLayout(recent_group)
        self.recent_layout.setContentsMargins(16, 12, 16, 12)
        self.recent_layout.setSpacing(8)
        self.recent_placeholder = QLabel("（暂无最近文件记录）", recent_group)
        self.recent_placeholder.setFont(QFont("Arial", 9))
        self.recent_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recent_layout.addWidget(self.recent_placeholder)

        self.recent_model = QStandardItemModel(0, 2, recent_group)
        self.recent_model.setHorizontalHeaderLabels(["文件", "路径"])
        self.recent_view = QTreeView(recent_group)
        self.recent_view.setModel(self.recent_model)
        self.recent_view.setRootIsDecorated(False)
        self.recent_view.setAlternatingRowColors(True)
        self.recent_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.recent_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.recent_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.recent_view.setUniformRowHeights(True)
        self.recent_view.setSortingEnabled(False)
        self.recent_view.setToolTip("可用 Ctrl/Shift 多选；双击打开单个文件")
        self.recent_view.doubleClicked.connect(self._open_recent_index)
        self.recent_view.hide()
        header = self.recent_view.header()
        header.setStretchLastSection(True)
        header.resizeSection(0, 180)
        self.recent_layout.addWidget(self.recent_view, 1)

        self.open_recent_button = QPushButton("打开所选文件", recent_group)
        self.open_recent_button.setEnabled(False)
        self.open_recent_button.setToolTip("一次打开选中的一个或多个最近文件")
        self.open_recent_button.clicked.connect(self._open_selected_recent_files)
        self.recent_view.selectionModel().selectionChanged.connect(
            self._update_recent_selection
        )
        self.open_recent_button.hide()
        self.recent_layout.addWidget(self.open_recent_button)

        right_layout.addWidget(action_group)
        right_layout.addWidget(recent_group, 1)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

        self.setLayout(main_layout)

    def set_recent_files(self, paths: list[str]) -> None:
        self.recent_model.removeRows(0, self.recent_model.rowCount())

        has_items = bool(paths)
        self.recent_placeholder.setVisible(not has_items)
        self.recent_view.setVisible(has_items)
        self.open_recent_button.setVisible(has_items)
        self._update_recent_selection()
        if not has_items:
            return

        for path in paths:
            path_obj = Path(path)
            name = path_obj.name or path
            parent = str(path_obj.parent) if path_obj.name else ""
            name_item = QStandardItem(name)
            path_item = QStandardItem(parent)
            for item in (name_item, path_item):
                item.setEditable(False)
                item.setToolTip(path)
            name_item.setData(path, Qt.ItemDataRole.UserRole)
            self.recent_model.appendRow([name_item, path_item])

        self.recent_view.resizeColumnToContents(0)

    def _selected_recent_paths(self) -> list[str]:
        indexes = sorted(
            self.recent_view.selectionModel().selectedRows(0),
            key=lambda index: index.row(),
        )
        return [
            str(path)
            for index in indexes
            if (path := index.data(Qt.ItemDataRole.UserRole))
        ]

    def _update_recent_selection(self, *_args) -> None:
        count = len(self._selected_recent_paths())
        self.open_recent_button.setEnabled(count > 0)
        self.open_recent_button.setText(
            f"打开所选文件（{count}）" if count > 1 else "打开所选文件"
        )

    def _open_selected_recent_files(self) -> None:
        paths = self._selected_recent_paths()
        if len(paths) == 1:
            self.recent_file_requested.emit(paths[0])
        elif paths:
            self.recent_files_requested.emit(paths)

    def _open_recent_index(self, index) -> None:
        if not index.isValid():
            return
        path_index = self.recent_model.index(index.row(), 0)
        path: str | None = path_index.data(Qt.ItemDataRole.UserRole)
        if path:
            self.recent_file_requested.emit(str(path))
