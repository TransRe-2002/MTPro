"""Plugin discovery report and scan-path management UI."""
from __future__ import annotations

from pathlib import Path
import shutil

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from base.plugin_loader import LoadReport, default_plugin_dir, load_all_plugins
from base.app_state import AppState, save_app_state


class PluginManagerDialog(QDialog):
    def __init__(
        self,
        app_state: AppState,
        report: LoadReport,
        parent=None,
    ):
        super().__init__(parent)
        self.app_state = app_state
        self.report = report
        self.setWindowTitle("插件管理器")
        self.resize(900, 620)

        layout = QVBoxLayout(self)
        tabs = QTabWidget(self)
        self.loaded_table = self._make_table(("插件 ID", "名称", "版本", "来源"))
        self.failed_table = self._make_table(("插件 ID", "来源", "失败原因"))
        tabs.addTab(self.loaded_table, "已加载")
        tabs.addTab(self.failed_table, "加载失败")
        layout.addWidget(tabs, 1)

        path_label = QLabel("扫描路径（修改后重新扫描；工具列表需重启应用才会更新）", self)
        layout.addWidget(path_label)
        self.path_list = QListWidget(self)
        self.path_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.path_list)

        controls = QHBoxLayout()
        add_button = QPushButton("添加目录…", self)
        remove_button = QPushButton("移除", self)
        install_button = QPushButton("从文件安装…", self)
        rescan_button = QPushButton("重新扫描", self)
        add_button.clicked.connect(self._add_path)
        remove_button.clicked.connect(self._remove_path)
        install_button.clicked.connect(self._install)
        rescan_button.clicked.connect(self._rescan)
        controls.addWidget(add_button)
        controls.addWidget(remove_button)
        controls.addWidget(install_button)
        controls.addStretch(1)
        controls.addWidget(rescan_button)
        layout.addLayout(controls)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._refresh_paths()
        self._refresh_report()

    @staticmethod
    def _make_table(headers: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setWordWrap(False)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _refresh_report(self) -> None:
        self.loaded_table.setRowCount(len(self.report.loaded))
        for row, loaded in enumerate(self.report.loaded):
            values = (
                loaded.info.plugin_id,
                loaded.info.display_name,
                loaded.info.version,
                loaded.source,
            )
            for column, value in enumerate(values):
                self.loaded_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.loaded_table.resizeColumnsToContents()

        self.failed_table.setRowCount(len(self.report.failed))
        for row, failure in enumerate(self.report.failed):
            values = (failure.plugin_id, failure.source, failure.reason)
            for column, value in enumerate(values):
                self.failed_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.failed_table.resizeColumnsToContents()

    def _refresh_paths(self) -> None:
        self.path_list.clear()
        default_item = QListWidgetItem(f"{default_plugin_dir()}  （默认）")
        default_item.setData(Qt.ItemDataRole.UserRole, "")
        default_item.setFlags(default_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.path_list.addItem(default_item)
        for path in self.app_state.plugin_paths:
            item = QListWidgetItem(path)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.path_list.addItem(item)

    def _add_path(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "添加插件扫描目录")
        if not selected:
            return
        normalized = str(Path(selected).expanduser().resolve(strict=False))
        if normalized not in self.app_state.plugin_paths:
            self.app_state.plugin_paths.append(normalized)
            save_app_state(self.app_state)
            self._refresh_paths()

    def _remove_path(self) -> None:
        item = self.path_list.currentItem()
        if item is None:
            return
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not path:
            return
        self.app_state.plugin_paths = [value for value in self.app_state.plugin_paths if value != path]
        save_app_state(self.app_state)
        self._refresh_paths()

    def _install(self) -> None:
        choice = QMessageBox.question(
            self,
            "安装插件",
            "选择“是”安装 zip 插件文件；选择“否”安装插件文件夹。",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
        )
        if choice == QMessageBox.StandardButton.Cancel:
            return
        if choice == QMessageBox.StandardButton.Yes:
            source, _ = QFileDialog.getOpenFileName(
                self, "选择插件 zip", "", "MTPro 插件 (*.zip)"
            )
        else:
            source = QFileDialog.getExistingDirectory(self, "选择插件文件夹")
        if not source:
            return

        source_path = Path(source).expanduser().resolve(strict=False)
        target_root = default_plugin_dir()
        target = target_root / source_path.name
        try:
            target_root.mkdir(parents=True, exist_ok=True)
            if target.exists():
                overwrite = QMessageBox.question(
                    self,
                    "目标已存在",
                    f"{target.name} 已存在，是否覆盖？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if overwrite != QMessageBox.StandardButton.Yes:
                    return
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            if source_path.is_dir():
                shutil.copytree(source_path, target)
            else:
                shutil.copy2(source_path, target)
        except OSError as exc:
            QMessageBox.critical(self, "安装失败", repr(exc))
            return
        self._rescan()

    def _rescan(self) -> None:
        self.report = load_all_plugins(state=self.app_state)
        self._refresh_report()
        QMessageBox.information(
            self,
            "扫描完成",
            f"成功 {len(self.report.loaded)} 个，失败 {len(self.report.failed)} 个。\n"
            "插件列表将在重启 MTPro 后更新。",
        )
