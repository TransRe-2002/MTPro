"""Dialog for applying a plugin recipe to multiple open channels."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from base.batch_runner import BatchRunner, BatchTarget, BatchTargetResult
from core.recipe import load_recipe
from plugin_api import ChannelMeta
from ui.tools import registry as tool_registry
from utils.qt_bridge import FutureBridge


class BatchProcessDialog(QDialog):
    result_applied = Signal(int, str)

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self._bridges: list[FutureBridge] = []
        self._pending = 0
        self._completed = 0
        self.setWindowTitle("插件批处理")
        self.resize(900, 660)

        layout = QVBoxLayout(self)
        recipe_row = QHBoxLayout()
        recipe_row.addWidget(QLabel("Recipe"))
        self.recipe_path = QLineEdit(self)
        browse = QPushButton("选择…", self)
        browse.clicked.connect(self._browse_recipe)
        recipe_row.addWidget(self.recipe_path, 1)
        recipe_row.addWidget(browse)
        layout.addLayout(recipe_row)

        layout.addWidget(QLabel("目标数据集 / 通道"))
        self.target_tree = QTreeWidget(self)
        self.target_tree.setHeaderLabels(("目标", "样点数"))
        self.target_tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        layout.addWidget(self.target_tree, 1)
        self._populate_targets()

        run_row = QHBoxLayout()
        self.run_button = QPushButton("运行", self)
        self.run_button.clicked.connect(self._run)
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 1)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.progress, 1)
        layout.addLayout(run_row)

        self.results = QTableWidget(0, 4, self)
        self.results.setHorizontalHeaderLabels(("目标", "状态", "诊断", "错误"))
        self.results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_targets(self) -> None:
        self.target_tree.clear()
        for data_id, data in self.data_manager.items():
            parent = QTreeWidgetItem((f"#{data_id} {data.name}", ""))
            parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsAutoTristate | Qt.ItemFlag.ItemIsUserCheckable)
            parent.setCheckState(0, Qt.CheckState.Unchecked)
            self.target_tree.addTopLevelItem(parent)
            for channel in data.channel_names():
                series = data.get_channel_series(channel)
                item = QTreeWidgetItem((channel, str(len(series))))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Unchecked)
                item.setData(0, Qt.ItemDataRole.UserRole, (data_id, channel))
                parent.addChild(item)
            parent.setExpanded(True)

    def _browse_recipe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择处理 Recipe", "", "MTPro Recipe (*.toml)"
        )
        if path:
            self.recipe_path.setText(path)

    def _selected_targets(self) -> list[BatchTarget]:
        targets: list[BatchTarget] = []
        for root_index in range(self.target_tree.topLevelItemCount()):
            parent = self.target_tree.topLevelItem(root_index)
            for child_index in range(parent.childCount()):
                item = parent.child(child_index)
                if item.checkState(0) != Qt.CheckState.Checked:
                    continue
                data_id, channel = item.data(0, Qt.ItemDataRole.UserRole)
                data = self.data_manager.get(data_id)
                if data is None or not data.has_channel(channel):
                    continue
                series = data.get_channel_series(channel)
                if len(series.index) > 1:
                    dt = float((series.index[1] - series.index[0]).total_seconds())
                else:
                    dt = 0.0
                targets.append(
                    BatchTarget(
                        data_id,
                        channel,
                        series,
                        ChannelMeta(sample_rate=(1.0 / dt if dt > 0 else 0.0)),
                    )
                )
        return targets

    def _run(self) -> None:
        path = self.recipe_path.text().strip()
        if not path:
            QMessageBox.warning(self, "缺少 Recipe", "请先选择 Recipe TOML 文件。")
            return
        try:
            recipe = load_recipe(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Recipe 无效", repr(exc))
            return
        targets = self._selected_targets()
        if not targets:
            QMessageBox.warning(self, "缺少目标", "请至少勾选一个通道。")
            return
        plugins = {
            spec.tool_id: tool_registry.get_plugin(spec.tool_id)
            for spec in tool_registry.all_specs()
            if tool_registry.get_plugin(spec.tool_id) is not None
        }
        futures = BatchRunner(plugins).submit(recipe, targets)
        self.results.setRowCount(0)
        self._bridges.clear()
        self._pending = len(futures)
        self._completed = 0
        self.progress.setRange(0, self._pending)
        self.progress.setValue(0)
        self.run_button.setEnabled(False)
        for future in futures.values():
            bridge = FutureBridge(self)
            bridge.done.connect(self._on_result)
            bridge.failed.connect(self._on_future_error)
            bridge.attach(future)
            self._bridges.append(bridge)

    def _on_result(self, result: BatchTargetResult) -> None:
        if result.status == "ok":
            data = self.data_manager.get(result.target.data_id)
            if data is not None:
                data.set_channel_series(result.target.channel, result.series)
                self.data_manager.notify_changed(result.target.data_id, structural=False)
                self.result_applied.emit(result.target.data_id, result.target.channel)
        diagnostics = "; ".join(str(value) for value in result.diagnostics)
        if result.skipped_steps:
            diagnostics = "; ".join((diagnostics, *result.skipped_steps)).strip("; ")
        self._append_result(
            f"#{result.target.data_id} {result.target.channel}",
            result.status,
            diagnostics,
            result.error,
        )
        self._advance()

    def _on_future_error(self, exc: object) -> None:
        self._append_result("unknown", "error", "", repr(exc))
        self._advance()

    def _append_result(self, target: str, status: str, diagnostics: str, error: str) -> None:
        row = self.results.rowCount()
        self.results.insertRow(row)
        for column, value in enumerate((target, status, diagnostics, error)):
            self.results.setItem(row, column, QTableWidgetItem(value))

    def _advance(self) -> None:
        self._completed += 1
        self.progress.setValue(self._completed)
        if self._completed >= self._pending:
            self.run_button.setEnabled(True)
