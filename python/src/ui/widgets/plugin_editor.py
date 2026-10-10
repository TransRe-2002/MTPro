"""Qt editor hosts shared by first- and third-party processing plugins."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from base.plugin_runtime import run_plugin_process
from base.qt_bridge import run as run_in_pool
from plugin_api import (
    ChannelBlock,
    ChannelMeta,
    TransformResult,
    complete_params,
)
from ui.dialogs.plugin_parameters import PluginParamDialog

logger = logging.getLogger(__name__)


class PluginEditorHost(QWidget):
    """Main-thread implementation of the plugin ``EditorHost`` protocol."""

    result_ready = Signal(object, object, object)  # TransformResult, params, broadcast
    failed = Signal(object)

    def __init__(self, plugin: object, block: ChannelBlock, parent=None):
        super().__init__(parent)
        self.plugin = plugin
        self.block = block
        self._future = None
        self._pending_broadcast = frozenset()

    def _relay_process_done(self, payload) -> None:
        self.result_ready.emit(payload[0], payload[1], self._pending_broadcast)

    def _relay_process_failed(self, exc) -> None:
        self.failed.emit(exc)

    def apply_params(
        self,
        params: Mapping[str, Any],
        broadcast: frozenset[str] = frozenset(),
    ) -> None:
        unsupported = set(broadcast) - set(self.plugin.info.capabilities)
        if unsupported:
            self.failed.emit(ValueError(f"plugin requested undeclared capabilities: {unsupported}"))
            return
        try:
            values = complete_params(self.plugin.info, params)
            json.dumps(values)
        except (TypeError, ValueError) as exc:
            self.failed.emit(exc)
            return
        self._pending_broadcast = frozenset(broadcast)
        self._future = run_in_pool(
            "compute",
            run_plugin_process,
            self.plugin,
            self.block,
            values,
            on_done=self._relay_process_done,
            on_error=self._relay_process_failed,
            owner=self,
        )

    def run_heavy(
        self,
        compute_fn: Callable[[], Any],
        apply_fn: Callable[[Any], None],
        description: str = "",
    ) -> None:
        if description:
            self.log(description)
        run_in_pool(
            "compute",
            compute_fn,
            on_done=apply_fn,
            on_error=self.failed.emit,
            owner=self,
        )

    def log(self, message: str) -> None:
        logger.info("%s: %s", self.plugin.info.plugin_id, message)


class PluginSliceEditor(QWidget):
    """Generic slice editor that exposes plugin and result signals."""

    data_requested = Signal(int, str)
    result_submitted = Signal(int, str, object)
    result_signal = Signal(str, object)
    params_applied = Signal(object, object)
    channel_sync_requested = Signal(int, str, object)
    region_sync_requested = Signal(object)

    def __init__(
        self,
        plugin: object,
        data_id: int,
        channel_name: str,
        parent=None,
        params: Mapping[str, Any] | None = None,
    ):
        super().__init__(parent)
        self.plugin = plugin
        self.data_id = data_id
        self.ch_name = channel_name
        self.params = complete_params(plugin.info, params or {})
        self._host: PluginEditorHost | None = None
        self._editor: QWidget | None = None
        self._loaded = False
        self._layout = QVBoxLayout(self)
        self._placeholder = QLabel("正在加载数据…", self)
        self._layout.addWidget(self._placeholder)

    def wire_data_ready(self, data_ready_signal) -> None:
        data_ready_signal.connect(self._on_data_ready)

    def load_series(self, channel_name: str, series: pd.Series) -> None:
        from plugins.common.base_processor import build_pkg_from_series

        self.ch_name = channel_name
        self._on_data_ready(self.data_id, channel_name, build_pkg_from_series(channel_name, series))

    def _on_data_ready(self, data_id: int, channel: str, pkg: dict) -> None:
        if self._loaded or data_id != self.data_id or channel != self.ch_name:
            return
        self._loaded = True
        raw = np.asarray(pkg["y"], dtype=np.float64)
        index = pkg.get("datetime_index")
        if index is None:
            start = pd.Timestamp.fromtimestamp(float(pkg.get("start_time", 0.0)))
            index = pd.date_range(start, periods=len(raw), freq=pd.to_timedelta(float(pkg.get("dt", 1.0)), unit="s"))
        series = pd.Series(raw.copy(), index=index, name=channel)
        dt = float(pkg.get("dt", 0.0))
        block = ChannelBlock(
            channel,
            series,
            ChannelMeta(sample_rate=(1.0 / dt if dt > 0 else 0.0)),
        )
        self._host = PluginEditorHost(self.plugin, block, self)
        self._host.result_ready.connect(self._on_result_ready)
        self._host.failed.connect(self._on_failed)
        self._layout.removeWidget(self._placeholder)
        self._placeholder.deleteLater()

        create_editor = getattr(self.plugin, "create_editor", None)
        if callable(create_editor):
            try:
                self._editor = create_editor(self._host, block, self.params)
            except Exception as exc:
                self._on_failed(exc)
                return
        elif bool(getattr(self.plugin, "pipeline_preview", False)):
            from ui.widgets.filter_preview_editor import FilterPreviewEditor

            self._editor = FilterPreviewEditor(
                self._host,
                block,
                self.params,
                self,
            )
        if self._editor is not None:
            self._layout.addWidget(self._editor)
        else:
            QTimer.singleShot(0, self._open_param_dialog)

    def _open_param_dialog(self) -> None:
        dialog = PluginParamDialog(self.plugin.info, self.params, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and self._host is not None:
            self._host.apply_params(dialog.params)

    def _on_result_ready(
        self,
        result: TransformResult,
        params: Mapping[str, Any],
        broadcast: frozenset[str],
    ) -> None:
        self.params = dict(params)
        values = result.series.to_numpy(dtype=np.float64, copy=True)
        self.params_applied.emit(dict(params), frozenset(broadcast))
        if "channel_sync" in broadcast:
            self.channel_sync_requested.emit(self.data_id, self.ch_name, values)
        if "region_sync" in broadcast:
            sync_params = dict(params)
            sync_params.setdefault("source_data_id", self.data_id)
            self.region_sync_requested.emit(sync_params)
        self.result_submitted.emit(self.data_id, self.ch_name, values)
        self.result_signal.emit(self.ch_name, values)

    def _on_failed(self, exc: object) -> None:
        logger.error("Plugin %s failed: %r", self.plugin.info.plugin_id, exc)
        QMessageBox.critical(self, "插件处理失败", repr(exc))

    def attach_pipeline_link_checkbox(self, checkbox: QCheckBox) -> bool:
        """Place the Pipeline group control beside the editor's final action."""
        roots: list[QWidget] = []
        current = self._editor
        while isinstance(current, QWidget) and current not in roots:
            roots.append(current)
            nested = getattr(current, "editor", None)
            current = nested if isinstance(nested, QWidget) else None

        for root in roots:
            legacy_sync = getattr(root, "sync_checkbox", None)
            if isinstance(legacy_sync, QCheckBox):
                legacy_sync.hide()

        action_button = None
        for name in ("confirm_btn", "confirm_button", "apply_button", "apply_btn"):
            for root in roots:
                candidate = getattr(root, name, None)
                if isinstance(candidate, QWidget):
                    action_button = candidate
                    break
            if action_button is not None:
                break
        if action_button is None:
            return False

        located = self._find_widget_layout(self, action_button)
        if located is None:
            return False
        container, index = located
        if isinstance(container, QHBoxLayout):
            container.insertWidget(index, checkbox)
            return True
        if isinstance(container, QVBoxLayout):
            container.removeWidget(action_button)
            row = QHBoxLayout()
            row.addStretch(1)
            row.addWidget(checkbox)
            row.addWidget(action_button)
            container.insertLayout(index, row)
            return True
        return False

    @classmethod
    def _find_widget_layout(cls, container: QWidget, target: QWidget):
        root_layout = container.layout()
        if root_layout is None:
            return None

        def walk(layout):
            for index in range(layout.count()):
                item = layout.itemAt(index)
                if item.widget() is target:
                    return layout, index
                child_layout = item.layout()
                if child_layout is not None:
                    found = walk(child_layout)
                    if found is not None:
                        return found
                child_widget = item.widget()
                if child_widget is not None and child_widget.layout() is not None:
                    found = walk(child_widget.layout())
                    if found is not None:
                        return found
            return None

        return walk(root_layout)


def create_pipeline_plugin_editor(
    plugin: object,
    data_id: int,
    channel_name: str,
) -> PluginSliceEditor:
    return PluginSliceEditor(plugin, data_id, channel_name)
