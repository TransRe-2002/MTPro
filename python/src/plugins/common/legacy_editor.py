"""Qt bridge that embeds slice-based editors in the plugin runtime."""
from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from plugin_api import TransformResult, default_params

class SlicePluginEditor(QWidget):
    """Connect a visual editor fed by a Pipeline slice to ``EditorHost``.

    The wrapped widget never selects a dataset or requests data.  It receives
    the current ``ChannelBlock`` slice and returns an edited slice; this bridge
    records the corresponding deterministic plugin parameters in Pipeline.
    """
    def __init__(
        self,
        host,
        block,
        params: Mapping,
        editor: QWidget,
        extract_params: Callable[[QWidget, object, object, Mapping], dict],
        parent=None,
    ):
        super().__init__(parent)
        self.host = host
        self.block = block
        self.params = dict(params)
        self.editor = editor
        self.extract_params = extract_params

        apply_saved_params = getattr(editor, "apply_plugin_params", None)
        if callable(apply_saved_params):
            apply_saved_params(self.params)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        info = getattr(getattr(host, "plugin", None), "info", None)
        defaults = default_params(info) if info is not None else {}
        internal = (
            {"_pipeline_step_range", "region", "effect_to_end"}
            if "_pipeline_step_range" in self.params else set()
        )
        if any(key not in internal and value != defaults.get(key)
               for key, value in self.params.items()):
            note = QLabel("已保存参数将在确认后由处理器确定性重放。", self)
            layout.addWidget(note)
        layout.addWidget(editor)
        editor.result_signal.connect(self._on_slice_result)

    def _on_slice_result(self, _channel, result) -> None:
        try:
            values = self.extract_params(self.editor, self.block, result, self.params)
            broadcast: set[str] = set()
            sync_checkbox = getattr(self.editor, "sync_checkbox", None)
            if sync_checkbox is not None and sync_checkbox.isChecked():
                broadcast.add("channel_sync")
            all_data_checkbox = getattr(self.editor, "all_data_checkbox", None)
            if all_data_checkbox is not None and all_data_checkbox.isChecked():
                broadcast.add("region_sync")
            accept_preview = getattr(self.host, "accept_preview", None)
            last = getattr(self.editor, "last_result", None)
            info = getattr(getattr(self.host, "plugin", None), "info", None)
            if (getattr(info, "plugin_id", "") == "remove_step_baseline"
                    and values.get("effect_to_end") and last is not None
                    and callable(accept_preview) and not broadcast):
                series = pd.Series(result, index=self.block.series.index, name=self.block.series.name)
                accept_preview(TransformResult(series, {
                    "mode": "preview", "step_height": last.step_height, "cost": last.cost,
                }), values)
                return
            self.host.apply_params(values, frozenset(broadcast))
        except Exception as exc:
            failed = getattr(self.host, "failed", None)
            if failed is not None:
                failed.emit(exc)
            else:
                self.host.log(repr(exc))


def slice_processor_editor(processor_class, host, block, params, extractor):
    editor = processor_class(block.series, block.channel)
    return SlicePluginEditor(host, block, params, editor, extractor)


def slice_standalone_editor(processor_class, host, block, params, extractor):
    editor = processor_class(block.series, block.channel)
    return SlicePluginEditor(host, block, params, editor, extractor)


# Kept as import aliases for external plugins during the transition.  New
# first-party plugins use the slice_* names above.
LegacyPluginEditor = SlicePluginEditor
base_processor_editor = slice_processor_editor
standalone_processor_editor = slice_standalone_editor


__all__ = [
    "SlicePluginEditor",
    "slice_processor_editor",
    "slice_standalone_editor",
    "LegacyPluginEditor",
    "base_processor_editor",
    "standalone_processor_editor",
]
