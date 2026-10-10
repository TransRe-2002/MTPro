"""Schema-driven plugin parameter dialog and shared Pipeline controls."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from plugin_api import ParamSpec, PluginInfo, complete_params


PIPELINE_LINK_ALL_TEXT = "应用到当前 Pipeline 的全部通道"


def create_pipeline_link_all_checkbox(
    parent: QWidget,
    *,
    checked: bool = False,
) -> QCheckBox:
    """Build the shared all-channel control used by every Pipeline editor."""
    checkbox = QCheckBox(PIPELINE_LINK_ALL_TEXT, parent)
    checkbox.setChecked(checked)
    checkbox.setToolTip(
        "各通道独立执行和缓存；修改公共设置时整组重新计算。"
    )
    return checkbox


class PluginParamDialog(QDialog):
    """Automatic editor for plugins that only declare ``ParamSpec`` fields."""

    def __init__(
        self,
        info: PluginInfo,
        params: Mapping[str, Any] | None = None,
        parent=None,
        *,
        allow_link_all: bool = False,
        link_all: bool = False,
    ):
        super().__init__(parent)
        self.info = info
        self.params: dict[str, Any] = {}
        self._editors: dict[str, tuple[QWidget, ParamSpec]] = {}
        current = complete_params(info, params or {})
        self.setWindowTitle(info.display_name)

        layout = QVBoxLayout(self)
        if info.description:
            description = QLabel(info.description, self)
            description.setWordWrap(True)
            layout.addWidget(description)
        form = QFormLayout()
        unsupported: list[str] = []
        for spec in info.params_schema:
            editor = self._create_field(spec, current.get(spec.name, spec.default))
            if editor is None:
                unsupported.append(spec.label)
                continue
            form.addRow(spec.label, editor)
            self._editors[spec.name] = (editor, spec)
        layout.addLayout(form)
        if unsupported:
            warning = QLabel(
                "以下 list 参数需要插件提供自定义编辑器：" + "、".join(unsupported),
                self,
            )
            warning.setWordWrap(True)
            layout.addWidget(warning)

        self.link_all_checkbox: QCheckBox | None = None
        if allow_link_all:
            self.link_all_checkbox = create_pipeline_link_all_checkbox(
                self,
                checked=link_all,
            )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        action_row = QHBoxLayout()
        action_row.addStretch(1)
        if self.link_all_checkbox is not None:
            action_row.addWidget(self.link_all_checkbox)
        action_row.addWidget(buttons)
        layout.addLayout(action_row)

    def _create_field(self, spec: ParamSpec, value: Any) -> QWidget | None:
        if spec.type is float:
            editor = QDoubleSpinBox(self)
            editor.setDecimals(8)
            editor.setRange(
                float(spec.minimum) if spec.minimum is not None else -1.0e100,
                float(spec.maximum) if spec.maximum is not None else 1.0e100,
            )
            editor.setValue(float(value))
            return editor
        if spec.type is int:
            editor = QSpinBox(self)
            editor.setRange(
                int(spec.minimum) if spec.minimum is not None else -2147483648,
                int(spec.maximum) if spec.maximum is not None else 2147483647,
            )
            editor.setValue(int(value))
            return editor
        if spec.type is bool:
            editor = QCheckBox(self)
            editor.setChecked(bool(value))
            return editor
        if spec.type is str and spec.options:
            editor = QComboBox(self)
            editor.addItems(spec.options)
            index = editor.findText(str(value))
            editor.setCurrentIndex(max(0, index))
            return editor
        if spec.type is str:
            editor = QLineEdit(str(value), self)
            return editor
        return None

    def accept(self) -> None:
        values = complete_params(self.info, {})
        for name, (editor, spec) in self._editors.items():
            if spec.type is float:
                values[name] = editor.value()
            elif spec.type is int:
                values[name] = editor.value()
            elif spec.type is bool:
                values[name] = editor.isChecked()
            elif spec.type is str and spec.options:
                values[name] = editor.currentText()
            elif spec.type is str:
                values[name] = editor.text()
        try:
            json.dumps(values)
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return
        self.params = values
        super().accept()
