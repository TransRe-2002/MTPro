from __future__ import annotations

"""Pipeline Block 的通用参数编辑对话框。"""

from collections.abc import Mapping

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from core.pipeline_model import PipelineBlock


class PipelineParamDialog(QDialog):
    def __init__(
        self,
        channel_name: str,
        layer_label: str,
        block: PipelineBlock,
        tool_meta: Mapping[str, Mapping[str, object]],
        parent=None,
    ):
        super().__init__(parent)
        meta = tool_meta.get(block.tool_id, {})
        self.params: dict = {}
        self._fields: dict[str, tuple[QLineEdit, type]] = {}
        self.setWindowTitle(meta.get("name", block.tool_id))

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("channel", QLabel(channel_name))
        form.addRow("layer", QLabel(layer_label))
        form.addRow("tool", QLabel(str(meta.get("name", block.tool_id))))
        form.addRow("range", QLabel(f"{block.start} ~ {block.end}"))

        for key, label, parser, default in meta.get("fields", []):
            edit = QLineEdit(str(block.params.get(key, default)), self)
            form.addRow(label, edit)
            self._fields[key] = (edit, parser)

        layout.addLayout(form)

        tip = QLabel(
            "这类 block 先保存参数，不直接执行。\n"
            "后续接入真实 processor 后再参与 preview/commit。"
        )
        tip.setWordWrap(True)
        layout.addWidget(tip)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self) -> None:
        values: dict[str, object] = {}
        for key, (edit, parser) in self._fields.items():
            text = edit.text().strip()
            if not text:
                QMessageBox.warning(self, "参数错误", f"{key} 不能为空")
                return
            try:
                values[key] = parser(text)
            except ValueError:
                QMessageBox.warning(self, "参数错误", f"{key} 的值无效: {text}")
                return

        self.params = values
        super().accept()
