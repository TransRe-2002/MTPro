from __future__ import annotations

from PySide6.QtWidgets import QInputDialog, QLineEdit, QWidget

from core.impedance_result import default_impedance_result_name


def prompt_result_name(parent: QWidget, payload: dict) -> str | None:
    """Ask for a result name; cancelling leaves the calculation unsaved."""
    method = str(payload.get('method') or 'robust estimate')
    default = default_impedance_result_name(payload, method)
    name, accepted = QInputDialog.getText(
        parent, '保存阻抗结果', '结果名称：', QLineEdit.EchoMode.Normal, default,
    )
    return (name.strip() or default) if accepted else None
