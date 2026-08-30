"""单通道重命名对话框（通道级右键菜单）。

通道名前缀决定场类型分类（E* → 电场，B*/H* → 磁场），
重命名会同步改变分类与波形视图中的通道排序。
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from core.em_data import EMData


class RenameChannelDialog(QDialog):
    """重命名单个通道。结构性变更。"""

    def __init__(
        self,
        em_data: EMData,
        old_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("重命名通道 — MTPro")
        self.setMinimumWidth(420)

        self._em_data = em_data
        self._old_name = old_name

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        info_label = QLabel(
            "通道名前缀决定场类型：E* 为电场，B*/H* 为磁场。\n"
            "改名会同时更新波形视图的通道排序与保存文件。",
            self,
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        form = QFormLayout()
        form.setSpacing(6)

        self._name_edit = QLineEdit(self)
        self._name_edit.setText(old_name)
        self._name_edit.selectAll()
        form.addRow("新通道名：", self._name_edit)

        layout.addLayout(form)

        self._error_label = QLabel(self)
        self._error_label.setStyleSheet("QLabel { color: red; font-size: 11px; }")
        self._error_label.setWordWrap(True)
        self._error_label.hide()
        layout.addWidget(self._error_label)

        layout.addStretch(1)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def new_name(self) -> str:
        return self._name_edit.text().strip()

    # ------------------------------------------------------------------
    # 私有槽
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        new_name = self.new_name()

        if not new_name:
            self._show_error("通道名不能为空。")
            return
        if new_name == self._old_name:
            self.accept()
            return
        if self._em_data.has_channel(new_name):
            self._show_error(f"通道 {new_name!r} 已存在。")
            return

        self._error_label.hide()
        self.accept()

    def _show_error(self, message: str) -> None:
        self._error_label.setText(message)
        self._error_label.show()
