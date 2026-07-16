from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class BatchSaveDialog(QDialog):
    """批量保存对话框：文件夹选择 + 后缀 + 预览。"""

    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量保存")
        self.resize(550, 350)

        layout = QVBoxLayout(self)

        # 文件夹选择
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("目标文件夹:"))
        self.folder_path = QLineEdit()
        self.folder_path.setReadOnly(True)
        folder_row.addWidget(self.folder_path, 1)
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse)
        folder_row.addWidget(browse_btn)
        layout.addLayout(folder_row)

        # 后缀
        suffix_row = QHBoxLayout()
        suffix_row.addWidget(QLabel("文件名后缀:"))
        self.suffix_edit = QLineEdit()
        self.suffix_edit.setPlaceholderText("如 _fixed")
        suffix_row.addWidget(self.suffix_edit, 1)
        layout.addLayout(suffix_row)

        # 预览
        layout.addWidget(QLabel("保存预览（最多显示前 5 个）:"))
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(120)
        layout.addWidget(self.preview)

        # 按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        save_btn = QPushButton("批量保存")
        save_btn.clicked.connect(self.accept)
        save_btn.setDefault(True)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        self._items = items
        self._update_preview()

        self.suffix_edit.textChanged.connect(self._update_preview)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(
            self, "选择批量保存的目标文件夹",
            options=QFileDialog.Option.ShowDirsOnly,
        )
        if folder:
            self.folder_path.setText(folder)
            self._update_preview()

    def _update_preview(self):
        folder = self.folder_path.text() or "<未选择>"
        suffix = self.suffix_edit.text().strip()
        lines = []
        for i, (_, data) in enumerate(self._items):
            if i >= 5:
                lines.append(f"  ... 共 {len(self._items)} 个文件")
                break
            name = data.name if data.name else f"dataset_{i}"
            lines.append(f"  {name}{suffix}.mat")
        self.preview.setPlainText(f"目标: {folder}\n\n" + "\n".join(lines))