from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QFont, QPalette
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QFrame,
)


class _LogEmitter(QObject):
    message_ready = Signal(str, str)


class QtTextEditHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.emitter = _LogEmitter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return
        self.emitter.message_ready.emit(record.levelname, message)


class LogConsoleWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(0)
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.document().setMaximumBlockCount(3000)
        self.text_edit.setFont(QFont("Courier New", 9))
        self.text_edit.setAcceptRichText(True)
        self.text_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

        self.title_label = QLabel("命令行窗口 / Command Window", self)
        self.title_label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        self.clear_button = QPushButton("清空", self)
        self.clear_button.setFixedWidth(52)
        self.clear_button.setFixedHeight(22)
        self.clear_button.setFont(QFont("Arial", 9))
        self.copy_button = QPushButton("复制", self)
        self.copy_button.setFixedWidth(52)
        self.copy_button.setFixedHeight(22)
        self.copy_button.setFont(QFont("Arial", 9))

        # 标题分隔线
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)

        controls = QHBoxLayout()
        controls.setContentsMargins(2, 2, 2, 0)
        controls.setSpacing(4)
        controls.addWidget(self.title_label)
        controls.addStretch(1)
        controls.addWidget(self.copy_button)
        controls.addWidget(self.clear_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 4)
        layout.setSpacing(2)
        layout.addLayout(controls)
        layout.addWidget(sep)
        layout.addWidget(self.text_edit, 1)

        self.clear_button.clicked.connect(self.text_edit.clear)
        self.copy_button.clicked.connect(self.text_edit.selectAll)
        self.copy_button.clicked.connect(self.text_edit.copy)

    def preferred_height(self) -> int:
        return 120

    def append_log(self, level_name: str, message: str) -> None:
        base_color = self.text_edit.palette().color(QPalette.ColorRole.Text)
        is_dark = self.text_edit.palette().color(QPalette.ColorRole.Base).lightness() < 128

        if is_dark:
            color = {
                "DEBUG": "#9aa4b2",
                "INFO": base_color.name(),
                "WARNING": "#f2c14e",
                "ERROR": "#ff7b72",
                "CRITICAL": "#ff4d6d",
            }.get(level_name, base_color.name())
        else:
            color = {
                "DEBUG": "#5a5a5a",
                "INFO": base_color.name(),
                "WARNING": "#8a4f00",
                "ERROR": "#a31919",
                "CRITICAL": "#7a0015",
            }.get(level_name, base_color.name())
        html = (
            f'<span style="color:{color}; white-space: pre;">'
            f"{message.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')}"
            "</span>"
        )
        self.text_edit.append(html)
