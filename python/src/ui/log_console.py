from __future__ import annotations

import io
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
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
        self.text_edit.setFont(QFont("Courier New", 10))
        self.text_edit.setAcceptRichText(True)

        self.title_label = QLabel("运行日志", self)
        self.clear_button = QPushButton("清空", self)
        self.copy_button = QPushButton("复制", self)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 2)
        controls.addWidget(self.title_label)
        controls.addStretch(1)
        controls.addWidget(self.copy_button)
        controls.addWidget(self.clear_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(controls)
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


class StreamToLogger(io.TextIOBase):
    def __init__(self, logger: logging.Logger, level: int):
        super().__init__()
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.logger.log(self.level, line)
        return len(text)

    def flush(self) -> None:
        line = self._buffer.strip()
        if line:
            self.logger.log(self.level, line)
        self._buffer = ""


_LOG_PATH: Path | None = None


def configure_application_logging(console_handler: logging.Handler | None = None) -> Path:
    global _LOG_PATH

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if _LOG_PATH is None:
        log_dir = Path(tempfile.gettempdir()) / "mtpro_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _LOG_PATH = log_dir / "mtpro.log"

    file_handler_exists = any(isinstance(handler, logging.FileHandler) for handler in root_logger.handlers)
    if not file_handler_exists:
        file_handler = logging.FileHandler(_LOG_PATH, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root_logger.addHandler(file_handler)

    if console_handler is not None and console_handler not in root_logger.handlers:
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        root_logger.addHandler(console_handler)

    return _LOG_PATH


def install_standard_stream_logging() -> tuple[io.TextIOBase, io.TextIOBase]:
    stdout_logger = logging.getLogger("stdout")
    stderr_logger = logging.getLogger("stderr")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)
    return original_stdout, original_stderr


def restore_standard_streams(stdout: io.TextIOBase, stderr: io.TextIOBase) -> None:
    sys.stdout = stdout
    sys.stderr = stderr


def log_session_banner(logger: logging.Logger, log_path: Path) -> None:
    logger.info("=".ljust(72, "="))
    logger.info("MTPro session started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("log file: %s", log_path)
