from __future__ import annotations

"""应用日志文件和标准流重定向配置，不依赖 Qt。"""

import io
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path


class StreamToLogger(io.TextIOBase):
    def __init__(
        self,
        logger: logging.Logger,
        level: int,
        original_stream: io.TextIOBase | None = None,
    ):
        super().__init__()
        self.logger = logger
        self.level = level
        self.original_stream = original_stream
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        if self.original_stream is not None:
            self.original_stream.write(text)
            self.original_stream.flush()
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
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO, original_stdout)
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR, original_stderr)
    return original_stdout, original_stderr


def restore_standard_streams(stdout: io.TextIOBase, stderr: io.TextIOBase) -> None:
    sys.stdout = stdout
    sys.stderr = stderr


def log_session_banner(logger: logging.Logger, log_path: Path) -> None:
    logger.info("=".ljust(72, "="))
    logger.info("MTPro session started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("log file: %s", log_path)

