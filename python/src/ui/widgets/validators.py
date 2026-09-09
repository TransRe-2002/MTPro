from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QLineEdit


def read_positive_float(edit: QLineEdit, label: str, log: Callable[[str], None]) -> float | None:
    text = edit.text().strip()
    if not text:
        log(f"{label} 不能为空")
        return None
    value = float(text)
    if value <= 0.0:
        log(f"{label} 必须大于 0")
        return None
    return value


def read_nonnegative_int(edit: QLineEdit, label: str, log: Callable[[str], None]) -> int | None:
    text = edit.text().strip()
    if not text:
        log(f"{label} 不能为空")
        return None
    value = int(text)
    if value < 0:
        log(f"{label} 必须大于等于 0")
        return None
    return value


def read_positive_int(edit: QLineEdit, label: str, log: Callable[[str], None]) -> int | None:
    value = read_nonnegative_int(edit, label, log)
    if value is None:
        return None
    if value <= 0:
        log(f"{label} 必须大于 0")
        return None
    return value