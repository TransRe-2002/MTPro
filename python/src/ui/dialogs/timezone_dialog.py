"""无时区文件的时区选择对话框。

MAT 格式的 tsStruct 中时间字段（startTime / endTime / zeroTime）
仅存储为 [year, month, day, hour, minute, second] 数组，不包含时区信息。
用户在打开文件后必须先指定时区，才能正确解析时间戳。
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytz
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QComboBox,
    QVBoxLayout,
    QWidget,
    QLineEdit,
)

# MT 数据采集常见时区，按使用频率排序
_COMMON_TIMEZONES: list[str] = [
    "Asia/Shanghai",
    "Asia/Urumqi",
    "Asia/Tokyo",
    "Asia/Kolkata",
    "Asia/Bangkok",
    "UTC",
    "Europe/London",
    "Europe/Paris",
    "Europe/Moscow",
    "America/Denver",
    "America/Los_Angeles",
    "America/New_York",
    "America/Chicago",
    "Pacific/Auckland",
    "Australia/Sydney",
]


def _format_utc_offset(tz_name: str) -> str:
    """返回格式化的 UTC 偏移字符串，如 ``"(UTC+08:00)"``。"""
    tz = pytz.timezone(tz_name)
    # 使用当前时刻的偏移量（考虑夏令时）
    offset = tz.utcoffset(datetime.datetime.now())
    if offset is None:
        return ""
    total = int(offset.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    hours, remainder = divmod(total, 3600)
    minutes = remainder // 60
    return f"(UTC{sign}{hours:02d}:{minutes:02d})"


def _build_display_items() -> tuple[list[str], dict[str, str]]:
    """构建下拉列表显示文本和显示文本→时区名的映射。"""
    items: list[str] = []
    mapping: dict[str, str] = {}
    for tz_name in _COMMON_TIMEZONES:
        offset = _format_utc_offset(tz_name)
        display = f"{tz_name}  {offset}" if offset else tz_name
        items.append(display)
        mapping[display] = tz_name
    return items, mapping


class TimezoneDialog(QDialog):
    """弹窗让用户选择 MAT 文件内时间数据的时区。

    用法::

        dialog = TimezoneDialog("/path/to/data.mat", parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            tz_str = dialog.selected_timezone()
    """

    def __init__(self, filepath: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("时区设置 — MTPro")
        self.setMinimumWidth(520)

        self._filepath = filepath
        self._display_items, self._display_to_tz = _build_display_items()

        # ── 主布局 ──────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        # 说明文字
        info_label = QLabel(self)
        info_label.setWordWrap(True)
        suffix = Path(filepath).suffix.lower()
        format_name = "TXT" if suffix == ".txt" else "MAT"
        unit_note = (
            "\nTXT 不声明物理单位，导入后按磁场 nT、电场 mV/km 解释。"
            if suffix == ".txt"
            else ""
        )
        info_label.setText(
            f"{format_name} 文件中存储的时间数据不包含时区信息。\n"
            f"请选择该数据采集站点所在的时区，以便正确解析时间坐标。{unit_note}"
        )
        layout.addWidget(info_label)

        # 文件名显示
        file_label = QLabel(self)
        file_label.setWordWrap(True)
        file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        file_label.setStyleSheet("QLabel { color: #4070a0; font-weight: bold; }")
        file_label.setText(Path(filepath).name)
        layout.addWidget(file_label)

        # ── 时区选择 ────────────────────────────────────────────
        form = QFormLayout()
        form.setSpacing(6)

        self._combo = QComboBox(self)
        self._combo.setEditable(True)
        self._combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._combo.addItems(self._display_items)
        # 默认选中 Asia/Shanghai
        default_display = next(
            (d for d, t in self._display_to_tz.items() if t == "Asia/Shanghai"),
            self._display_items[0],
        )
        self._combo.setCurrentText(default_display)

        line_edit = self._combo.lineEdit()
        if line_edit is not None:
            line_edit.setPlaceholderText("输入时区名称，例如 Asia/Shanghai")
            line_edit.setClearButtonEnabled(True)

        self._combo.setToolTip(
            "选择或输入 pytz 时区名称。\n"
            "格式：Continent/City，例如 Asia/Shanghai、America/Denver"
        )
        form.addRow("时区：", self._combo)

        layout.addLayout(form)

        # ── 错误提示 ────────────────────────────────────────────
        self._error_label = QLabel(self)
        self._error_label.setStyleSheet("QLabel { color: red; font-size: 11px; }")
        self._error_label.setWordWrap(True)
        self._error_label.hide()
        layout.addWidget(self._error_label)

        layout.addStretch(1)

        # ── 按钮 ────────────────────────────────────────────────
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

    def selected_timezone(self) -> str:
        """返回用户选择的时区字符串（如 ``"Asia/Shanghai"`` 或 ``"UTC"``）。

        如果用户从下拉列表中选择，返回原始时区名（去除 UTC 偏移后缀）；
        如果用户手动输入，直接返回输入文本。
        """
        text = self._combo.currentText().strip()
        # 优先查映射表（下拉选择项）
        return self._display_to_tz.get(text, text)

    # ------------------------------------------------------------------
    # 私有槽
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        """校验用户输入的时区是否有效。"""
        tz_str = self.selected_timezone()

        if not tz_str:
            self._error_label.setText("请选择或输入一个时区。")
            self._error_label.show()
            self._combo.setFocus()
            return

        try:
            pytz.timezone(tz_str)
        except pytz.exceptions.UnknownTimeZoneError:
            self._error_label.setText(
                f"\"{tz_str}\" 不是有效的时区名称。\n"
                "请使用 Continent/City 格式，例如 Asia/Shanghai、America/Denver。"
            )
            self._error_label.show()
            self._combo.setFocus()
            return

        self._error_label.hide()
        self.accept()
