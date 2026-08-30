"""台站元数据编辑对话框（数据管理器就地预处理）。

编辑数据集的名称、经纬度、高程与 E/B 场单位；修改经
``EMData.set_metadata`` 写入，保存时随 ``update_meta`` 回写 MAT。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from core.em_data import EMData

# "未提供"哨兵值：spin 的最小值 + specialValueText 显示为空
_MISSING = -99999.0


def _spin(minimum: float, maximum: float, decimals: int = 6) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setMinimumWidth(160)
    return box


def _fill_optional(box: QDoubleSpinBox, value) -> None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        box.setValue(_MISSING)
    else:
        box.setValue(float(value))


def _read_optional(box: QDoubleSpinBox):
    value = box.value()
    return None if value == _MISSING else value


class StationMetadataDialog(QDialog):
    """编辑台站标量元数据。非结构性变更。"""

    def __init__(self, em_data: EMData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑台站元数据 — MTPro")
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        info_label = QLabel(
            "修改将应用于当前数据集并随保存写回文件。留空的坐标/高程保持原值。",
            self,
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        form = QFormLayout()
        form.setSpacing(6)

        self._name_edit = QLineEdit(self)
        self._name_edit.setText(em_data.name or "")
        form.addRow("数据名称：", self._name_edit)

        self._lat_spin = _spin(_MISSING, 90.0)
        self._lat_spin.setSpecialValueText("")
        _fill_optional(self._lat_spin, em_data.latitude)
        form.addRow("纬度：", self._lat_spin)

        self._lon_spin = _spin(_MISSING, 180.0)
        self._lon_spin.setSpecialValueText("")
        _fill_optional(self._lon_spin, em_data.longitude)
        form.addRow("经度：", self._lon_spin)

        self._elev_spin = _spin(_MISSING, 99999.0, decimals=2)
        self._elev_spin.setSpecialValueText("")
        _fill_optional(self._elev_spin, em_data.elevation)
        form.addRow("高程 (m)：", self._elev_spin)

        self._e_units_edit = QLineEdit(self)
        self._e_units_edit.setText(em_data.e_units or "")
        self._e_units_edit.setPlaceholderText("如 mV/km")
        form.addRow("电场单位：", self._e_units_edit)

        self._m_units_edit = QLineEdit(self)
        self._m_units_edit.setText(em_data.m_units or "")
        self._m_units_edit.setPlaceholderText("如 nT")
        form.addRow("磁场单位：", self._m_units_edit)

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

    def apply_to(self, em_data: EMData) -> None:
        """把对话框中的值写入 em_data（None 表示保持原值）。"""
        em_data.set_metadata(
            name=self._name_edit.text().strip(),
            latitude=_read_optional(self._lat_spin),
            longitude=_read_optional(self._lon_spin),
            elevation=_read_optional(self._elev_spin),
            e_units=self._e_units_edit.text().strip(),
            m_units=self._m_units_edit.text().strip(),
        )

    # ------------------------------------------------------------------
    # 私有槽
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        if not self._name_edit.text().strip():
            self._show_error("数据名称不能为空。")
            self._name_edit.setFocus()
            return

        lat = _read_optional(self._lat_spin)
        if lat is not None and not (-90.0 <= lat <= 90.0):
            self._show_error("纬度必须在 -90 到 90 之间。")
            return
        lon = _read_optional(self._lon_spin)
        if lon is not None and not (-180.0 <= lon <= 180.0):
            self._show_error("经度必须在 -180 到 180 之间。")
            return

        self._error_label.hide()
        self.accept()

    def _show_error(self, message: str) -> None:
        self._error_label.setText(message)
        self._error_label.show()
