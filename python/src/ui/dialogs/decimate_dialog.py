"""降采样对话框（数据管理器就地预处理）。

选择整数倍抽取因子，实时预览降采样后的采样周期/点数/时长与
内存估计。确认后由 ``EMData.decimate`` 就地执行（不可撤销）。
"""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.em_data import EMData
from utils.limits import MAX_POINTS_PER_CHANNEL, format_points, suggested_decimate_factor
from utils.signal_decimate import decimated_length


def format_duration(delta: pd.Timedelta) -> str:
    """把时长格式化为紧凑的中文描述，如 ``3.2 天`` / ``5 小时``。"""
    total_seconds = delta.total_seconds()
    if total_seconds >= 86400:
        return f"{total_seconds / 86400:.2f} 天"
    if total_seconds >= 3600:
        return f"{total_seconds / 3600:.2f} 小时"
    if total_seconds >= 60:
        return f"{total_seconds / 60:.2f} 分钟"
    return f"{total_seconds:.0f} 秒"


def format_sample_rate(dt: pd.Timedelta) -> str:
    """采样周期 → 采样率描述。"""
    seconds = dt.total_seconds()
    if seconds <= 0:
        return "未知"
    hz = 1.0 / seconds
    if hz >= 1.0:
        return f"{hz:g} Hz"
    return f"1/{seconds:g} Hz"


class DecimateDialog(QDialog):
    """选择整数倍降采样因子。结构性变更。"""

    def __init__(self, em_data: EMData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("数据降采样 — MTPro")
        self.setMinimumWidth(480)

        self._em_data = em_data

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        info_label = QLabel(
            "对当前数据集做抗混叠整数倍抽取（就地修改，不可撤销）。\n"
            "数据中的 NaN 缺口会保持：覆盖缺口的输出样点仍为 NaN。",
            self,
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # ── 当前状态 ──────────────────────────────────────────
        duration = em_data.end_time - em_data.start_time
        current_label = QLabel(
            f"当前：采样周期 {em_data.dt.total_seconds():g}s "
            f"({format_sample_rate(em_data.dt)})，"
            f"{em_data.npts} 点，时长 {format_duration(duration)}，"
            f"通道数 {em_data.NCh}",
            self,
        )
        current_label.setStyleSheet("QLabel { color: #4070a0; font-weight: bold; }")
        current_label.setWordWrap(True)
        layout.addWidget(current_label)

        form = QFormLayout()
        form.setSpacing(6)

        self._factor_spin = QSpinBox(self)
        self._factor_spin.setRange(2, 1000)
        self._factor_spin.setValue(5)
        self._factor_spin.setSuffix(" 倍")
        self._factor_spin.setToolTip("整数抽取因子，如 5 表示 1s → 5s")
        form.addRow("降采样因子：", self._factor_spin)

        layout.addLayout(form)

        # ── 预览 ──────────────────────────────────────────────
        self._preview_label = QLabel(self)
        self._preview_label.setWordWrap(True)
        self._preview_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._preview_label)

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

        self._factor_spin.valueChanged.connect(self._update_preview)
        self._update_preview()

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def factor(self) -> int:
        return self._factor_spin.value()

    # ------------------------------------------------------------------
    # 私有槽
    # ------------------------------------------------------------------

    def _update_preview(self) -> None:
        factor = self._factor_spin.value()
        em = self._em_data

        new_dt = em.dt * factor
        new_npts = decimated_length(em.npts, factor)
        duration = em.end_time - em.start_time

        # 内存估计：通道数 × 点数 × 8 字节（float64）
        old_mb = em.NCh * em.npts * 8 / 1e6
        new_mb = em.NCh * new_npts * 8 / 1e6

        # 与 xlsx 导入共用同一经验上限（utils.limits），互通判断口径
        if new_npts > MAX_POINTS_PER_CHANNEL:
            suggest = suggested_decimate_factor(em.npts, MAX_POINTS_PER_CHANNEL)
            limit_line = (
                f"⚠ 降采样后仍高于建议上限 "
                f"{format_points(MAX_POINTS_PER_CHANNEL)} 点/通道"
                f"（内存与处理开销偏大），建议因子 ≥ {suggest}"
            )
        else:
            limit_line = (
                f"✓ 低于建议上限 {format_points(MAX_POINTS_PER_CHANNEL)} 点/通道"
            )

        self._preview_label.setText(
            f"降采样后：采样周期 {new_dt.total_seconds():g}s "
            f"({format_sample_rate(new_dt)})，"
            f"{new_npts} 点，时长 {format_duration(duration)}\n"
            f"数据内存估计：{old_mb:.0f} MB → {new_mb:.0f} MB\n"
            f"{limit_line}"
        )
        self._error_label.hide()

    def _on_accept(self) -> None:
        factor = self._factor_spin.value()
        new_npts = decimated_length(self._em_data.npts, factor)
        if new_npts < 10:
            self._error_label.setText(
                f"降采样后仅剩 {new_npts} 点，因子过大。"
            )
            self._error_label.show()
            return

        self._error_label.hide()
        self.accept()
