"""从二进制文件导入 — 四步向导。

Page 1 — 文件选择：选择 .dat / .inf 文件。
Page 2 — 通道与仪器：通道名称、电极对、缩放因子、偶极距、采样率。
Page 3 — 测点与时区：GPS 坐标、时区。
Page 4 — 确认导入。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import  Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
    QApplication,
)

from base.data_manager import DataManager
from io_utils.bin_io import DAT_CH_FIELDS, GPS_DIVISOR, read_bin_header, read_gps_header

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 时区
# ---------------------------------------------------------------------------

_TIMEZONE_DATA: list[tuple[str, str]] = [
    ("Asia/Shanghai",        "UTC+08:00"),
    ("Asia/Tokyo",           "UTC+09:00"),
    ("Asia/Bangkok",         "UTC+07:00"),
    ("Asia/Kolkata",         "UTC+05:30"),
    ("Asia/Almaty",          "UTC+05:00"),
    ("Europe/Moscow",        "UTC+03:00"),
    ("Europe/Berlin",        "UTC+01:00"),
    ("Europe/London",        "UTC+00:00"),
    ("UTC",                  "UTC+00:00"),
    ("America/New_York",     "UTC-05:00"),
    ("America/Denver",       "UTC-07:00"),
    ("America/Los_Angeles",  "UTC-08:00"),
    ("Pacific/Auckland",     "UTC+12:00"),
]
_TZ_DISPLAY_NAMES: list[str] = [f"{tz} ({off})" for tz, off in _TIMEZONE_DATA]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_DEFAULT_MAG_SCALE = 33.3333333
_DEFAULT_EX_DIST = 100.0
_DEFAULT_EY_DIST = 100.0

# ---------------------------------------------------------------------------
# 预设
# ---------------------------------------------------------------------------

_PRESET_LMT1 = {
    "names":  ["Hx", "Hy", "Hz", "", "Ex1", "Ey1", "Ex2", "Ey2"],
    "pairs":  ["",   "",   "",   "", "Ex",  "Ey",  "Ex",  "Ey"],
    "mag_scale": _DEFAULT_MAG_SCALE,
    "ex_distance": _DEFAULT_EX_DIST,
    "ey_distance": _DEFAULT_EY_DIST,
}
_PRESET_MT5 = {
    "names":  ["Ex", "Ey", "Hx", "Hy", "Hz", "", "", ""],
    "pairs":  ["Ex", "Ey", "",   "",   "",   "", "", ""],
    "mag_scale": _DEFAULT_MAG_SCALE,
    "ex_distance": _DEFAULT_EX_DIST,
    "ey_distance": _DEFAULT_EY_DIST,
}
_PRESET_DEFAULT = {
    "names":  [f"Ch{i}" for i in range(1, 9)],
    "pairs":  [""] * 8,
    "mag_scale": 1.0,
    "ex_distance": None,
    "ey_distance": None,
}

CHANNEL_PRESETS: dict[str, dict] = {
    "LMT-1 8通道 (Hx,Hy,Hz,—,Ex1..Ey2)": _PRESET_LMT1,
    "MT 5通道 (Ex,Ey,Hx,Hy,Hz)":          _PRESET_MT5,
    "默认 (Ch1~Ch8)":                     _PRESET_DEFAULT,
}


# ===================================================================
# Wizard
# ===================================================================


class ImportBinWizard(QWizard):
    def __init__(self, data_manager: Optional[DataManager] = None, parent=None) -> None:
        super().__init__(parent)
        self._data_manager = data_manager
        self._imported_key: int | None = None

        self.setWindowTitle("从二进制文件导入")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setMinimumWidth(580)
        self.setMinimumHeight(500)
        self.resize(600, 520)

        self._page1 = FileSelectPage()
        self._page2 = ChannelConfigPage()
        self._page3 = GpsTimezonePage()
        self._page4 = ConfirmPage()

        self.addPage(self._page1)
        self.addPage(self._page2)
        self.addPage(self._page3)
        self.addPage(self._page4)

    # -- accessors ----------------------------------------------------------

    def dat_path(self) -> str:
        return self._page1._dat_edit.text().strip()

    def gps_path(self) -> str | None:
        p = self._page1._gps_edit.text().strip()
        return p if p else None

    def data_name(self) -> str:
        n = self._page2._data_name_edit.text().strip()
        return n if n else ""

    def channel_configs(self) -> list[dict]:
        return self._page2._collect_configs()

    def dt_override(self) -> float | None:
        return self._page2._dt_override()

    def ex_distance(self) -> float:
        return self._page2._ex_dist_spin.value()

    def ey_distance(self) -> float:
        return self._page2._ey_dist_spin.value()

    def selected_timezone(self) -> str:
        return self._page3._selected_timezone()

    def gps_manual_values(self) -> tuple[float | None, float | None, float | None]:
        return self._page3._manual_gps_values()

    def imported_key(self) -> int | None:
        return self._imported_key

    def accept(self) -> None:
        super().accept()


# ===================================================================
# Page 1 — 文件选择
# ===================================================================


class FileSelectPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("选择数据文件")
        self.setSubTitle("选择要导入的二进制 .dat 文件，可同时关联 GPS 文件 (.inf)。")

        layout = QVBoxLayout(self)

        # -- 数据文件 --
        dat_group = QGroupBox("数据文件 (.dat)")
        dat_layout = QFormLayout(dat_group)

        dat_row = QHBoxLayout()
        self._dat_edit = QLineEdit()
        self._dat_edit.setPlaceholderText("选择 .dat 文件...")
        self._dat_edit.textChanged.connect(self._on_dat_changed)
        dat_row.addWidget(self._dat_edit)
        dat_browse = QPushButton("浏览...")
        dat_browse.clicked.connect(self._browse_dat)
        dat_row.addWidget(dat_browse)
        dat_layout.addRow(dat_row)

        self._dat_info = QLabel("")
        self._dat_info.setWordWrap(True)
        self._dat_info.setStyleSheet("color: #666;")
        dat_layout.addRow(self._dat_info)

        layout.addWidget(dat_group)

        # -- GPS 文件 --
        gps_group = QGroupBox("GPS 文件 (.inf) — 可选")
        gps_layout = QFormLayout(gps_group)

        self._auto_gps = QCheckBox("自动查找同名 .inf 文件")
        self._auto_gps.setChecked(True)
        self._auto_gps.toggled.connect(self._on_auto_gps_toggled)
        gps_layout.addRow(self._auto_gps)

        gps_row = QHBoxLayout()
        self._gps_edit = QLineEdit()
        self._gps_edit.setPlaceholderText("手动选择 .inf 文件...")
        self._gps_edit.textChanged.connect(self._refresh_gps_info)
        gps_row.addWidget(self._gps_edit)
        gps_browse = QPushButton("浏览...")
        gps_browse.clicked.connect(self._browse_gps)
        gps_row.addWidget(gps_browse)
        gps_layout.addRow(gps_row)

        self._gps_info = QLabel("")
        self._gps_info.setWordWrap(True)
        self._gps_info.setStyleSheet("color: #666;")
        gps_layout.addRow(self._gps_info)

        layout.addWidget(gps_group)
        layout.addStretch()

        self.registerField("dat_path*", self._dat_edit)

    def _browse_dat(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "",
            "二进制数据文件 (*.dat);;所有文件 (*)",
        )
        if path:
            self._dat_edit.setText(path)

    def _browse_gps(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 GPS 文件", "",
            "GPS 文件 (*.inf);;所有文件 (*)",
        )
        if path:
            self._gps_edit.setText(path)

    def _on_dat_changed(self, text: str) -> None:
        text = text.strip()
        if not text or not os.path.exists(text):
            self._dat_info.setText("")
            self._gps_info.setText("")
            self.completeChanged.emit()
            return
        try:
            info = read_bin_header(text)
        except Exception as e:
            self._dat_info.setText(f"⚠ 无法读取文件: {e}")
            self._dat_info.setStyleSheet("color: #c00;")
            self.completeChanged.emit()
            return
        self._dat_info.setStyleSheet("color: #333;")
        if info["npts"] == 0:
            self._dat_info.setText(
                f"文件大小: {info['file_size_mb']:.2f} MB  |  "
                f"记录数: 0（空文件或格式不符）"
            )
            self.completeChanged.emit()
            return
        start_str = str(info["start_time"])[:19] if info["start_time"] is not None else "?"
        end_str = str(info["end_time"])[:19] if info["end_time"] is not None else "?"
        freq = 1.0 / info["dt_sec"] if info["dt_sec"] and info["dt_sec"] > 0 else 0
        self._dat_info.setText(
            f"记录数: {info['npts']:,}  |  "
            f"文件大小: {info['file_size_mb']:.2f} MB\n"
            f"时间范围: {start_str}  ~  {end_str}\n"
            f"采样周期: {info['dt_sec']:.3f} s  (~{freq:.1f} Hz)"
        )
        if self._auto_gps.isChecked():
            p = Path(text)
            gps_candidate = str(p.with_suffix(".inf"))
            if os.path.exists(gps_candidate):
                self._gps_edit.setText(gps_candidate)
            else:
                self._gps_edit.clear()
        self._refresh_gps_info()
        self.completeChanged.emit()

    def _on_auto_gps_toggled(self, checked: bool) -> None:
        if checked and self._dat_edit.text().strip():
            self._on_dat_changed(self._dat_edit.text())

    def _refresh_gps_info(self) -> None:
        gps_path = self._gps_edit.text().strip()
        if not gps_path:
            self._gps_info.setText("未选择 GPS 文件")
            self._gps_info.setStyleSheet("color: #666;")
            return
        if not os.path.exists(gps_path):
            self._gps_info.setText("⚠ GPS 文件不存在")
            self._gps_info.setStyleSheet("color: #c00;")
            return
        try:
            info = read_gps_header(gps_path)
        except Exception as e:
            self._gps_info.setText(f"⚠ 无法读取 GPS 文件: {e}")
            self._gps_info.setStyleSheet("color: #c00;")
            return
        self._gps_info.setStyleSheet("color: #333;")
        lat_str = f"{info['lat']:.6f}°" if info["lat"] is not None else "?"
        lon_str = f"{info['lon']:.6f}°" if info["lon"] is not None else "?"
        self._gps_info.setText(
            f"✓ GPS 记录数: {info['npts']:,}  |  "
            f"坐标: {lat_str}, {lon_str}  (÷{GPS_DIVISOR:.0f})  |  "
            f"Alt={info['alt']:.1f} m"
        )

    def isComplete(self) -> bool:
        p = self._dat_edit.text().strip()
        return bool(p) and os.path.exists(p)


# ===================================================================
# Page 2 — 通道与仪器配置
# ===================================================================


class ChannelConfigPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("通道与仪器配置")
        self.setSubTitle("配置通道名称、电极对、缩放因子和偶极距。留空表示跳过该通道。")

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # -- 数据名称 --
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("数据名称:"))
        self._data_name_edit = QLineEdit()
        self._data_name_edit.setPlaceholderText("自动使用文件名")
        name_row.addWidget(self._data_name_edit, 1)
        layout.addLayout(name_row)

        # -- 通道配置 --
        ch_group = QGroupBox("通道配置")
        ch_layout = QVBoxLayout(ch_group)

        # 预设
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("快速预设:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(list(CHANNEL_PRESETS.keys()))
        self._preset_combo.currentTextChanged.connect(self._apply_preset)
        preset_row.addWidget(self._preset_combo, 1)
        ch_layout.addLayout(preset_row)
        ch_layout.addSpacing(6)

        # 表头 + 8 行
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(2)

        hf = QFont()
        hf.setBold(True)
        grid.addWidget(QLabel("原始列"), 0, 0)
        grid.addWidget(QLabel("通道名称"), 0, 1)
        grid.addWidget(QLabel("电极对"), 0, 2)

        self._name_edits: list[QLineEdit] = []
        self._pair_combos: list[QComboBox] = []
        self._dir_combos: list[tuple[QComboBox, QLabel]] = []

        for i in range(8):
            row = i + 1
            grid.addWidget(QLabel(f"  {DAT_CH_FIELDS[i]}  "), row, 0)

            name_edit = QLineEdit()
            name_edit.setPlaceholderText("(Empty)")
            name_edit.setMinimumWidth(100)
            name_edit.textChanged.connect(lambda text, r=row: self._on_name_changed(r))
            grid.addWidget(name_edit, row, 1)
            self._name_edits.append(name_edit)

            pair_combo = QComboBox()
            pair_combo.addItems(["—", "Ex", "Ey"])
            pair_combo.setMaximumWidth(72)
            idx = len(self._pair_combos)
            pair_combo.currentIndexChanged.connect(lambda _i, i=idx: self._update_dir_combo(self._dir_combos[i][0]))
            grid.addWidget(pair_combo, row, 2)
            self._pair_combos.append(pair_combo)

            # 每通道独立方向（Ex=N/S, Ey=E/W），同步模式隐藏
            dir_combo = QComboBox()
            dir_label = QLabel("")
            self._dir_combos.append((dir_combo, dir_label))
            grid.addWidget(dir_combo, row, 3)
            grid.addWidget(dir_label, row, 4)
            dir_combo.hide()
            dir_label.hide()

        grid_widget = QWidget()
        grid_widget.setLayout(grid)
        grid_scroll = QScrollArea()
        grid_scroll.setWidgetResizable(True)
        grid_scroll.setWidget(grid_widget)
        grid_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        grid_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        ch_layout.addWidget(grid_scroll, 1)  # stretch=1，由布局分配高度
        ch_layout.addSpacing(6)

        # 磁场缩放因子 + 偶极距
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("磁场缩放因子:"))
        self._mag_scale_spin = QDoubleSpinBox()
        self._mag_scale_spin.setRange(0.001, 999999.0)
        self._mag_scale_spin.setDecimals(6)
        self._mag_scale_spin.setValue(_DEFAULT_MAG_SCALE)
        self._mag_scale_spin.setToolTip("应用于所有磁场通道（H/B 开头）的缩放因子")
        param_row.addWidget(self._mag_scale_spin)
        param_row.addSpacing(16)
        param_row.addWidget(QLabel("Ex 偶极距:"))
        self._ex_dist_spin = QDoubleSpinBox()
        self._ex_dist_spin.setRange(0.0, 100000.0)
        self._ex_dist_spin.setDecimals(2)
        self._ex_dist_spin.setValue(_DEFAULT_EX_DIST)
        self._ex_dist_spin.setSuffix(" m")
        param_row.addWidget(self._ex_dist_spin)
        param_row.addWidget(QLabel("Ey 偶极距:"))
        self._ey_dist_spin = QDoubleSpinBox()
        self._ey_dist_spin.setRange(0.0, 100000.0)
        self._ey_dist_spin.setDecimals(2)
        self._ey_dist_spin.setValue(_DEFAULT_EY_DIST)
        self._ey_dist_spin.setSuffix(" m")
        param_row.addWidget(self._ey_dist_spin)
        param_row.addSpacing(16)
        self._dir_sync_cb = QCheckBox("电场方向同步")
        self._dir_sync_cb.setChecked(True)
        self._dir_sync_cb.setToolTip(
            "勾选时所有 Ex 通道共用方向，Ey 通道共用方向；\n"
            "取消后每个通道可独立设置电极方向，如 Ex1=N, Ex2=S 等。"
        )
        self._dir_sync_cb.toggled.connect(self._on_dir_sync_toggled)
        param_row.addWidget(self._dir_sync_cb)
        param_row.addWidget(QLabel("Ex:"))
        self._ex_dir_combo = QComboBox()
        self._ex_dir_combo.addItems(["N", "S"])
        param_row.addWidget(self._ex_dir_combo)
        param_row.addWidget(QLabel("Ey:"))
        self._ey_dir_combo = QComboBox()
        self._ey_dir_combo.addItems(["E", "W"])
        param_row.addWidget(self._ey_dir_combo)
        param_row.addStretch()
        ch_layout.addLayout(param_row)

        layout.addWidget(ch_group)

        # -- 采样周期 --
        sr_group = QGroupBox("采样周期")
        sr_layout = QHBoxLayout(sr_group)

        self._sr_radio_group = QButtonGroup(self)
        self._sr_auto_radio = QRadioButton("自动检测")
        self._sr_auto_radio.setChecked(True)
        self._sr_radio_group.addButton(self._sr_auto_radio, 0)
        sr_layout.addWidget(self._sr_auto_radio)

        self._sr_auto_label = QLabel("")
        self._sr_auto_label.setStyleSheet("color: #666;")
        sr_layout.addWidget(self._sr_auto_label)
        sr_layout.addSpacing(16)

        self._sr_manual_radio = QRadioButton("手动设置:")
        self._sr_radio_group.addButton(self._sr_manual_radio, 1)
        sr_layout.addWidget(self._sr_manual_radio)

        self._sr_manual_spin = QDoubleSpinBox()
        self._sr_manual_spin.setRange(0.0001, 999999.0)
        self._sr_manual_spin.setDecimals(4)
        self._sr_manual_spin.setValue(1.0)
        self._sr_manual_spin.setSuffix(" s")
        self._sr_manual_spin.setEnabled(False)
        sr_layout.addWidget(self._sr_manual_spin)

        self._sr_radio_group.buttonToggled.connect(self._on_sr_mode_changed)
        sr_layout.addStretch()
        layout.addWidget(sr_group)

        layout.addStretch()

        self._apply_preset(list(CHANNEL_PRESETS.keys())[0])

    # -- channel name change → auto-disable electrode pair -----------------

    def _on_dir_sync_toggled(self, checked: bool) -> None:
        """同步模式：显示全局方向，隐藏每通道方向。"""
        self._ex_dir_combo.setVisible(checked)
        self._ey_dir_combo.setVisible(checked)
        for _, label in self._dir_combos:
            label.setVisible(not checked)
        for combo, _ in self._dir_combos:
            combo.setVisible(not checked)
            if not checked:
                self._update_dir_combo(combo)

    def _update_dir_combo(self, combo: QComboBox) -> None:
        """根据通道的电极对设置方向下拉选项。"""
        for i, (dc, _) in enumerate(self._dir_combos):
            if dc is combo:
                pair = self._pair_combos[i].currentText()
                break
        else:
            return
        if pair == "Ex":
            combo.clear(); combo.addItems(["N (正)", "S (反)"])
        elif pair == "Ey":
            combo.clear(); combo.addItems(["E (正)", "W (反)"])
        else:
            combo.clear(); combo.setEnabled(False)

    def _on_name_changed(self, row: int) -> None:
        idx = row - 1
        name = self._name_edits[idx].text().strip()
        is_empty = not name or name.lower() == "empty"
        is_mag = not is_empty and name[0] in ("HhBb")
        if is_empty or is_mag:
            self._pair_combos[idx].setCurrentIndex(0)
            self._pair_combos[idx].setDisabled(True)
        else:
            self._pair_combos[idx].setDisabled(False)
        self._update_dir_combo(self._dir_combos[idx][0])

    # -- preset ------------------------------------------------------------

    def _apply_preset(self, preset_name: str) -> None:
        preset = CHANNEL_PRESETS.get(preset_name)
        if preset is None:
            return
        names = preset["names"]
        pairs = preset.get("pairs", [""] * 8)
        for i in range(8):
            self._name_edits[i].setText(names[i])
            p = pairs[i] if i < len(pairs) else ""
            if p == "Ex":   self._pair_combos[i].setCurrentIndex(1)
            elif p == "Ey": self._pair_combos[i].setCurrentIndex(2)
            else:           self._pair_combos[i].setCurrentIndex(0)
            self._update_dir_combo(self._dir_combos[i][0])
        ms = preset.get("mag_scale")
        if ms is not None: self._mag_scale_spin.setValue(ms)
        exd = preset.get("ex_distance")
        if exd is not None: self._ex_dist_spin.setValue(exd)
        eyd = preset.get("ey_distance")
        if eyd is not None: self._ey_dist_spin.setValue(eyd)
        for row in range(1, 9):
            self._on_name_changed(row)

    # -- collect -----------------------------------------------------------

    def _collect_configs(self) -> list[dict]:
        mag_scale = self._mag_scale_spin.value()
        sync = self._dir_sync_cb.isChecked()
        ex_global = self._ex_dir_combo.currentText()
        ey_global = self._ey_dir_combo.currentText()
        result = []
        for i in range(8):
            name = self._name_edits[i].text().strip()
            scale = mag_scale if (name and name[0] in "HhBb") else 1.0
            idx = self._pair_combos[i].currentIndex()
            pair = ["", "Ex", "Ey"][idx] if 0 <= idx <= 2 else ""
            # 方向
            direction = ""
            if sync:
                if pair == "Ex": direction = ex_global
                elif pair == "Ey": direction = ey_global
            else:
                t = self._dir_combos[i][0].currentText()
                if t and pair == "Ex" and t.startswith("S"): direction = "S"
                elif t and pair == "Ex": direction = "N"
                elif t and pair == "Ey" and t.startswith("W"): direction = "W"
                elif t and pair == "Ey": direction = "E"
            result.append({
                "name": name, "scale": scale,
                "electrode_pair": pair, "direction": direction,
            })
        return result

    # -- dt override -------------------------------------------------------

    def _on_sr_mode_changed(self, button, checked: bool) -> None:
        if not checked: return
        self._sr_manual_spin.setEnabled(self._sr_manual_radio.isChecked())

    def _dt_override(self) -> float | None:
        return self._sr_manual_spin.value() if self._sr_manual_radio.isChecked() else None

    # -- initialize --------------------------------------------------------

    def initializePage(self) -> None:
        dat = self.wizard().dat_path()
        try:
            info = read_bin_header(dat)
            dt_sec = info.get("dt_sec")
            if dt_sec:
                self._sr_auto_label.setText(f"({dt_sec:.3f}s, ~{1/dt_sec:.1f} Hz)")
                self._sr_manual_spin.setValue(dt_sec)
        except Exception:
            self._sr_auto_label.setText("(未检测)")
        dat_name = Path(dat).stem
        if not self._data_name_edit.text():
            self._data_name_edit.setText(dat_name)

    def isComplete(self) -> bool:
        return True


# ===================================================================
# Page 3 — GPS + 时区
# ===================================================================


class GpsTimezonePage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("测点与时区")
        self.setSubTitle("选择 GPS 坐标来源和采集站所在时区。")

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # -- GPS --
        gps_group = QGroupBox("GPS / 测点信息")
        gps_layout = QVBoxLayout(gps_group)

        self._gps_radio_group = QButtonGroup(self)

        self._gps_inf_radio = QRadioButton("从 INF 文件读取（已自动 ÷100）")
        self._gps_inf_radio.setChecked(True)
        self._gps_radio_group.addButton(self._gps_inf_radio, 0)
        gps_layout.addWidget(self._gps_inf_radio)

        inf_row = QHBoxLayout()
        inf_row.setContentsMargins(24, 0, 0, 0)
        self._gps_inf_preview = QLabel("")
        self._gps_inf_preview.setStyleSheet("color: #666;")
        inf_row.addWidget(self._gps_inf_preview, 1)
        gps_layout.addLayout(inf_row)

        self._gps_manual_radio = QRadioButton("手动输入:")
        self._gps_radio_group.addButton(self._gps_manual_radio, 1)
        gps_layout.addWidget(self._gps_manual_radio)

        manual_row = QHBoxLayout()
        manual_row.setContentsMargins(24, 0, 0, 0)
        manual_row.addWidget(QLabel("纬度 (°N):"))
        self._manual_lat = QLineEdit()
        self._manual_lat.setPlaceholderText("31.0723")
        self._manual_lat.setMaximumWidth(100)
        manual_row.addWidget(self._manual_lat)
        manual_row.addWidget(QLabel("经度 (°E):"))
        self._manual_lon = QLineEdit()
        self._manual_lon.setPlaceholderText("103.1471")
        self._manual_lon.setMaximumWidth(100)
        manual_row.addWidget(self._manual_lon)
        manual_row.addWidget(QLabel("海拔 (m):"))
        self._manual_alt = QLineEdit()
        self._manual_alt.setPlaceholderText("1705")
        self._manual_alt.setMaximumWidth(80)
        manual_row.addWidget(self._manual_alt)
        manual_row.addStretch()
        gps_layout.addLayout(manual_row)

        self._gps_radio_group.buttonToggled.connect(self._on_gps_mode_changed)
        layout.addWidget(gps_group)

        # -- 时区 --
        tz_group = QGroupBox("采集站时区")
        tz_layout = QFormLayout(tz_group)
        self._timezone_combo = QComboBox()
        self._timezone_combo.setEditable(True)
        self._timezone_combo.setMinimumWidth(240)
        self._timezone_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._timezone_combo.addItems(_TZ_DISPLAY_NAMES)
        self._timezone_combo.setCurrentIndex(0)
        tz_layout.addRow("时区:", self._timezone_combo)
        layout.addWidget(tz_group)

        layout.addStretch()

    def _on_gps_mode_changed(self, button, checked: bool) -> None:
        if not checked: return
        self._manual_lat.setEnabled(self._gps_manual_radio.isChecked())
        self._manual_lon.setEnabled(self._gps_manual_radio.isChecked())
        self._manual_alt.setEnabled(self._gps_manual_radio.isChecked())

    def _selected_timezone(self) -> str:
        display = self._timezone_combo.currentText().strip()
        idx = display.find(" (UTC")
        return display[:idx] if idx > 0 else display

    def _manual_gps_values(self) -> tuple[float | None, float | None, float | None]:
        if not self._gps_manual_radio.isChecked():
            return None, None, None
        return _try_float(self._manual_lat.text()), \
               _try_float(self._manual_lon.text()), \
               _try_float(self._manual_alt.text())

    def initializePage(self) -> None:
        wiz = self.wizard()
        gps_path = wiz.gps_path()
        if gps_path and os.path.exists(gps_path):
            try:
                info = read_gps_header(gps_path)
                lat_str = f"{info['lat']:.6f}°" if info["lat"] is not None else "?"
                lon_str = f"{info['lon']:.6f}°" if info["lon"] is not None else "?"
                alt_str = f"{info['alt']:.1f} m" if info["alt"] is not None else "?"
                self._gps_inf_preview.setText(
                    f"↳ {gps_path}\n   lat={lat_str}  lon={lon_str}  alt={alt_str}")
                self._gps_inf_preview.setStyleSheet("color: #333;")
                self._gps_inf_radio.setEnabled(True)
            except Exception:
                self._gps_inf_preview.setText("↳ 无法读取 INF 文件")
                self._gps_inf_preview.setStyleSheet("color: #c00;")
                self._gps_inf_radio.setEnabled(False)
                self._gps_manual_radio.setChecked(True)
        else:
            self._gps_inf_preview.setText("↳ 未选择 INF 文件")
            self._gps_inf_preview.setStyleSheet("color: #666;")
            self._gps_inf_radio.setEnabled(False)
            self._gps_manual_radio.setChecked(True)
        self._on_gps_mode_changed(None, True)

    def isComplete(self) -> bool:
        return True


# ===================================================================
# Page 4 — 确认导入
# ===================================================================


class ConfirmPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("确认导入")
        self.setSubTitle("检查导入信息，点击「完成」开始导入。")

        layout = QVBoxLayout(self)
        self._summary_label = QLabel("")
        self._summary_label.setWordWrap(True)
        self._summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._summary_label)
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        layout.addStretch()

    def initializePage(self) -> None:
        from io_utils.bin_io import _is_empty_channel

        wiz: ImportBinWizard = self.wizard()
        dat = wiz.dat_path()
        gps = wiz.gps_path()
        configs = wiz.channel_configs()
        tz = wiz.selected_timezone()
        exd = wiz.ex_distance()
        eyd = wiz.ey_distance()
        data_name = wiz.data_name() or Path(dat).stem

        try:
            info = read_bin_header(dat)
        except Exception:
            info = {}
        npts = info.get("npts", 0)
        dt_ov = wiz.dt_override()
        dt_src = "手动" if dt_ov is not None else "自动"
        dt_sec = dt_ov if dt_ov is not None else info.get("dt_sec")
        start = info.get("start_time")
        end = info.get("end_time")
        start_str = str(start)[:19] if start is not None else "?"
        end_str = str(end)[:19] if end is not None else "?"
        freq = 1.0 / dt_sec if dt_sec and dt_sec > 0 else 0

        parts: list[str] = []
        parts.append(
            "<table cellpadding='2' cellspacing='0'>"
            f"<tr><td><b>数据文件:</b></td><td>&nbsp;{dat}</td></tr>"
            f"<tr><td><b>数据名称:</b></td><td>&nbsp;{data_name}</td></tr>"
            f"<tr><td><b>记录数:</b></td><td>&nbsp;{npts:,}  |  "
            f"文件大小: {info.get('file_size_mb', 0):.2f} MB</td></tr>"
            f"<tr><td><b>时间范围:</b></td><td>&nbsp;{start_str} ~ {end_str}</td></tr>"
            f"<tr><td><b>采样周期:</b></td><td>&nbsp;{dt_sec:.3f} s  "
            f"(~{freq:.1f} Hz) [{dt_src}]</td></tr>"
            "</table>"
        )

        ch_rows = [
            "<tr><th align='left'>原始列</th><th align='left'>通道名称</th>"
            "<th align='left'>换算</th><th align='left'>电极对</th></tr>"
        ]
        for i, cfg in enumerate(configs):
            name = cfg["name"]
            if _is_empty_channel(name):
                ch_rows.append(
                    f"<tr><td>{DAT_CH_FIELDS[i]}</td>"
                    f"<td colspan='3' style='color:#888;'>&nbsp;(空) — 跳过</td></tr>")
            else:
                pair = cfg.get("electrode_pair", "")
                scale = cfg.get("scale", 1.0)
                ops = []
                if scale != 1.0: ops.append(f"×{scale:.6g}")
                if pair == "Ex" and exd: ops.append(f"÷{exd:.0f}m")
                elif pair == "Ey" and eyd: ops.append(f"÷{eyd:.0f}m")
                direction = cfg.get("direction", "")
                if direction in ("S", "W"): ops.append("×(–1)")
                conv = " ".join(ops) if ops else "—"
                ch_rows.append(
                    f"<tr><td>{DAT_CH_FIELDS[i]}</td>"
                    f"<td><b>{name}</b></td><td>{conv}</td>"
                    f"<td>{pair if pair else '—'}</td></tr>")
        parts.append(
            "<p></p>"
            "<table cellpadding='3' cellspacing='1'>" + "".join(ch_rows) + "</table>"
        )

        footer = (
            f"<p></p>"
            f"<b>Ex 偶极距:</b> {exd:.1f} m  |  <b>Ey 偶极距:</b> {eyd:.1f} m<br>"
        )
        mlat, mlon, malt = wiz.gps_manual_values()
        if mlat is not None:
            footer += f"<b>GPS (手动):</b> {mlat:.6f}°, {mlon:.6f}°, Alt={malt:.1f} m<br>"
        elif gps and os.path.exists(gps):
            try:
                gi = read_gps_header(gps)
                footer += f"<b>GPS (INF):</b> {gi['lat']:.6f}°, {gi['lon']:.6f}°, Alt={gi['alt']:.1f} m<br>"
            except Exception:
                footer += "<b>GPS:</b> 读取失败<br>"
        else:
            footer += "<b>GPS:</b> 无<br>"
        footer += f"<br><b>时区:</b> {tz}"
        parts.append(footer)

        self._summary_label.setText("".join(parts))
        self._status_label.setText("")

    def validatePage(self) -> bool:
        from PySide6.QtWidgets import QApplication

        wiz: ImportBinWizard = self.wizard()
        dat = wiz.dat_path()
        gps = wiz.gps_path()
        configs = wiz.channel_configs()
        tz = wiz.selected_timezone()
        exd = wiz.ex_distance()
        eyd = wiz.ey_distance()
        mlat, mlon, malt = wiz.gps_manual_values()

        progress = QProgressDialog(f"正在读取 {Path(dat).name} ...", "", 0, 3, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(100)
        progress.setWindowTitle("导入二进制数据")
        progress.setCancelButton(None)
        progress.setValue(0)
        QApplication.processEvents()

        try:
            from io_utils.bin_io import BinLoader
            progress.setValue(1)
            progress.setLabelText("正在解析二进制数据...")
            QApplication.processEvents()

            em_data = BinLoader.load(
                dat, tz=tz, channel_configs=configs,
                gps_path=gps, ex_distance=exd, ey_distance=eyd,
                manual_lat=mlat, manual_lon=mlon, manual_alt=malt,
                name=wiz.data_name() or None,
                dt_override=wiz.dt_override(),
            )

            progress.setValue(2)
            progress.setLabelText("正在注册数据...")
            QApplication.processEvents()
            key = wiz._data_manager.add(em_data)
            wiz._imported_key = key
            progress.setValue(3)
            progress.close()

            self._status_label.setText(
                f"<span style='color:green;'>✓ 导入成功!</span>"
                f"  #{key} {Path(dat).name} — {em_data.npts:,} 点, {em_data.NCh} 通道")
            self._status_label.setStyleSheet("")
            return True
        except Exception as e:
            progress.close()
            logger.exception("导入二进制文件失败: %s", dat)
            self._status_label.setText(f"<span style='color:#c00;'>✗ 导入失败: {e}</span>")
            self._status_label.setStyleSheet("")
            QMessageBox.critical(self, "导入失败", f"无法导入文件:\n{dat}\n\n错误: {e}")
            return False


def _try_float(s: str) -> float | None:
    try:
        return float(s.strip())
    except (ValueError, TypeError):
        return None

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    wiz = ImportBinWizard(None)
    wiz.show()
    sys.exit(app.exec())