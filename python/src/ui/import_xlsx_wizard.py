"""xlsx 时序文件导入向导。

提供 4 步交互式导入：选择文件/文件夹范围 → 确认通道映射 → 选定时区/
采样周期/插值方法 → 确认并真正调用 ``XlsxLoader.load_group`` 批量导入
到 ``DataManager``。支持「同前缀」和「整个文件夹」两种批量模式，按
文件名前缀自动分组。

入口在 ``MainWindow.import_xlsx``（菜单「文件 → 从 xlsx 时序文件导入」）。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QEventLoop, QObject, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from base.data_manager import DataManager
from base.qt_bridge import run as run_in_pool
from io_utils.xlsx_io import (
    INTERP_LINEAR,
    INTERPOLATION_METHODS,
    MAX_POINTS_PER_CHANNEL,
    XlsxImportCancelled,
    XlsxLoader,
    XlsxFileSummary,
    XlsxGroupSummary,
    XlsxWorkbookSummary,
    default_channel_configs,
    estimate_channel_points,
    format_points,
    is_real_xlsx_file,
    read_xlsx_structure,
    recommend_min_dt,
    summarize_xlsx_groups,
    xlsx_filename_prefix,
)

logger = logging.getLogger(__name__)


class _ImportSignals(QObject):
    """后台导入的信号总线。

    导入任务跑在全局共享 compute 池上（见 ``base.concurrency``），
    通过本对象的信号回报进度——信号从池线程 emit，经 Qt 排队连接
    回主线程槽，因此导入期间 UI 保持响应、取消按钮可用。
    """

    progress = Signal(int, int, str)               # (已完成 sheet 数, 总 sheet 数, 当前文件名)
    phase = Signal(str)                            # 阶段文案（读取 / 合并重采样）
    dataset_ready = Signal(object, int, int, int)  # (em_data, npts, 通道数, NaN 数)

_TIMEZONE_DATA: list[tuple[str, str]] = [
    ("Asia/Shanghai", "UTC+08:00"),
    ("Asia/Tokyo", "UTC+09:00"),
    ("Asia/Bangkok", "UTC+07:00"),
    ("Asia/Kolkata", "UTC+05:30"),
    ("Europe/Moscow", "UTC+03:00"),
    ("Europe/London", "UTC+00:00"),
    ("UTC", "UTC+00:00"),
    ("America/New_York", "UTC-05:00"),
    ("America/Los_Angeles", "UTC-08:00"),
]
_TZ_DISPLAY_NAMES = [f"{tz} ({offset})" for tz, offset in _TIMEZONE_DATA]


class ImportXlsxWizard(QWizard):
    def __init__(self, data_manager: DataManager | None = None, parent=None) -> None:
        super().__init__(parent)
        self._data_manager = data_manager
        self._imported_key: int | None = None
        self._imported_keys: list[int] = []
        self._summary: XlsxWorkbookSummary | None = None
        self._file_summaries: list[XlsxFileSummary] = []
        self._group_summaries: list[XlsxGroupSummary] = []
        self._selected_files: list[str] = []

        self.setWindowTitle("从 xlsx 时序文件导入")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setMinimumWidth(680)
        self.setMinimumHeight(560)
        self.resize(720, 600)

        self._page1 = XlsxFileSelectPage()
        self._page2 = XlsxChannelPage()
        self._page3 = XlsxSamplingPage()
        self._page4 = XlsxConfirmPage()

        self.addPage(self._page1)
        self.addPage(self._page2)
        self.addPage(self._page3)
        self.addPage(self._page4)

    def xlsx_path(self) -> str:
        return self._page1._path_edit.text().strip()

    def import_mode(self) -> str:
        return self._page1._mode_key()

    def selected_files(self) -> list[str]:
        return list(self._selected_files)

    def selected_groups(self) -> list[XlsxGroupSummary]:
        return list(self._group_summaries)

    def data_name(self) -> str:
        name = self._page2._data_name_edit.text().strip()
        return name if name else Path(self.xlsx_path()).stem

    def channel_configs(self) -> list[dict]:
        return self._page2._collect_configs()

    def selected_timezone(self) -> str:
        display = self._page3._timezone_combo.currentText().strip()
        idx = display.find(" (UTC")
        return display[:idx] if idx > 0 else display

    def dt_override(self) -> float | None:
        return self._page3._dt_override()

    def uses_manual_dt(self) -> bool:
        return self._page3._manual_dt_selected()

    def dt_for_group(self, group: XlsxGroupSummary) -> float | None:
        return self._page3._dt_for_group(group)

    def interpolation(self) -> str:
        return self._page3._interpolation_key()

    def max_side_gap_factor(self) -> float:
        return self._page3._max_side_gap_factor_spin.value()

    def imported_key(self) -> int | None:
        return self._imported_key

    def imported_keys(self) -> list[int]:
        return list(self._imported_keys)

    def import_workers(self) -> int:
        """导入线程池大小：0 = 自动（min(8, CPU 核数, 文件数)）；>0 = 手动指定。"""
        return self._page3._custom_workers()


class XlsxFileSelectPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("选择 xlsx 批量导入范围")
        self.setSubTitle("同前缀模式会收集同目录下相同文件名前缀的 xlsx；文件夹模式会导入目录下全部 xlsx。")

        layout = QVBoxLayout(self)

        mode_group = QGroupBox("批量模式")
        mode_layout = QVBoxLayout(mode_group)
        self._mode_group = QButtonGroup(self)
        self._same_prefix_radio = QRadioButton("同文件名前缀一并导入")
        self._same_prefix_radio.setChecked(True)
        self._folder_radio = QRadioButton("导入整个文件夹下的 xlsx 文件")
        self._mode_group.addButton(self._same_prefix_radio, 0)
        self._mode_group.addButton(self._folder_radio, 1)
        mode_layout.addWidget(self._same_prefix_radio)
        mode_layout.addWidget(self._folder_radio)
        self._mode_group.buttonToggled.connect(self._on_mode_changed)
        layout.addWidget(mode_group)

        file_group = QGroupBox("选择范围")
        file_layout = QFormLayout(file_group)

        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("选择一个代表 .xlsx 文件...")
        self._path_edit.textChanged.connect(self._on_path_changed)
        path_row.addWidget(self._path_edit)
        browse = QPushButton("浏览...")
        browse.clicked.connect(self._browse)
        path_row.addWidget(browse)
        self._browse_button = browse
        file_layout.addRow(path_row)

        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        self._info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        file_layout.addRow(self._info_label)

        layout.addWidget(file_group)
        layout.addStretch()
        self.registerField("xlsx_path*", self._path_edit)

    def _browse(self) -> None:
        if self._folder_radio.isChecked():
            path = QFileDialog.getExistingDirectory(self, "选择包含 xlsx 的文件夹", "")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "选择代表 xlsx 时序文件",
                "",
                "Excel 工作簿 (*.xlsx);;所有文件 (*)",
            )
        if path:
            self._path_edit.setText(path)

    def _mode_key(self) -> str:
        return "folder" if self._folder_radio.isChecked() else "same_prefix"

    def _on_mode_changed(self, button, checked: bool) -> None:
        if not checked:
            return
        if self._folder_radio.isChecked():
            self._path_edit.setPlaceholderText("选择包含 .xlsx 文件的文件夹...")
        else:
            self._path_edit.setPlaceholderText("选择一个代表 .xlsx 文件...")
        self._on_path_changed(self._path_edit.text())

    def _on_path_changed(self, text: str) -> None:
        path = text.strip()
        wizard: ImportXlsxWizard = self.wizard()
        wizard._summary = None
        wizard._file_summaries = []
        wizard._group_summaries = []
        wizard._selected_files = []
        if not path or not os.path.exists(path):
            self._info_label.setText("")
            self.completeChanged.emit()
            return

        try:
            files = self._resolve_selected_files(Path(path))
            if not files:
                self._info_label.setText("<span style='color:#c00;'>没有找到 xlsx 文件</span>")
                self.completeChanged.emit()
                return
            groups = summarize_xlsx_groups(files)
            wizard._selected_files = [str(file) for file in files]
            wizard._group_summaries = groups
            wizard._file_summaries = [file for group in groups for file in group.files]
            representative = groups[0].files[0].path
            wizard._summary = read_xlsx_structure(representative)
        except Exception as exc:
            self._info_label.setText(f"<span style='color:#c00;'>无法读取文件: {exc}</span>")
            self.completeChanged.emit()
            return

        self._info_label.setText(self._format_plan_html(wizard._group_summaries))
        self.completeChanged.emit()

    def _resolve_selected_files(self, path: Path) -> list[Path]:
        if self._folder_radio.isChecked():
            if not path.is_dir():
                raise ValueError("请选择文件夹")
            return sorted(
                item for item in path.rglob("*.xlsx")
                if is_real_xlsx_file(item)
            )
        if not path.is_file():
            raise ValueError("请选择一个 xlsx 文件")
        prefix = xlsx_filename_prefix(path)
        return sorted(
            item for item in path.parent.glob("*.xlsx")
            if is_real_xlsx_file(item) and xlsx_filename_prefix(item) == prefix
        )

    def _format_plan_html(self, groups: list[XlsxGroupSummary]) -> str:
        mode_text = "整个文件夹" if self._folder_radio.isChecked() else "同前缀"
        total_files = sum(len(group.files) for group in groups)
        rows = [
            "<tr><th align='left'>前缀</th><th align='right'>文件数</th>"
            "<th align='left'>时间范围</th><th align='left'>dt 分布</th>"
            "<th align='right'>重叠</th><th align='right'>贴边</th><th align='right'>gap</th></tr>"
        ]
        for group in groups:
            start = str(group.first_time)[:19] if group.first_time is not None else "?"
            end = str(group.last_time)[:19] if group.last_time is not None else "?"
            dt_text = ", ".join(
                f"{dt:g}s:{count}" for dt, count in group.dt_counts.most_common(4)
            ) or "?"
            rows.append(
                f"<tr><td><b>{group.prefix}</b></td>"
                f"<td align='right'>{len(group.files)}</td>"
                f"<td>{start} ~ {end}</td>"
                f"<td>{dt_text}</td>"
                f"<td align='right'>{len(group.overlaps)}</td>"
                f"<td align='right'>{len(group.touches)}</td>"
                f"<td align='right'>{len(group.gaps)}</td></tr>"
            )

        detail = ""
        if groups:
            first_group = groups[0]
            if first_group.overlaps:
                prev_name, current_name, seconds = first_group.overlaps[0]
                detail += (
                    f"<br><span style='color:#8a5a00;'>示例重叠:</span> "
                    f"{prev_name} / {current_name}，{seconds:g} s"
                )
            if first_group.gaps:
                prev_name, current_name, seconds = first_group.gaps[0]
                detail += (
                    f"<br><span style='color:#666;'>示例 gap:</span> "
                    f"{prev_name} / {current_name}，{seconds:g} s"
                )
        return (
            f"模式: {mode_text}<br>"
            f"计划导入: {total_files} 个 xlsx，{len(groups)} 个前缀组<br><br>"
            "<table cellpadding='3' cellspacing='0'>"
            + "".join(rows)
            + "</table>"
            + detail
        )

    def isComplete(self) -> bool:
        return (
            bool(self._path_edit.text().strip())
            and self.wizard()._summary is not None
            and bool(self.wizard()._group_summaries)
        )


class XlsxChannelPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("通道确认")
        self.setSubTitle("确认每个 sheet 对应的输出通道名。留空或取消勾选表示跳过。")

        layout = QVBoxLayout(self)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("数据名称:"))
        self._data_name_edit = QLineEdit()
        self._data_name_edit.setPlaceholderText("自动使用文件名")
        name_row.addWidget(self._data_name_edit, 1)
        layout.addLayout(name_row)

        group = QGroupBox("sheet -> 通道")
        group_layout = QVBoxLayout(group)

        self._grid = QGridLayout()
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(4)

        header_font = QFont()
        header_font.setBold(True)
        self._headers = ["导入", "sheet", "原始通道", "输出通道", "倍率"]
        for column, text in enumerate(self._headers):
            label = QLabel(text)
            label.setFont(header_font)
            self._grid.addWidget(label, 0, column)

        self._enabled_checks: list[QCheckBox] = []
        self._sheet_labels: list[QLabel] = []
        self._source_labels: list[QLabel] = []
        self._name_edits: list[QLineEdit] = []
        self._scale_spins: list[QDoubleSpinBox] = []

        self._grid_widget = QWidget()
        self._grid_widget.setLayout(self._grid)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._grid_widget)
        group_layout.addWidget(self._scroll)
        layout.addWidget(group, 1)

    def _clear_rows(self) -> None:
        """清除除表头外的所有数据行（用于 initializePage 之前重建）。"""
        for widgets in (
            self._enabled_checks,
            self._sheet_labels,
            self._source_labels,
            self._name_edits,
            self._scale_spins,
        ):
            for w in widgets:
                self._grid.removeWidget(w)
                w.setParent(None)
                w.deleteLater()
        self._enabled_checks.clear()
        self._sheet_labels.clear()
        self._source_labels.clear()
        self._name_edits.clear()
        self._scale_spins.clear()

    def _append_row(self, config: dict) -> None:
        """在 grid 末尾追加一行通道控件。"""
        row = len(self._enabled_checks) + 1  # 第 0 行是表头
        check = QCheckBox()
        check.setChecked(bool(config.get("enabled", True)))
        self._grid.addWidget(check, row, 0)
        self._enabled_checks.append(check)

        sheet_label = QLabel(config["sheet"])
        self._grid.addWidget(sheet_label, row, 1)
        self._sheet_labels.append(sheet_label)

        source_label = QLabel(config["source_name"])
        self._grid.addWidget(source_label, row, 2)
        self._source_labels.append(source_label)

        name_edit = QLineEdit(config["name"])
        name_edit.setMinimumWidth(120)
        self._grid.addWidget(name_edit, row, 3)
        self._name_edits.append(name_edit)

        scale_spin = QDoubleSpinBox()
        scale_spin.setRange(-999999.0, 999999.0)
        scale_spin.setDecimals(6)
        scale_spin.setValue(float(config.get("scale", 1.0)))
        scale_spin.setMaximumWidth(120)
        self._grid.addWidget(scale_spin, row, 4)
        self._scale_spins.append(scale_spin)

    def initializePage(self) -> None:
        wizard: ImportXlsxWizard = self.wizard()
        summary = wizard._summary
        if summary is None:
            return
        # 仅在摘要源变化时重建通道网格；否则保留用户已编辑的通道名/
        # 倍率/勾选状态（initializePage 每次进页都会被调用）
        if getattr(self, "_built_summary_path", None) != summary.path:
            configs = default_channel_configs(summary)
            self._clear_rows()
            for config in configs:
                self._append_row(config)
                # 通道名编辑变化时刷新"完成"按钮状态
                self._name_edits[-1].textChanged.connect(self.completeChanged)
            self._built_summary_path = summary.path
        if not self._data_name_edit.text():
            self._data_name_edit.setText(Path(summary.path).stem)

    def _collect_configs(self) -> list[dict]:
        configs: list[dict] = []
        for row in range(len(self._enabled_checks)):
            configs.append({
                "sheet": self._sheet_labels[row].text().strip(),
                "source_name": self._source_labels[row].text().strip(),
                "name": self._name_edits[row].text().strip(),
                "enabled": self._enabled_checks[row].isChecked(),
                "scale": self._scale_spins[row].value(),
            })
        return configs

    def isComplete(self) -> bool:
        return any(
            check.isChecked() and edit.text().strip()
            for check, edit in zip(self._enabled_checks, self._name_edits)
        )


class XlsxSamplingPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("采样周期与 gap")
        self.setSubTitle("选择目标采样周期，并设置对齐时的插值和 gap 策略。")

        layout = QVBoxLayout(self)

        tz_group = QGroupBox("采集站时区")
        tz_layout = QFormLayout(tz_group)
        self._timezone_combo = QComboBox()
        self._timezone_combo.setEditable(True)
        self._timezone_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._timezone_combo.addItems(_TZ_DISPLAY_NAMES)
        self._timezone_combo.setCurrentIndex(0)
        tz_layout.addRow("时区:", self._timezone_combo)
        layout.addWidget(tz_group)

        dt_group = QGroupBox("目标采样周期")
        dt_layout = QHBoxLayout(dt_group)
        self._dt_radio_group = QButtonGroup(self)
        self._dt_auto_radio = QRadioButton("自动检测")
        self._dt_auto_radio.setChecked(True)
        self._dt_radio_group.addButton(self._dt_auto_radio, 0)
        dt_layout.addWidget(self._dt_auto_radio)
        self._dt_auto_label = QLabel("")
        self._dt_auto_label.setStyleSheet("color: #666;")
        dt_layout.addWidget(self._dt_auto_label)
        dt_layout.addSpacing(16)
        self._dt_manual_radio = QRadioButton("手动设置:")
        self._dt_radio_group.addButton(self._dt_manual_radio, 1)
        dt_layout.addWidget(self._dt_manual_radio)
        self._dt_manual_spin = QDoubleSpinBox()
        self._dt_manual_spin.setRange(0.001, 999999.0)
        self._dt_manual_spin.setDecimals(4)
        self._dt_manual_spin.setValue(1.0)
        self._dt_manual_spin.setSuffix(" s")
        self._dt_manual_spin.setEnabled(False)
        dt_layout.addWidget(self._dt_manual_spin)
        dt_layout.addStretch()
        self._dt_radio_group.buttonToggled.connect(self._on_dt_mode_changed)
        self._dt_manual_spin.valueChanged.connect(lambda _: self._refresh_dt_estimate())
        layout.addWidget(dt_group)

        # 点数估计与超限预警（随 dt 模式/数值实时更新）
        self._dt_estimate_label = QLabel("")
        self._dt_estimate_label.setWordWrap(True)
        self._dt_estimate_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._dt_estimate_label)

        gap_group = QGroupBox("对齐与插值")
        gap_layout = QFormLayout(gap_group)
        self._interpolation_combo = QComboBox()
        for key, label in INTERPOLATION_METHODS.items():
            self._interpolation_combo.addItem(label, key)
        index = self._interpolation_combo.findData(INTERP_LINEAR)
        if index >= 0:
            self._interpolation_combo.setCurrentIndex(index)
        gap_layout.addRow("插值方式:", self._interpolation_combo)

        self._max_side_gap_factor_spin = QDoubleSpinBox()
        self._max_side_gap_factor_spin.setRange(0.1, 100.0)
        self._max_side_gap_factor_spin.setDecimals(2)
        self._max_side_gap_factor_spin.setValue(1.0)
        self._max_side_gap_factor_spin.setSingleStep(0.25)
        self._max_side_gap_factor_spin.setSuffix(" x dt")
        self._max_side_gap_factor_spin.setToolTip(
            "对线性插值，目标点到左、右两个有效实际点的距离都必须不超过该阈值。"
        )
        gap_layout.addRow("每侧最大距离:", self._max_side_gap_factor_spin)

        policy_label = QLabel(
            "默认策略: 原始命中点保留；非命中点只在左右有效点都足够近时插值，"
            "否则写入 NaN。"
        )
        policy_label.setWordWrap(True)
        policy_label.setStyleSheet("color: #666;")
        gap_layout.addRow(policy_label)
        interp_note = QLabel(
            "注: 手动 dt 为线性插值重采样（无抗混叠低通）。浏览足够；"
            "若用于频谱类处理，请使用 EMTF 处理链中的正规 decimation。"
        )
        interp_note.setWordWrap(True)
        interp_note.setStyleSheet("color: #666;")
        gap_layout.addRow(interp_note)
        layout.addWidget(gap_group)

        pool_group = QGroupBox("导入并发")
        pool_layout = QHBoxLayout(pool_group)
        self._workers_check = QCheckBox("自定义工作线程数:")
        self._workers_spin = QSpinBox()
        self._workers_spin.setRange(1, 32)
        self._workers_spin.setValue(8)
        self._workers_spin.setEnabled(False)
        self._workers_check.toggled.connect(self._workers_spin.setEnabled)
        pool_layout.addWidget(self._workers_check)
        pool_layout.addWidget(self._workers_spin)
        pool_note = QLabel("（未勾选时自动 = min(8, CPU 核数, 文件数)；只影响导入速度，不影响结果）")
        pool_note.setStyleSheet("color: #666;")
        pool_layout.addWidget(pool_note)
        pool_layout.addStretch()
        layout.addWidget(pool_group)
        layout.addStretch()

    def initializePage(self) -> None:
        wizard: ImportXlsxWizard = self.wizard()
        self._dt_manual_radio.setEnabled(True)
        self._dt_manual_spin.setEnabled(self._dt_manual_radio.isChecked())
        groups = wizard.selected_groups()
        dt_sec = groups[0].inferred_dt_sec if groups else None
        if dt_sec:
            self._dt_auto_label.setText(f"({dt_sec:.3f} s, ~{1.0 / dt_sec:.4g} Hz)")
            self._dt_manual_spin.setValue(dt_sec)
        else:
            self._dt_auto_label.setText("(未检测)")
            self._dt_manual_radio.setChecked(True)
        if wizard.import_mode() == "folder":
            self._dt_auto_label.setText(
                "(文件夹模式: 选自动则每组按自身 dt，选手动则统一使用下方 dt)"
            )
        self._refresh_dt_estimate()

    def _on_dt_mode_changed(self, button, checked: bool) -> None:
        if not checked:
            return
        self._dt_manual_spin.setEnabled(self._dt_manual_radio.isChecked())
        self._refresh_dt_estimate()

    def _refresh_dt_estimate(self) -> None:
        """按当前 dt 设置估计每组单通道点数，超限时给出预警和 dt 下限建议。"""
        wizard: ImportXlsxWizard = self.wizard()
        groups = wizard.selected_groups()
        lines: list[str] = []
        worst_recommend = 0.0
        over_limit = False
        for group in groups:
            if group.first_time is None or group.last_time is None:
                continue
            range_sec = (group.last_time - group.first_time).total_seconds()
            dt_sec = wizard.dt_for_group(group)
            if dt_sec is None or dt_sec <= 0:
                continue
            npts = estimate_channel_points(range_sec, dt_sec)
            prefix = f"{group.prefix}: " if len(groups) > 1 else ""
            lines.append(
                f"{prefix}约 <b>{format_points(npts)}</b> 点/通道（dt = {dt_sec:g} s，"
                f"时间跨度 {range_sec / 86400.0:g} 天）"
            )
            if npts > MAX_POINTS_PER_CHANNEL:
                over_limit = True
                worst_recommend = max(
                    worst_recommend,
                    recommend_min_dt(range_sec, MAX_POINTS_PER_CHANNEL),
                )
        if not lines:
            self._dt_estimate_label.setText("")
            return
        text = "预计数据量: " + "；".join(lines)
        if over_limit:
            text += (
                f"<br><span style='color:#8a5a00;'>⚠ 超过建议上限 "
                f"{format_points(MAX_POINTS_PER_CHANNEL)} 点/通道：内存占用与处理耗时"
                f"会明显上升。若要压到上限以内，建议 dt ≥ {worst_recommend:g} s"
                f"（在「目标采样周期」选手动设置），"
                f"也可先按原样导入、之后在数据树上右键「数据降采样」。</span>"
            )
        self._dt_estimate_label.setText(text)

    def _dt_override(self) -> float | None:
        if self._dt_manual_radio.isChecked():
            return self._dt_manual_spin.value()
        groups = self.wizard().selected_groups()
        return groups[0].inferred_dt_sec if groups else None

    def _manual_dt_selected(self) -> bool:
        return self._dt_manual_radio.isChecked()

    def _dt_for_group(self, group: XlsxGroupSummary) -> float | None:
        if self._dt_manual_radio.isChecked():
            return self._dt_manual_spin.value()
        return group.inferred_dt_sec

    def _interpolation_key(self) -> str:
        return str(self._interpolation_combo.currentData())

    def _custom_workers(self) -> int:
        if self._workers_check.isChecked():
            return self._workers_spin.value()
        return 0


class XlsxConfirmPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("确认导入")
        self.setSubTitle("检查参数，点击「完成」开始读取并重采样。")

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
        wizard: ImportXlsxWizard = self.wizard()
        groups = wizard.selected_groups()
        dt = wizard.dt_override()
        interpolation = INTERPOLATION_METHODS.get(wizard.interpolation(), wizard.interpolation())

        group_rows = [
            "<tr><th align='left'>前缀</th><th align='right'>文件数</th>"
            "<th align='left'>时间范围</th><th align='left'>dt</th>"
            "<th align='right'>~点数/通道</th>"
            "<th align='right'>重叠</th><th align='right'>gap</th></tr>"
        ]
        over_limit = False
        worst_recommend = 0.0
        for group in groups:
            start = str(group.first_time)[:19] if group.first_time is not None else "?"
            end = str(group.last_time)[:19] if group.last_time is not None else "?"
            group_dt = wizard.dt_for_group(group)
            dt_text = f"{group_dt:.4f} s" if group_dt else "?"
            npts_text = "?"
            if (
                group_dt
                and group.first_time is not None
                and group.last_time is not None
            ):
                range_sec = (group.last_time - group.first_time).total_seconds()
                npts = estimate_channel_points(range_sec, group_dt)
                npts_text = format_points(npts)
                if npts > MAX_POINTS_PER_CHANNEL:
                    over_limit = True
                    npts_text = f"<span style='color:#c00;'>{npts_text}</span>"
                    worst_recommend = max(
                        worst_recommend,
                        recommend_min_dt(range_sec, MAX_POINTS_PER_CHANNEL),
                    )
            group_rows.append(
                f"<tr><td><b>{group.prefix}</b></td>"
                f"<td align='right'>{len(group.files)}</td>"
                f"<td>{start} ~ {end}</td>"
                f"<td>{dt_text}</td>"
                f"<td align='right'>{npts_text}</td>"
                f"<td align='right'>{len(group.overlaps)}</td>"
                f"<td align='right'>{len(group.gaps)}</td></tr>"
            )

        rows = [
            "<tr><th align='left'>sheet</th><th align='left'>输出通道</th>"
            "<th align='left'>倍率</th></tr>"
        ]
        for config in wizard.channel_configs():
            if not config.get("enabled", True) or not config.get("name"):
                continue
            rows.append(
                f"<tr><td>{config['sheet']}</td><td><b>{config['name']}</b></td>"
                f"<td>{float(config.get('scale', 1.0)):.6g}</td></tr>"
            )

        mode_text = "整个文件夹" if wizard.import_mode() == "folder" else "同前缀"
        if wizard.uses_manual_dt() and dt is not None and dt > 0:
            dt_text = f"手动 {dt:.4f} s (~{1.0 / dt:.4g} Hz)"
        elif wizard.import_mode() == "folder":
            dt_text = "自动: 每个前缀组按自身 dt"
        elif dt is not None and dt > 0:
            dt_text = f"自动 {dt:.4f} s (~{1.0 / dt:.4g} Hz)"
        else:
            dt_text = "?"
        warning_html = ""
        if over_limit:
            warning_html = (
                "<p style='color:#8a5a00;'>⚠ 有前缀组单通道点数超过建议上限 "
                f"{format_points(MAX_POINTS_PER_CHANNEL)}：导入可以继续，但大数据集"
                f"会占用更多内存、处理更慢。若只是浏览，建议返回上一步手动设置 "
                f"dt ≥ {worst_recommend:g} s，或导入后在数据树上右键"
                f"「数据降采样」缩小数据量。</p>"
            )
        self._summary_label.setText(
            "<table cellpadding='3' cellspacing='0'>"
            f"<tr><td><b>模式:</b></td><td>&nbsp;{mode_text}</td></tr>"
            f"<tr><td><b>范围:</b></td><td>&nbsp;{wizard.xlsx_path()}</td></tr>"
            f"<tr><td><b>目标 dt:</b></td><td>&nbsp;{dt_text}</td></tr>"
            f"<tr><td><b>插值:</b></td><td>&nbsp;{interpolation}</td></tr>"
            f"<tr><td><b>每侧最大距离:</b></td><td>&nbsp;"
            f"{wizard.max_side_gap_factor():.2f} x dt</td></tr>"
            f"<tr><td><b>时区:</b></td><td>&nbsp;{wizard.selected_timezone()}</td></tr>"
            "</table><p></p>"
            "<table cellpadding='3' cellspacing='0'>"
            + "".join(group_rows)
            + "</table>"
            + warning_html
            + "<p></p>"
            "<table cellpadding='3' cellspacing='0'>"
            + "".join(rows)
            + "</table>"
        )
        self._status_label.setText("")

    def validatePage(self) -> bool:
        wizard: ImportXlsxWizard = self.wizard()
        groups = wizard.selected_groups()

        # 所有设置先在主线程快照，后台线程只读快照、不碰 widget 状态
        channel_configs = wizard.channel_configs()
        tz = wizard.selected_timezone()
        interpolation = wizard.interpolation()
        max_gap = wizard.max_side_gap_factor()
        folder_mode = wizard.import_mode() == "folder"
        base_name = wizard.data_name()
        requested_workers = wizard.import_workers()
        group_plans = [
            (group, [item.path for item in group.files], wizard.dt_for_group(group))
            for group in groups
        ]
        enabled_count = max(
            1,
            sum(1 for c in channel_configs if c.get("enabled", True) and c.get("name")),
        )
        total_units = max(
            1, sum(len(paths) for _, paths, _ in group_plans) * enabled_count,
        )

        # 复位之前的导入结果（重新点"完成"时清掉旧状态）
        wizard._imported_key = None
        wizard._imported_keys = []
        added_keys: list[int] = []
        failed_all: list[dict[str, str]] = []
        stats = {"npts": 0, "nch": 0, "nan": 0}
        cancel_flag = [False]
        outcome = {"cancelled": False, "error": ""}

        progress = QProgressDialog("准备导入...", "取消", 0, total_units, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setWindowTitle("导入 xlsx 时序数据")
        progress.setValue(0)
        progress.canceled.connect(lambda: cancel_flag.__setitem__(0, True))

        # 导入期间的临时状态挂到 self 上，供主线程槽函数读写。
        # 注意：进度信号必须连到本页的 QObject 槽（下方 _on_import_*），
        # 才能走排队连接回主线程——连到普通闭包会变成 DirectConnection，
        # 在池线程里直接碰 Qt 控件，会死锁。
        self._import_progress = progress
        self._import_added_keys = added_keys
        self._import_failed = failed_all
        self._import_stats = stats
        self._import_outcome = outcome

        def task(signals: _ImportSignals) -> None:
            offset = 0
            for group, group_paths, dt_override in group_plans:
                data_name = group.prefix if folder_mode else base_name
                base = offset

                def cb(done: int, _total: int, label: str, base: int = base) -> None:
                    signals.progress.emit(base + done, total_units, label)

                signals.phase.emit(f"正在导入 {group.prefix} ({len(group_paths)} 个文件)...")
                if requested_workers > 0:
                    workers = max(1, min(requested_workers, len(group_paths)))
                else:
                    workers = max(1, min(8, os.cpu_count() or 4, len(group_paths)))
                em_data = XlsxLoader.load_group(
                    group_paths,
                    tz=tz,
                    channel_configs=channel_configs,
                    name=data_name,
                    dt_override=dt_override,
                    interpolation=interpolation,
                    max_side_gap_factor=max_gap,
                    duplicate_policy="error",
                    workers=workers,
                    progress_callback=cb,
                    cancel_check=lambda: cancel_flag[0],
                )
                offset += len(group_paths) * enabled_count
                signals.phase.emit(f"正在合并与重采样 {group.prefix}...")
                failed_all.extend(
                    em_data.import_diagnostics.get("xlsx_import", {}).get(
                        "failed_files", []
                    )
                )
                nan_count = int(sum(
                    em_data.get_channel_series(ch).isna().sum()
                    for ch in em_data.channel_names()
                ))
                signals.dataset_ready.emit(
                    em_data, int(em_data.npts), int(em_data.NCh), nan_count,
                )

        signals = _ImportSignals(self)
        signals.progress.connect(self._on_import_progress)
        signals.phase.connect(self._on_import_phase)
        signals.dataset_ready.connect(self._on_import_dataset)

        loop = QEventLoop(self)
        self._import_loop = loop
        # 任务跑在全局共享 compute 池；完成/异常经 FutureBridge 排队回
        # 主线程（取消表现为 XlsxImportCancelled 异常，在 _on_import_failed
        # 里与错误分流）。
        run_in_pool(
            "compute", task, signals,
            on_done=self._on_import_succeeded,
            on_error=self._on_import_failed,
            owner=self,
        )
        loop.exec()   # UI 保持响应：进度刷新与取消按钮都走事件循环
        progress.close()
        self._import_loop = None
        self._import_progress = None

        if outcome["cancelled"]:
            logger.info("用户取消了 xlsx 导入")
            self._status_label.setText(
                "<span style='color:#8a5a00;'>已取消导入，已回滚</span>"
            )
            self._rollback(wizard, added_keys)
            return False

        if outcome["error"]:
            logger.error("导入 xlsx 文件失败: %s", outcome["error"])
            self._status_label.setText(
                f"<span style='color:#c00;'>导入失败: {outcome['error']}</span>"
            )
            QMessageBox.critical(
                self,
                "导入失败",
                f"无法导入:\n{wizard.xlsx_path()}\n\n错误: {outcome['error']}",
            )
            self._rollback(wizard, added_keys)
            return False

        # 成功 — 把 added_keys commit 到 wizard 状态后再统计展示
        wizard._imported_keys = list(added_keys)
        target = (
            f"{len(added_keys)} 个数据集"
            if wizard._data_manager is not None
            else "独立原型"
        )
        status_html = (
            f"<span style='color:green;'>导入成功</span>  {target}，"
            f"{stats['npts']:,} 点, {stats['nch']} 通道, NaN={stats['nan']:,}"
        )
        if failed_all:
            names = ", ".join(Path(item["path"]).name for item in failed_all)
            status_html += (
                f"<br><span style='color:#8a5a00;'>已跳过 {len(failed_all)} "
                f"个无法读取的文件: {names}</span>"
            )
            QMessageBox.warning(
                self,
                "部分文件被跳过",
                "以下文件无法读取，已跳过（其余文件导入成功）:\n\n"
                + "\n".join(
                    f"{Path(item['path']).name}: {item['error']}"
                    for item in failed_all
                ),
            )
        self._status_label.setText(status_html)
        return True

    # -- 后台导入信号槽（排队连接，主线程执行） -------------------------

    def _on_import_progress(self, done: int, total: int, label: str) -> None:
        progress = self._import_progress
        if progress is None:
            return
        progress.setMaximum(max(1, total))
        progress.setValue(done)
        progress.setLabelText(f"正在读取 {label} ({done}/{total} sheet)...")

    def _on_import_phase(self, text: str) -> None:
        if self._import_progress is not None:
            self._import_progress.setLabelText(text)

    def _on_import_dataset(self, em_data, npts: int, nch: int, nan_count: int) -> None:
        stats = self._import_stats
        stats["npts"] += npts
        stats["nch"] += nch
        stats["nan"] += nan_count
        wizard: ImportXlsxWizard = self.wizard()
        if wizard._data_manager is not None:
            key = wizard._data_manager.add(em_data)
            self._import_added_keys.append(key)
            if wizard._imported_key is None:
                wizard._imported_key = key

    def _on_import_succeeded(self, _result=None) -> None:
        if self._import_loop is not None:
            self._import_loop.quit()

    def _on_import_failed(self, exc: BaseException) -> None:
        """异常终结：XlsxImportCancelled 视为用户取消，其余为错误。"""
        if isinstance(exc, XlsxImportCancelled):
            self._import_outcome["cancelled"] = True
        else:
            logger.error("后台导入 xlsx 失败: %r", exc, exc_info=exc)
            self._import_outcome["error"] = str(exc)
        if self._import_loop is not None:
            self._import_loop.quit()

    @staticmethod
    def _rollback(wizard: "ImportXlsxWizard", added_keys: list[int]) -> None:
        """失败/取消回滚：移除已 add 到 DataManager 的数据集，避免半成品泄漏。"""
        if wizard._data_manager is not None:
            for key in added_keys:
                wizard._data_manager.remove(key)
        wizard._imported_key = None
        wizard._imported_keys = []


if __name__ == "__main__":
    app = QApplication(sys.argv)
    wizard = ImportXlsxWizard(None)
    if len(sys.argv) > 1:
        wizard._page1._path_edit.setText(sys.argv[1])
    wizard.show()
    sys.exit(app.exec())
