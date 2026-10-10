"""从 MT 时序 TXT 文件导入 — 三步向导。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QListWidget,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from base.data_manager import DataManager


_TIMEZONES = [
    "Asia/Shanghai",
    "Asia/Urumqi",
    "Asia/Tokyo",
    "Asia/Bangkok",
    "Asia/Kolkata",
    "UTC",
    "Europe/London",
    "Europe/Paris",
    "Europe/Moscow",
    "America/Denver",
    "America/Los_Angeles",
    "America/New_York",
]


class ImportTxtWizard(QWizard):
    def __init__(self, data_manager: DataManager | None = None, parent=None) -> None:
        super().__init__(parent)
        self._data_manager = data_manager
        self._imported_key: int | None = None
        self._selected_files: list[str] = []

        self.setWindowTitle("从 MT TXT 时序文件导入")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setMinimumWidth(650)
        self.setMinimumHeight(500)
        self.resize(700, 540)

        self._page1 = TxtFilesPage()
        self._page2 = TxtSettingsPage()
        self._page3 = TxtConfirmPage()
        self.addPage(self._page1)
        self.addPage(self._page2)
        self.addPage(self._page3)

    def selected_files(self) -> list[str]:
        return list(self._selected_files)

    def data_name(self) -> str:
        value = self._page2._name_edit.text().strip()
        return value or Path(self._selected_files[0]).stem

    def selected_timezone(self) -> str:
        return self._page2._timezone_combo.currentText().strip()

    def imported_key(self) -> int | None:
        return self._imported_key


class TxtFilesPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("一次选择所有 TXT 文件")
        self.setSubTitle("选择组成同一条时序记录的全部文件；完成导入后不能再向该数据集追加文件。")

        layout = QVBoxLayout(self)
        group = QGroupBox("TXT 文件组")
        group_layout = QVBoxLayout(group)

        self._browse_button = QPushButton("选择多个 TXT 文件…")
        self._browse_button.clicked.connect(self._browse)
        group_layout.addWidget(self._browse_button, alignment=Qt.AlignmentFlag.AlignLeft)

        self._file_list = QListWidget()
        group_layout.addWidget(self._file_list, 1)

        self._info_label = QLabel("尚未选择文件")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #666;")
        group_layout.addWidget(self._info_label)
        layout.addWidget(group, 1)

    def _browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "一次选择所有 TXT 时序文件",
            "",
            "TXT 时序文件 (*.txt *.TXT);;所有文件 (*)",
        )
        if not paths:
            return

        # 每次选择都会替换整组，向导没有导入后继续追加的入口。
        unique_paths = list(dict.fromkeys(
            str(Path(path).expanduser().resolve(strict=False)) for path in paths
        ))
        wizard: ImportTxtWizard = self.wizard()
        wizard._selected_files = unique_paths
        self._file_list.clear()
        total_size = 0
        for path in unique_paths:
            try:
                size = Path(path).stat().st_size
            except OSError:
                size = 0
            total_size += size
            self._file_list.addItem(f"{Path(path).name}    {size / 1024 / 1024:.2f} MB\n{path}")

        self._info_label.setText(
            f"已选 {len(unique_paths)} 个文件，共 {total_size / 1024 / 1024:.2f} MB。"
            "文件会按时间合并为一个数据集。"
        )
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        wizard = self.wizard()
        return wizard is not None and bool(wizard._selected_files)


class TxtSettingsPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("数据名称与时区")
        self.setSubTitle("所有文件使用同一时区；采样间隔由文件时间戳检测。")

        layout = QVBoxLayout(self)
        settings = QGroupBox("导入参数")
        form = QFormLayout(settings)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("默认使用第一个文件的文件名")
        form.addRow("数据名称:", self._name_edit)

        self._timezone_combo = QComboBox()
        self._timezone_combo.setEditable(True)
        self._timezone_combo.addItems(_TIMEZONES)
        self._timezone_combo.setCurrentText("Asia/Shanghai")
        form.addRow("采集站时区:", self._timezone_combo)

        layout.addWidget(settings)
        note = QLabel(
            "TXT 没有单位元数据，导入后按磁场 nT、电场 mV/km 解释。\n"
            "文件间缺失的样点保留为 NaN；采样间隔不一致、时间网格不对齐，"
            "或重叠时间点数值冲突时会停止导入。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)


class TxtConfirmPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("确认导入")
        self.setSubTitle("确认文件组和参数后，点击「完成」导入为一个数据集。")

        layout = QVBoxLayout(self)
        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        self._summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._summary_label)
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        layout.addStretch(1)
        self._import_loop: QEventLoop | None = None
        self._import_result: dict[str, object] = {}

    def initializePage(self) -> None:
        wizard: ImportTxtWizard = self.wizard()
        file_rows = "".join(
            f"<tr><td>{index}. {Path(path).name}</td></tr>"
            for index, path in enumerate(wizard.selected_files(), 1)
        )
        self._summary_label.setText(
            "<table cellpadding='3' cellspacing='0'>"
            f"<tr><td><b>数据名称:</b></td><td>{wizard.data_name()}</td></tr>"
            f"<tr><td><b>文件数:</b></td><td>{len(wizard.selected_files())}</td></tr>"
            f"<tr><td><b>时区:</b></td><td>{wizard.selected_timezone()}</td></tr>"
            "</table><p><b>本次导入文件:</b></p>"
            "<table cellpadding='2' cellspacing='0'>"
            f"{file_rows}</table>"
            "<p>物理单位按磁场 nT、电场 mV/km 解释。</p>"
        )
        self._status_label.clear()

    def validatePage(self) -> bool:
        from io_utils.txt_io import TxtLoader
        from base.qt_bridge import run as run_in_pool

        wizard: ImportTxtWizard = self.wizard()
        paths = wizard.selected_files()
        progress = QProgressDialog("正在读取并合并 TXT 文件…", "", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setWindowTitle("导入 TXT 时序数据")
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()

        self._import_result = {}
        self._import_loop = QEventLoop(self)
        run_in_pool(
            "io",
            TxtLoader.load_group,
            paths,
            wizard.selected_timezone(),
            wizard.data_name(),
            on_done=self._on_load_done,
            on_error=self._on_load_failed,
            owner=self,
        )
        self._import_loop.exec()
        self._import_loop = None
        progress.close()

        try:
            error = self._import_result.get("error")
            if error is not None:
                raise error
            em_data = self._import_result["data"]
            key = wizard._data_manager.add(em_data) if wizard._data_manager else None
            wizard._imported_key = key
            gps_text = (
                f"；GPS {em_data.latitude:.6f}°, {em_data.longitude:.6f}°"
                if em_data.latitude is not None and em_data.longitude is not None
                else "；无 GPS"
            )
            self._status_label.setText(
                "<span style='color:green;'>导入成功</span>"
                f"　{len(paths)} 个文件，{em_data.npts:,} 点，{em_data.NCh} 通道，"
                f"dt={em_data.dt.total_seconds():g} s{gps_text}"
            )
            return True
        except Exception as exc:
            self._status_label.setText(f"<span style='color:#c00;'>导入失败：{exc}</span>")
            QMessageBox.critical(self, "TXT 导入失败", str(exc))
            return False

    def _on_load_done(self, em_data) -> None:
        self._import_result["data"] = em_data
        if self._import_loop is not None:
            self._import_loop.quit()

    def _on_load_failed(self, error: BaseException) -> None:
        self._import_result["error"] = error
        if self._import_loop is not None:
            self._import_loop.quit()
