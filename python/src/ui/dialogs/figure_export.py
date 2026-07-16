from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


PIXEL_PRESETS: list[tuple[str, Optional[tuple[int, int]]]] = [
    ("640 x 480", (640, 480)),
    ("800 x 600", (800, 600)),
    ("1024 x 768", (1024, 768)),
    ("1280 x 720", (1280, 720)),
    ("1600 x 1000", (1600, 1000)),
    ("1920 x 1080", (1920, 1080)),
    ("2560 x 1440", (2560, 1440)),
    ("自定义", None),
]


def default_export_name(station_name: str, plot_slug: str) -> str:
    station = sanitize_filename(station_name or "station")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{station}_{plot_slug}_{timestamp}"


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "figure"


class MatplotlibExportDialog(QDialog):
    def __init__(
        self,
        parent: Optional[QWidget],
        default_directory: str | Path,
        default_name: str,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("输出图像")
        self.setMinimumWidth(520)
        self._syncing_preset = False

        directory = Path(default_directory).expanduser()
        if not directory.exists():
            directory = Path.home()

        self.directory_edit = QLineEdit(str(directory))
        self.directory_edit.setReadOnly(True)
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_directory)
        directory_row = QHBoxLayout()
        directory_row.addWidget(self.directory_edit, 1)
        directory_row.addWidget(browse_btn)

        self.name_edit = QLineEdit(sanitize_filename(default_name))
        self.format_combo = QComboBox()
        for fmt in ("png", "svg", "pdf"):
            self.format_combo.addItem(fmt.upper(), fmt)
        self.format_combo.currentIndexChanged.connect(self._sync_extension)

        self.preset_combo = QComboBox()
        for label, size in PIXEL_PRESETS:
            self.preset_combo.addItem(label, size)
        self.preset_combo.setCurrentIndex(5)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 30000)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 30000)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(1, 1200)
        self.dpi_spin.setValue(300)
        self.width_spin.valueChanged.connect(self._on_size_edited)
        self.height_spin.valueChanged.connect(self._on_size_edited)
        self._apply_preset(1920, 1080)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("宽"))
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("高"))
        size_row.addWidget(self.height_spin)
        size_row.addWidget(QLabel("px"))

        form = QFormLayout()
        form.addRow("保存目录:", directory_row)
        form.addRow("图像名称:", self.name_edit)
        form.addRow("格式:", self.format_combo)
        form.addRow("尺寸预设:", self.preset_combo)
        form.addRow("像素尺寸:", size_row)
        form.addRow("DPI:", self.dpi_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self._sync_extension()

    @property
    def selected_format(self) -> str:
        return str(self.format_combo.currentData())

    @property
    def output_path(self) -> Path:
        name = self.name_edit.text().strip()
        path = Path(self.directory_edit.text()) / name
        suffix = f".{self.selected_format}"
        if path.suffix.lower() != suffix:
            path = path.with_suffix(suffix)
        return path

    @property
    def width_px(self) -> int:
        return int(self.width_spin.value())

    @property
    def height_px(self) -> int:
        return int(self.height_spin.value())

    @property
    def dpi(self) -> int:
        return int(self.dpi_spin.value())

    def accept(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "名称无效", "请输入图像名称。")
            return
        if Path(name).name != name:
            QMessageBox.warning(self, "名称无效", "图像名称不能包含目录。")
            return
        directory = Path(self.directory_edit.text())
        if not directory.exists():
            QMessageBox.warning(self, "目录无效", "请选择有效的保存目录。")
            return
        path = self.output_path
        if path.exists():
            answer = QMessageBox.question(
                self,
                "覆盖文件",
                f"{path.name} 已存在，是否覆盖？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        super().accept()

    def _browse_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择图像保存目录", self.directory_edit.text())
        if directory:
            self.directory_edit.setText(directory)

    def _sync_extension(self) -> None:
        text = self.name_edit.text().strip()
        if not text:
            return
        path = Path(text)
        suffix = f".{self.selected_format}"
        if path.suffix.lower() in {".png", ".svg", ".pdf"}:
            self.name_edit.setText(path.with_suffix(suffix).name)
        elif not path.suffix:
            self.name_edit.setText(f"{text}{suffix}")

    def _on_preset_changed(self, index: int) -> None:
        size = self.preset_combo.itemData(index)
        if size is None:
            return
        width, height = size
        self._apply_preset(width, height)

    def _apply_preset(self, width: int, height: int) -> None:
        self._syncing_preset = True
        try:
            self.width_spin.setValue(width)
            self.height_spin.setValue(height)
        finally:
            self._syncing_preset = False

    def _on_size_edited(self) -> None:
        if self._syncing_preset:
            return
        custom_index = self.preset_combo.findText("自定义")
        if custom_index >= 0 and self.preset_combo.currentIndex() != custom_index:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(custom_index)
            self.preset_combo.blockSignals(False)


def export_matplotlib_figure_with_dialog(
    parent: QWidget,
    figure,
    canvas,
    *,
    default_directory: str | Path,
    default_name: str,
) -> Optional[Path]:
    dialog = MatplotlibExportDialog(parent, default_directory, default_name)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    path = dialog.output_path
    save_matplotlib_figure(
        figure,
        canvas,
        path,
        width_px=dialog.width_px,
        height_px=dialog.height_px,
        dpi=dialog.dpi,
        fmt=dialog.selected_format,
    )
    return path


def save_matplotlib_figure(
    figure,
    canvas,
    path: str | Path,
    *,
    width_px: int,
    height_px: int,
    dpi: int,
    fmt: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old_size = figure.get_size_inches()
    try:
        figure.set_size_inches(width_px / dpi, height_px / dpi, forward=False)
        figure.savefig(
            path,
            dpi=dpi,
            format=fmt,
            facecolor="white",
            edgecolor="white",
            bbox_inches=None,
        )
    finally:
        figure.set_size_inches(old_size, forward=False)
        if canvas is not None:
            canvas.draw_idle()