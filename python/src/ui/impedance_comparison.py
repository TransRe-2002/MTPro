"""Independent window for comparing saved impedance and tipper results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.impedance_result import ImpedanceResult
from ui.dialogs.figure_export import default_export_name, export_matplotlib_figure_with_dialog
from ui.impedance_plot import draw_impedance_comparison, draw_tipper_comparison
from ui.theme import configure_matplotlib_canvas
from utils.plot_colors import DistinctColorAllocator


_BACKEND_LABELS = {
    "emtf_pywrap": "EMTF pywrap",
    "emtf_rs": "emtf-rs",
}


def _comparison_label(result: ImpedanceResult, duplicate: bool) -> str:
    """Disambiguate equal snapshot names with their processing backend."""
    if not duplicate:
        return result.name
    backend = str(result.parameters.get("backend") or "").strip()
    if not backend:
        return result.name
    return f"{result.name} · {_BACKEND_LABELS.get(backend, backend)}"


class ImpedanceComparisonWindow(QWidget):
    """Modeless top-level comparison window updated only by explicit requests."""

    def __init__(
        self,
        color_allocator: DistinctColorAllocator,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self._color_allocator = color_allocator
        self._results: list[ImpedanceResult] = []
        self._tipper_results: list[ImpedanceResult] = []
        self._impedance_series: list[tuple[str, object, tuple[int, int, int]]] = []
        self._tipper_series: list[tuple[str, object, tuple[int, int, int]]] = []

        self.setWindowTitle("阻抗结果对比")
        self.setMinimumSize(800, 600)
        self.resize(1200, 800)

        title = QLabel("阻抗结果对比", self)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.count_label = QLabel("已选择 0 个结果", self)
        self.tipper_mode_label = QLabel("图形类型：", self)
        self.tipper_mode_combo = QComboBox(self)
        self.tipper_mode_combo.addItem("实部/虚部", "re_im")
        self.tipper_mode_combo.addItem("幅值/相位", "amp_phase")
        self.tipper_mode_combo.currentIndexChanged.connect(self._draw_tipper)
        self.export_button = QPushButton("输出图像", self)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_figure)

        toolbar = QHBoxLayout()
        toolbar.addWidget(title)
        toolbar.addWidget(self.count_label)
        toolbar.addStretch(1)
        toolbar.addWidget(self.tipper_mode_label)
        toolbar.addWidget(self.tipper_mode_combo)
        toolbar.addWidget(self.export_button)

        self.impedance_figure = Figure(figsize=(12, 8))
        self.impedance_canvas = FigureCanvas(self.impedance_figure)
        configure_matplotlib_canvas(self.impedance_canvas, self.impedance_figure, self)
        self.tipper_figure = Figure(figsize=(12, 8))
        self.tipper_canvas = FigureCanvas(self.tipper_figure)
        configure_matplotlib_canvas(self.tipper_canvas, self.tipper_figure, self)

        self.figure = self.impedance_figure
        self.canvas = self.impedance_canvas
        self.tabs = QTabWidget(self)
        self.impedance_tab_index = self.tabs.addTab(self.impedance_canvas, "阻抗响应")
        self.tipper_tab_index = self.tabs.addTab(self.tipper_canvas, "倾子响应")
        self.tabs.currentChanged.connect(self._sync_tipper_controls)
        self.tabs.setTabVisible(self.tipper_tab_index, False)
        self._sync_tipper_controls()

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self.tabs, 1)

    @property
    def result_ids(self) -> tuple[str, ...]:
        return tuple(result.snapshot_id for result in self._results)

    @property
    def tipper_result_ids(self) -> tuple[str, ...]:
        return tuple(result.snapshot_id for result in self._tipper_results)

    def set_results(
        self,
        results: Sequence[ImpedanceResult],
        tipper_mode: str | None = None,
    ) -> None:
        self._results = list(results)
        self.count_label.setText(f"已选择 {len(self._results)} 个结果")
        self.export_button.setEnabled(bool(self._results))
        name_counts = Counter(result.name for result in self._results)
        labels = {
            result.snapshot_id: _comparison_label(result, name_counts[result.name] > 1)
            for result in self._results
        }
        self._impedance_series = [
            (
                labels[result.snapshot_id],
                result.bands,
                self._color_allocator.color_for(result.snapshot_id),
            )
            for result in self._results
            if result.bands is not None
        ]
        self._tipper_results = [result for result in self._results if result.has_tipper]
        self._tipper_series = [
            (
                labels[result.snapshot_id],
                result.bands,
                self._color_allocator.color_for(result.snapshot_id),
            )
            for result in self._tipper_results
            if result.bands is not None
        ]

        if tipper_mode is not None:
            mode_index = self.tipper_mode_combo.findData(tipper_mode)
            if mode_index >= 0:
                self.tipper_mode_combo.blockSignals(True)
                self.tipper_mode_combo.setCurrentIndex(mode_index)
                self.tipper_mode_combo.blockSignals(False)

        show_tipper = len(self._tipper_results) >= 2
        if not show_tipper and self.tabs.currentIndex() == self.tipper_tab_index:
            self.tabs.setCurrentIndex(self.impedance_tab_index)
        self.tabs.setTabVisible(self.tipper_tab_index, show_tipper)
        draw_impedance_comparison(
            self.impedance_figure,
            self.impedance_canvas,
            self._impedance_series,
            self,
        )
        if show_tipper:
            self._draw_tipper()
        else:
            draw_tipper_comparison(
                self.tipper_figure,
                self.tipper_canvas,
                [],
                str(self.tipper_mode_combo.currentData()),
                self,
            )
        self._sync_tipper_controls()

    def _draw_tipper(self, _index: int = -1) -> None:
        if len(self._tipper_results) < 2:
            return
        draw_tipper_comparison(
            self.tipper_figure,
            self.tipper_canvas,
            self._tipper_series,
            str(self.tipper_mode_combo.currentData()),
            self,
        )

    def _sync_tipper_controls(self, _index: int = -1) -> None:
        visible = (
            self.tabs.currentIndex() == self.tipper_tab_index
            and self.tabs.isTabVisible(self.tipper_tab_index)
        )
        self.tipper_mode_label.setVisible(visible)
        self.tipper_mode_combo.setVisible(visible)

    def _export_figure(self) -> None:
        if not self._results:
            return
        if self.tabs.currentIndex() == self.tipper_tab_index:
            figure = self.tipper_figure
            canvas = self.tipper_canvas
            slug = f"tipper_comparison_{self.tipper_mode_combo.currentData()}"
        else:
            figure = self.impedance_figure
            canvas = self.impedance_canvas
            slug = "impedance_comparison"
        station = self._results[0].station_name or "impedance"
        export_matplotlib_figure_with_dialog(
            self,
            figure,
            canvas,
            default_directory=Path.cwd(),
            default_name=default_export_name(station, slug),
        )
