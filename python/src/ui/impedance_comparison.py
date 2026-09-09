"""Independent window for comparing saved impedance results."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from core.impedance_result import ImpedanceResult
from ui.dialogs.figure_export import default_export_name, export_matplotlib_figure_with_dialog
from ui.impedance_plot import draw_impedance_comparison
from ui.theme import configure_matplotlib_canvas
from utils.plot_colors import DistinctColorAllocator


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

        self.setWindowTitle("阻抗结果对比")
        self.setMinimumSize(800, 600)
        self.resize(1200, 800)

        title = QLabel("阻抗结果对比", self)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.count_label = QLabel("已选择 0 个结果", self)
        self.export_button = QPushButton("输出图像", self)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_figure)

        toolbar = QHBoxLayout()
        toolbar.addWidget(title)
        toolbar.addWidget(self.count_label)
        toolbar.addStretch(1)
        toolbar.addWidget(self.export_button)

        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        configure_matplotlib_canvas(self.canvas, self.figure, self)

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self.canvas, 1)

    @property
    def result_ids(self) -> tuple[str, ...]:
        return tuple(result.snapshot_id for result in self._results)

    def set_results(self, results: Sequence[ImpedanceResult]) -> None:
        self._results = list(results)
        self.count_label.setText(f"已选择 {len(self._results)} 个结果")
        self.export_button.setEnabled(bool(self._results))
        series = [
            (
                result.name,
                result.bands,
                self._color_allocator.color_for(result.snapshot_id),
            )
            for result in self._results
            if result.bands is not None
        ]
        draw_impedance_comparison(self.figure, self.canvas, series, self)

    def _export_figure(self) -> None:
        if not self._results:
            return
        station = self._results[0].station_name or "impedance"
        export_matplotlib_figure_with_dialog(
            self,
            self.figure,
            self.canvas,
            default_directory=Path.cwd(),
            default_name=default_export_name(station, "impedance_comparison"),
        )
