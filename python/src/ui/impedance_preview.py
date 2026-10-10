"""In-memory impedance result pool, comparison plots, and exports."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox,
    QPushButton, QSplitter, QStackedWidget, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from core.impedance_result import ImpedanceResult
from ui.impedance_comparison import ImpedanceComparisonWindow
from ui.impedance_stitch import ImpedanceStitchWidget
from ui.impedance_plot import draw_c_response, draw_impedance_response, draw_tipper_response
from output_tools.shared.c_response import c_response_from_bands
from ui.dialogs.figure_export import default_export_name, export_matplotlib_figure_with_dialog
from ui.theme import (
    configure_matplotlib_canvas,
    configure_matplotlib_fonts,
    hide_matplotlib_reset_button,
)
from utils.plot_colors import DistinctColorAllocator

configure_matplotlib_fonts()


ImpedanceSnapshot = ImpedanceResult


class ImpedanceResultPool(QObject):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[ImpedanceResult] = []


    def add_payload(self, payload: dict) -> ImpedanceResult:
        snapshot = ImpedanceResult.from_payload(payload)
        self._items.append(snapshot)
        self.changed.emit()
        return snapshot

    def add(self, method: str, result: object, name: str | None = None) -> ImpedanceResult:
        return self.add_payload({"method": method, "bands": result, "name": name})

    def get(self, snapshot_id: str) -> ImpedanceResult | None:
        return next((item for item in self._items if item.snapshot_id == snapshot_id), None)

    def update(self, snapshot_id: str, **changes) -> None:
        snapshot = self.get(snapshot_id)
        if snapshot is None:
            return
        for key, value in changes.items():
            if hasattr(snapshot, key):
                setattr(snapshot, key, value)
        self.changed.emit()

    def mark_exported(self, snapshot_id: str, file_format: str) -> None:
        snapshot = self.get(snapshot_id)
        if snapshot is not None:
            snapshot.exported_formats.add(file_format.lower())
            self.changed.emit()

    def remove_id(self, snapshot_id: str) -> None:
        previous = len(self._items)
        self._items = [item for item in self._items if item.snapshot_id != snapshot_id]
        if len(self._items) != previous:
            self.changed.emit()

    def remove(self, index: int) -> None:
        if 0 <= index < len(self._items):
            self._items.pop(index)
            self.changed.emit()

    def clear(self) -> None:
        if self._items:
            self._items.clear()
            self.changed.emit()

    def items(self) -> tuple[ImpedanceResult, ...]:
        return tuple(self._items)

    def unexported_count(self) -> int:
        return sum(not item.exported for item in self._items)


def banded_result_csv(bands: object) -> str:
    """Serialize the frequency-banded result (never raw FFT bins) as CSV."""
    periods = np.asarray(getattr(bands, "periods", []), dtype=float)
    count = len(periods)

    def arr(name: str, complex_values: bool = False):
        fill = np.nan + 0j if complex_values else np.nan
        value = getattr(bands, name, None)
        if value is None:
            return np.full(count, fill, dtype=complex if complex_values else float)
        value = np.asarray(value, dtype=complex if complex_values else float)
        if len(value) == count:
            return value
        output = np.full(count, fill, dtype=value.dtype)
        output[:min(count, len(value))] = value[:count]
        return output

    z_names = ("zxx", "zxy", "zyx", "zyy")
    z = {name: arr(name, True) for name in z_names}
    se = {name: arr(f"{name}_se") for name in z_names}
    derived_names = ("rho_xy", "rho_yx", "phi_xy", "phi_yx", "rho_xy_se", "rho_yx_se", "phi_xy_se", "phi_yx_se")
    derived = {name: arr(name) for name in derived_names}
    obs = arr("n_observations")
    tzx, tzy = arr("tzx", True), arr("tzy", True)
    tzx_se, tzy_se = arr("tzx_se"), arr("tzy_se")
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    header = ["frequency_hz", "period_s"]
    for name in z_names:
        header.extend((f"{name}_real", f"{name}_imag", f"{name}_se"))
    header.extend((*derived_names, "n_observations", "tzx_real", "tzx_imag", "tzx_se", "tzy_real", "tzy_imag", "tzy_se"))
    writer.writerow(header)
    for i, period in enumerate(periods):
        row = [1.0 / period if period > 0 else np.nan, period]
        for name in z_names:
            row.extend((z[name][i].real, z[name][i].imag, se[name][i]))
        row.extend(derived[name][i] for name in derived_names)
        row.extend((obs[i], tzx[i].real, tzx[i].imag, tzx_se[i], tzy[i].real, tzy[i].imag, tzy_se[i]))
        writer.writerow(row)
    return output.getvalue()


class ImpedancePreviewWidget(QWidget):
    def __init__(self, pool: ImpedanceResultPool, parent=None):
        super().__init__(parent)
        self.pool = pool
        self._comparison_ids: set[str] = set()
        self._color_allocator = DistinctColorAllocator()
        self._comparison_window: ImpedanceComparisonWindow | None = None

        # ── Left: temporary result management ──────────────────────
        self.list = QListWidget(self)
        self.list.setMinimumWidth(290)
        self.list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list.itemSelectionChanged.connect(self._refresh_selection)
        self.list.itemDoubleClicked.connect(self._begin_item_rename)
        self.list.itemChanged.connect(self._commit_item_rename)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._show_list_context_menu)

        self.method_filter = QComboBox(self)
        self.method_filter.addItem("全部", "all")
        self.method_filter.addItem("Robust 估计", "robust estimate")
        self.method_filter.addItem("远参考法", "remote reference")
        self.method_filter.addItem("拼接结果", "stitched impedance")
        self.method_filter.currentIndexChanged.connect(self._refresh_list)

        self.rename_btn = QPushButton("重命名", self)
        self.rename_btn.clicked.connect(self._rename_selected)
        self.delete_btn = QPushButton("删除", self)
        self.delete_btn.clicked.connect(self._delete_selected)
        self.clear_btn = QPushButton("清空", self)
        self.clear_btn.clicked.connect(self._clear_pool)
        self.select_all_btn = QPushButton("全选", self)
        self.select_all_btn.clicked.connect(lambda: self._set_visible_comparison_checked(True))
        self.clear_selection_btn = QPushButton("取消全选", self)
        self.clear_selection_btn.clicked.connect(lambda: self._set_visible_comparison_checked(False))
        self.compare_btn = QPushButton("对比所选 (0)", self)
        self.compare_btn.setEnabled(False)
        self.compare_btn.clicked.connect(self._open_comparison)
        self.stitch_btn = QPushButton("拼接所选 (0)", self)
        self.stitch_btn.setEnabled(False)
        self.stitch_btn.clicked.connect(self._open_stitch)

        left_toolbar = QHBoxLayout()
        left_toolbar.addWidget(self.method_filter, 1)
        left_toolbar.addWidget(self.rename_btn)
        left_toolbar.addWidget(self.delete_btn)
        left_toolbar.addWidget(self.clear_btn)

        comparison_toolbar = QHBoxLayout()
        comparison_toolbar.addWidget(self.select_all_btn)
        comparison_toolbar.addWidget(self.clear_selection_btn)
        comparison_toolbar.addStretch(1)
        comparison_toolbar.addWidget(self.stitch_btn)
        comparison_toolbar.addWidget(self.compare_btn)

        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("临时阻抗结果"))
        left_layout.addLayout(left_toolbar)
        left_layout.addWidget(self.list, 1)
        left_layout.addLayout(comparison_toolbar)

        # ── Right tab-bar exports ───────────────────────────────────
        self.export_edi_btn = QPushButton("输出 EDI", self)
        self.export_edi_btn.clicked.connect(lambda: self._export_data("edi"))
        self.export_z_btn = QPushButton("输出 Z", self)
        self.export_z_btn.clicked.connect(lambda: self._export_data("z"))
        self.export_csv_btn = QPushButton("输出 CSV", self)
        self.export_csv_btn.clicked.connect(lambda: self._export_data("csv"))
        self.export_figure_btn = QPushButton("输出图像", self)
        self.export_figure_btn.clicked.connect(self._export_figure)
        self.reset_view_btn = QPushButton("复原", self)
        self.reset_view_btn.clicked.connect(self._reset_current_figure)

        # ── Impedance tab: figure / adjustable table area ──────────
        self.main_figure = Figure(figsize=(8, 6), tight_layout=True)
        self.main_canvas = FigureCanvas(self.main_figure)
        configure_matplotlib_canvas(self.main_canvas, self.main_figure, self)
        hide_matplotlib_reset_button(self.main_canvas)

        self.impedance_table = self._make_table([
            "freq (Hz)", "period (s)", "Zxx", "Zxy", "Zyx", "Zyy",
            "Zxx SE", "Zxy SE", "Zyx SE", "Zyy SE",
        ])
        self.rho_table = self._make_table([
            "freq (Hz)", "period (s)", "ρxy", "ρyx", "φxy", "φyx",
            "ρxy SE", "ρyx SE", "φxy SE", "φyx SE",
        ])
        self.impedance_table_tabs = QTabWidget(self)
        self.impedance_table_tabs.addTab(self.impedance_table, "阻抗张量")
        self.impedance_table_tabs.addTab(self.rho_table, "视电阻率/相位")

        self.impedance_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.impedance_splitter.setChildrenCollapsible(False)
        self.impedance_splitter.addWidget(self.main_canvas)
        self.impedance_splitter.addWidget(self.impedance_table_tabs)
        self.impedance_splitter.setSizes([520, 280])
        impedance_page = QWidget(self)
        impedance_layout = QVBoxLayout(impedance_page)
        impedance_layout.setContentsMargins(0, 0, 0, 0)
        impedance_layout.addWidget(self.impedance_splitter)
        self._impedance_page = impedance_page

        # ── Tipper tab: plot-mode dropdown / adjustable table ──────
        self.tipper_table = self._make_table([
            "freq (Hz)", "period (s)", "Tzx", "Tzy", "Tzx SE", "Tzy SE",
        ])
        self.tipper_mode = QComboBox(self)
        self.tipper_mode.addItem("实部/虚部", "re_im")
        self.tipper_mode.addItem("幅值/相位", "amp_phase")
        self.tipper_mode.currentIndexChanged.connect(self._draw_tipper)
        tipper_mode_row = QHBoxLayout()
        tipper_mode_row.addStretch(1)
        tipper_mode_row.addWidget(QLabel("图形类型："))
        tipper_mode_row.addWidget(self.tipper_mode)

        self.tipper_figure = Figure(figsize=(8, 5), tight_layout=True)
        self.tipper_canvas = FigureCanvas(self.tipper_figure)
        configure_matplotlib_canvas(self.tipper_canvas, self.tipper_figure, self)
        hide_matplotlib_reset_button(self.tipper_canvas)

        tipper_plot_area = QWidget(self)
        tipper_plot_layout = QVBoxLayout(tipper_plot_area)
        tipper_plot_layout.setContentsMargins(0, 0, 0, 0)
        tipper_plot_layout.addLayout(tipper_mode_row)
        tipper_plot_layout.addWidget(self.tipper_canvas, 1)

        self.tipper_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.tipper_splitter.setChildrenCollapsible(False)
        self.tipper_splitter.addWidget(tipper_plot_area)
        self.tipper_splitter.addWidget(self.tipper_table)
        self.tipper_splitter.setSizes([520, 280])
        tipper_page = QWidget(self)
        tipper_layout = QVBoxLayout(tipper_page)
        tipper_layout.setContentsMargins(0, 0, 0, 0)
        tipper_layout.addWidget(self.tipper_splitter)
        self._tipper_page = tipper_page

        # ── C-response tab ────────────────────────────────────────
        self.c_response_table = self._make_table([
            "freq (Hz)", "period (s)", "Cxx (m)", "Cxy (m)", "Cyx (m)", "Cyy (m)",
        ])
        self.c_response_mode = QComboBox(self)
        self.c_response_mode.addItem("实部/虚部", "re_im")
        self.c_response_mode.addItem("幅值/相位", "amp_phase")
        self.c_response_mode.currentIndexChanged.connect(lambda _index: self._draw_c_response())
        c_response_mode_row = QHBoxLayout()
        c_response_mode_row.addStretch(1)
        c_response_mode_row.addWidget(QLabel("图形类型："))
        c_response_mode_row.addWidget(self.c_response_mode)

        self.c_response_figure = Figure(figsize=(8, 6), tight_layout=True)
        self.c_response_canvas = FigureCanvas(self.c_response_figure)
        configure_matplotlib_canvas(self.c_response_canvas, self.c_response_figure, self)
        hide_matplotlib_reset_button(self.c_response_canvas)
        c_response_plot_area = QWidget(self)
        c_response_plot_layout = QVBoxLayout(c_response_plot_area)
        c_response_plot_layout.setContentsMargins(0, 0, 0, 0)
        c_response_plot_layout.addLayout(c_response_mode_row)
        c_response_plot_layout.addWidget(self.c_response_canvas, 1)

        self.c_response_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.c_response_splitter.setChildrenCollapsible(False)
        self.c_response_splitter.addWidget(c_response_plot_area)
        self.c_response_splitter.addWidget(self.c_response_table)
        self.c_response_splitter.setSizes([520, 280])
        c_response_page = QWidget(self)
        c_response_layout = QVBoxLayout(c_response_page)
        c_response_layout.setContentsMargins(0, 0, 0, 0)
        c_response_layout.addWidget(self.c_response_splitter)
        self._c_response_page = c_response_page

        # ── Snapshot information tab ───────────────────────────────
        self.name_edit, self.station_edit, self.remote_edit = QLineEdit(), QLineEdit(), QLineEdit()
        self.lat_edit, self.lon_edit, self.elev_edit = QLineEdit(), QLineEdit(), QLineEdit()
        self.range_label, self.channels_label, self.params_label = QLabel(), QLabel(), QLabel()
        self.channels_label.setWordWrap(True)
        self.params_label.setWordWrap(True)
        apply_btn = QPushButton("应用快照信息", self)
        apply_btn.clicked.connect(self._apply_metadata)
        info_page = QWidget(self)
        info_form = QFormLayout(info_page)
        for label, widget in (("结果名称", self.name_edit), ("本地台站", self.station_edit), ("远参考站", self.remote_edit), ("纬度", self.lat_edit), ("经度", self.lon_edit), ("高程", self.elev_edit), ("计算时间段", self.range_label), ("通道映射", self.channels_label), ("计算参数", self.params_label)):
            info_form.addRow(label, widget)
        info_form.addRow("", apply_btn)

        self.tabs = QTabWidget(self)
        self._impedance_tab_index = self.tabs.addTab(impedance_page, "阻抗数据")
        self._tipper_tab_index = self.tabs.addTab(tipper_page, "倾子数据")
        self._c_response_tab_index = self.tabs.addTab(c_response_page, "C 响应函数")
        self._info_tab_index = self.tabs.addTab(info_page, "快照信息")
        self.tabs.currentChanged.connect(self._sync_export_controls)

        export_corner = QWidget(self.tabs)
        export_layout = QHBoxLayout(export_corner)
        export_layout.setContentsMargins(0, 0, 4, 0)
        export_layout.setSpacing(4)
        export_layout.addWidget(self.reset_view_btn)
        for button in (
            self.export_edi_btn, self.export_z_btn,
            self.export_csv_btn, self.export_figure_btn,
        ):
            export_layout.addWidget(button)
        self.tabs.setCornerWidget(export_corner, Qt.Corner.TopRightCorner)

        # Empty pool: no tabs/axes, just a centered two-line instruction.
        empty_page = QWidget(self)
        empty_layout = QVBoxLayout(empty_page)
        empty_layout.addStretch(1)
        empty_title = QLabel("暂未保存阻抗数据", empty_page)
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        empty_hint = QLabel('请在“阻抗输出”页面计算并保存一个阻抗结果', empty_page)
        empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_hint.setStyleSheet("color: #666666; font-size: 12px;")
        empty_layout.addWidget(empty_title)
        empty_layout.addWidget(empty_hint)
        empty_layout.addStretch(1)

        body = QSplitter(Qt.Orientation.Horizontal, self)
        body.addWidget(left_panel)
        body.addWidget(self.tabs)
        body.setSizes([310, 990])
        body.setStretchFactor(1, 1)

        content_page = QWidget(self)
        content_layout = QVBoxLayout(content_page)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(body)

        self.page_stack = QStackedWidget(self)
        self.page_stack.addWidget(empty_page)
        self.page_stack.addWidget(content_page)
        self.right_stack = self.page_stack  # compatibility alias

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.page_stack, 1)
        self.pool.changed.connect(self._refresh_list)
        self._refresh_list()

    @staticmethod
    def _make_table(headers: list[str]) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        return table

    @staticmethod
    def _snapshot_has_tipper(snapshot: ImpedanceSnapshot) -> bool:
        return snapshot.has_tipper

    def _begin_item_rename(self, item: QListWidgetItem, _column: int = 0) -> None:
        if item is None:
            return
        self.list.editItem(item)

    def _commit_item_rename(self, item: QListWidgetItem) -> None:
        snapshot_id = str(item.data(Qt.ItemDataRole.UserRole))
        snapshot = self.pool.get(snapshot_id)
        if snapshot is None:
            return

        if item.checkState() == Qt.CheckState.Checked:
            self._comparison_ids.add(snapshot_id)
        else:
            self._comparison_ids.discard(snapshot_id)
        self._sync_comparison_controls()

        name = item.text().strip()
        if not name:
            self._refresh_list()
            return
        if name != snapshot.name:
            snapshot.name = name
            self.pool.changed.emit()

    def _checked_results(self) -> list[ImpedanceResult]:
        return [
            result
            for result in self.pool.items()
            if result.snapshot_id in self._comparison_ids
        ]

    def _sync_comparison_controls(self) -> None:
        count = len(self._checked_results())
        self.compare_btn.setText(f"对比所选 ({count})")
        self.compare_btn.setEnabled(count >= 2)
        self.stitch_btn.setText(f"拼接所选 ({count})")
        self.stitch_btn.setEnabled(count == 2)
        self.select_all_btn.setEnabled(self.list.count() > 0)
        self.clear_selection_btn.setEnabled(count > 0)

    def _set_visible_comparison_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.list.blockSignals(True)
        for index in range(self.list.count()):
            item = self.list.item(index)
            snapshot_id = str(item.data(Qt.ItemDataRole.UserRole))
            item.setCheckState(state)
            if checked:
                self._comparison_ids.add(snapshot_id)
            else:
                self._comparison_ids.discard(snapshot_id)
        self.list.blockSignals(False)
        self._sync_comparison_controls()

    def _open_comparison(self) -> None:
        results = self._checked_results()
        if len(results) < 2:
            QMessageBox.information(self, "阻抗结果对比", "请至少勾选两个阻抗结果。")
            return
        if self._comparison_window is None:
            self._comparison_window = ImpedanceComparisonWindow(
                self._color_allocator,
                self.window(),
            )
        self._comparison_window.set_results(
            results,
            tipper_mode=str(self.tipper_mode.currentData()),
        )
        self._comparison_window.show()
        self._comparison_window.raise_()
        self._comparison_window.activateWindow()

    def _open_stitch(self) -> None:
        results = self._checked_results()
        if len(results) != 2:
            QMessageBox.information(self, "阻抗周期拼接", "请勾选恰好两个阻抗结果。")
            return
        widget = ImpedanceStitchWidget(results, self)
        widget.result_created.connect(self._save_stitched_result)
        widget.exec()

    def _save_stitched_result(self, payload: dict) -> None:
        result = self.pool.add_payload(payload)
        self.method_filter.setCurrentIndex(self.method_filter.findData("all"))
        self.select_snapshot(result.snapshot_id)

    def _show_list_context_menu(self, pos) -> None:
        item = self.list.itemAt(pos)
        if item is None:
            return
        if not item.isSelected():
            self.list.clearSelection()
            item.setSelected(True)
            self.list.setCurrentItem(item)
        menu = QMenu(self)
        rename_action = menu.addAction("重命名")
        delete_action = menu.addAction("删除")
        action = menu.exec(self.list.viewport().mapToGlobal(pos))
        if action is rename_action:
            self._begin_item_rename(item)
        elif action is delete_action:
            self._delete_selected()

    def _sync_export_controls(self, _index: int = -1) -> None:
        selected = self._selected()
        one = len(selected) == 1
        for button in (self.export_edi_btn, self.export_z_btn, self.export_csv_btn):
            button.setEnabled(one)
        current = self.tabs.currentWidget()
        figure_page = current is self._impedance_page or (
            current is self._tipper_page and self.tabs.isTabVisible(self._tipper_tab_index)
        ) or current is self._c_response_page
        self.reset_view_btn.setEnabled(bool(selected) and figure_page)
        self.export_figure_btn.setEnabled(bool(selected) and figure_page)

    def select_snapshot(self, snapshot_id: str) -> None:
        snapshot = self.pool.get(snapshot_id)
        current_filter = self.method_filter.currentData()
        if snapshot is not None and current_filter not in ("all", snapshot.method):
            self.method_filter.setCurrentIndex(self.method_filter.findData("all"))
        self._refresh_list()
        for index in range(self.list.count()):
            item = self.list.item(index)
            selected = item.data(Qt.ItemDataRole.UserRole) == snapshot_id
            item.setSelected(selected)
            if selected:
                self.list.setCurrentItem(item)
                self.list.scrollToItem(item)

    def _selected(self) -> list[ImpedanceSnapshot]:
        return [snapshot for item in self.list.selectedItems() if (snapshot := self.pool.get(str(item.data(Qt.ItemDataRole.UserRole))))]

    def _refresh_list(self) -> None:
        selected_ids = {
            str(item.data(Qt.ItemDataRole.UserRole))
            for item in self.list.selectedItems()
        }
        available_ids = {result.snapshot_id for result in self.pool.items()}
        self._comparison_ids.intersection_update(available_ids)

        if self._comparison_window is not None:
            missing_from_window = any(
                snapshot_id not in available_ids
                for snapshot_id in self._comparison_window.result_ids
            )
            if missing_from_window:
                remaining = [
                    result
                    for result in self.pool.items()
                    if result.snapshot_id in self._comparison_window.result_ids
                ]
                if len(remaining) >= 2:
                    self._comparison_window.set_results(remaining)
                else:
                    self._comparison_window.hide()

        self.list.blockSignals(True)
        self.list.clear()
        method = self.method_filter.currentData()
        for snapshot in self.pool.items():
            if method != "all" and snapshot.method != method:
                continue
            status = "已输出" if snapshot.exported else "未输出"
            item = QListWidgetItem(snapshot.name, self.list)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsEditable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(
                Qt.CheckState.Checked
                if snapshot.snapshot_id in self._comparison_ids
                else Qt.CheckState.Unchecked
            )
            item.setToolTip(f"{snapshot.method} · {status}")
            item.setData(Qt.ItemDataRole.UserRole, snapshot.snapshot_id)
            item.setSelected(snapshot.snapshot_id in selected_ids)
        self.list.blockSignals(False)
        self.right_stack.setCurrentIndex(1 if self.pool.items() else 0)
        self._sync_comparison_controls()
        self._refresh_selection()

    def _refresh_selection(self) -> None:
        selected = self._selected()
        one = len(selected) == 1
        has_tipper = any(self._snapshot_has_tipper(snapshot) for snapshot in selected)
        self.tabs.setTabVisible(self._tipper_tab_index, has_tipper)
        self._draw_main(selected)
        self._draw_tipper()
        self._draw_c_response(selected)
        self._fill_details(selected[0] if one else None)
        self.rename_btn.setEnabled(one)
        self.delete_btn.setEnabled(bool(selected))
        self.clear_btn.setEnabled(bool(self.pool.items()))
        self._sync_export_controls()

    def _draw_main(self, selected: list[ImpedanceSnapshot]) -> None:
        draw_impedance_response(
            self.main_figure,
            self.main_canvas,
            [snapshot.bands for snapshot in selected],
            self,
        )

    def _draw_tipper(self) -> None:
        draw_tipper_response(
            self.tipper_figure,
            self.tipper_canvas,
            [snapshot.bands for snapshot in self._selected()],
            str(self.tipper_mode.currentData()),
            self,
        )

    def _draw_c_response(self, selected: list[ImpedanceSnapshot] | None = None) -> None:
        selected = self._selected() if selected is None else selected
        draw_c_response(
            self.c_response_figure,
            self.c_response_canvas,
            [snapshot.bands for snapshot in selected],
            mode=str(self.c_response_mode.currentData()),
            widget=self,
        )

    def _fill_details(self, snapshot: ImpedanceSnapshot | None) -> None:
        for table in (
            self.impedance_table, self.rho_table, self.tipper_table, self.c_response_table,
        ):
            table.setRowCount(0)
        edits = (self.name_edit, self.station_edit, self.remote_edit, self.lat_edit, self.lon_edit, self.elev_edit)
        for edit in edits: edit.setEnabled(snapshot is not None); edit.clear()
        self.range_label.clear(); self.channels_label.clear()
        self.params_label.setText("多选时仅显示对比图" if snapshot is None and len(self._selected()) > 1 else "")
        if snapshot is None: return
        for edit, value in zip(edits, (snapshot.name, snapshot.station_name, snapshot.remote_station_name, snapshot.latitude, snapshot.longitude, snapshot.elevation)): edit.setText(str(value))
        self.range_label.setText(f"{snapshot.range_start} — {snapshot.range_end}")
        self.channels_label.setText(", ".join(f"{key}={value}" for key, value in snapshot.channels.items()))
        self.params_label.setText(", ".join(f"{key}={value}" for key, value in snapshot.parameters.items()))
        self._fill_tables(snapshot.bands)

    def _fill_tables(self, bands: object) -> None:
        if bands is None: return
        p = np.asarray(getattr(bands, "periods", []), dtype=float); f = np.asarray(getattr(bands, "center_frequencies_hz", []), dtype=float)
        z_names = ("zxx", "zxy", "zyx", "zyy"); z = [np.asarray(getattr(bands, name, [])) for name in z_names]; se = [np.asarray(getattr(bands, f"{name}_se", []), dtype=float) for name in z_names]
        n = min([len(p), len(f), *map(len, z), *map(len, se)], default=0)
        self._set_rows(self.impedance_table, [[f[i], p[i], *(self._complex_text(values[i]) for values in z), *(values[i] for values in se)] for i in range(n)])
        names = ("rho_xy", "rho_yx", "phi_xy", "phi_yx", "rho_xy_se", "rho_yx_se", "phi_xy_se", "phi_yx_se"); values = [np.asarray(getattr(bands, name, []), dtype=float) for name in names]
        n = min([len(p), len(f), *map(len, values)], default=0)
        self._set_rows(self.rho_table, [[f[i], p[i], *(array[i] for array in values)] for i in range(n)])
        tzx, tzy = getattr(bands, "tzx", None), getattr(bands, "tzy", None)
        if tzx is not None and tzy is not None:
            tzx, tzy = np.asarray(tzx), np.asarray(tzy); tzx_se = np.asarray(getattr(bands, "tzx_se", np.full(len(p), np.nan))); tzy_se = np.asarray(getattr(bands, "tzy_se", np.full(len(p), np.nan))); n = min(map(len, (p, f, tzx, tzy, tzx_se, tzy_se)))
            self._set_rows(self.tipper_table, [[f[i], p[i], self._complex_text(tzx[i]), self._complex_text(tzy[i]), tzx_se[i], tzy_se[i]] for i in range(n)])
        components = c_response_from_bands(bands)
        if components:
            n = min(len(p), len(f), *(len(values) for values in components.values()))
            self._set_rows(
                self.c_response_table,
                [[f[i], p[i], *(self._complex_text(components[name][i]) for name in ("cxx", "cxy", "cyx", "cyy"))] for i in range(n)],
            )

    @staticmethod
    def _set_rows(table: QTableWidget, rows: list[list[object]]) -> None:
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column, value in enumerate(row):
                table.setItem(row_index, column, QTableWidgetItem(value if isinstance(value, str) else f"{float(value):.6g}"))

    @staticmethod
    def _complex_text(value: complex) -> str:
        return f"{value.real:.6g}{value.imag:+.6g}j"

    def _single_selected(self) -> ImpedanceSnapshot | None:
        selected = self._selected()
        if len(selected) != 1:
            QMessageBox.information(self, "请选择结果", "阻抗数据每次只能输出一个结果。")
            return None
        return selected[0]

    def _rename_selected(self) -> None:
        selected_items = self.list.selectedItems()
        if len(selected_items) != 1:
            QMessageBox.information(self, "请选择结果", "请选择一个结果进行重命名。")
            return
        self._begin_item_rename(selected_items[0])

    def _apply_metadata(self) -> None:
        snapshot = self._single_selected()
        if snapshot is None: return
        try:
            changes = {"name": self.name_edit.text().strip() or snapshot.name, "station_name": self.station_edit.text().strip(), "remote_station_name": self.remote_edit.text().strip(), "latitude": float(self.lat_edit.text()), "longitude": float(self.lon_edit.text()), "elevation": float(self.elev_edit.text())}
        except ValueError:
            QMessageBox.warning(self, "信息无效", "纬度、经度和高程必须是数字。"); return
        self.pool.update(snapshot.snapshot_id, **changes)

    def _delete_selected(self) -> None:
        selected = self._selected()
        if not selected: return
        unexported = sum(not item.exported for item in selected); text = f"删除选中的 {len(selected)} 个临时结果？"
        if unexported: text += f"\n其中 {unexported} 个尚未输出为 EDI、Z 或 CSV。"
        if QMessageBox.question(self, "删除临时结果", text) == QMessageBox.StandardButton.Yes:
            for snapshot in selected: self.pool.remove_id(snapshot.snapshot_id)

    def _clear_pool(self) -> None:
        if not self.pool.items(): return
        count = self.pool.unexported_count(); text = "清空全部临时阻抗结果？" + (f"\n其中 {count} 个尚未输出为 EDI、Z 或 CSV。" if count else "")
        if QMessageBox.question(self, "清空临时结果", text) == QMessageBox.StandardButton.Yes: self.pool.clear()

    def _export_data(self, file_format: str) -> None:
        snapshot = self._single_selected()
        if snapshot is None or snapshot.bands is None: return
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in (snapshot.station_name or snapshot.name))
        suffix = {"edi": ".edi", "z": ".zrr" if snapshot.method == "remote reference" else ".zss", "csv": ".csv"}[file_format]
        filters = {"edi": "EDI 文件 (*.edi)", "z": "Z 文件 (*.zss *.zrr *.z)", "csv": "CSV 文件 (*.csv)"}[file_format]
        path, _ = QFileDialog.getSaveFileName(self, f"输出 {file_format.upper()}", str(Path.cwd() / f"{safe}{suffix}"), filters)
        if not path: return
        target = Path(path); target = target if target.suffix else target.with_suffix(suffix)
        kwargs = {"station_name": snapshot.station_name or "unknown", "remote_station_name": snapshot.remote_station_name, "mode": "rr" if snapshot.method == "remote reference" else "ss", "lat": snapshot.latitude, "lon": snapshot.longitude, "elev": snapshot.elevation}
        try:
            if file_format == "edi": target.write_text(snapshot.bands.format_edi_str(**kwargs), encoding="ascii")
            elif file_format == "z": target.write_text(snapshot.bands.format_z_str(**kwargs), encoding="ascii")
            else: target.write_text(banded_result_csv(snapshot.bands), encoding="utf-8", newline="")
        except Exception as exc:
            QMessageBox.critical(self, "输出失败", str(exc)); return
        self.pool.mark_exported(snapshot.snapshot_id, file_format)
        QMessageBox.information(self, "输出完成", f"已保存到：\n{target}")

    def _export_figure(self) -> None:
        selected = self._selected()
        if not selected: return
        if self.tabs.currentWidget() is self._impedance_page:
            figure, canvas, slug = self.main_figure, self.main_canvas, "impedance_compare"
        elif self.tabs.currentWidget() is self._tipper_page:
            mode = str(self.tipper_mode.currentData())
            figure, canvas, slug = self.tipper_figure, self.tipper_canvas, f"tipper_{mode}"
        elif self.tabs.currentWidget() is self._c_response_page:
            mode = str(self.c_response_mode.currentData())
            figure, canvas, slug = self.c_response_figure, self.c_response_canvas, f"c_response_{mode}"
        else: QMessageBox.information(self, "输出图像", "请先切换到视电阻率/相位图或倾子图。"); return
        export_matplotlib_figure_with_dialog(self, figure, canvas, default_directory=Path.cwd(), default_name=default_export_name(selected[0].station_name or "comparison", slug))

    def _reset_current_figure(self) -> None:
        canvas = {
            self._impedance_page: self.main_canvas,
            self._tipper_page: self.tipper_canvas,
            self._c_response_page: self.c_response_canvas,
        }.get(self.tabs.currentWidget())
        controller = getattr(canvas, "_mtpro_interaction_controller", None)
        if controller is not None:
            controller.reset()

    def closeEvent(self, event) -> None:
        if self._comparison_window is not None:
            self._comparison_window.close()
        super().closeEvent(event)
