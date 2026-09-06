"""Temporary impedance result pool and comparison preview."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget


@dataclass(frozen=True)
class ImpedanceSnapshot:
    name: str
    method: str
    result: object
    created_at: datetime


class ImpedanceResultPool(QObject):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[ImpedanceSnapshot] = []

    def add(self, method: str, result: object, name: str | None = None) -> ImpedanceSnapshot:
        stamp = datetime.now().astimezone()
        snapshot = ImpedanceSnapshot(name or f"{method} {stamp:%H:%M:%S}", method, deepcopy(result), stamp)
        self._items.append(snapshot)
        self.changed.emit()
        return snapshot

    def remove(self, index: int) -> None:
        if 0 <= index < len(self._items):
            self._items.pop(index)
            self.changed.emit()

    def clear(self) -> None:
        self._items.clear()
        self.changed.emit()

    def items(self) -> tuple[ImpedanceSnapshot, ...]:
        return tuple(self._items)


class ImpedancePreviewWidget(QWidget):
    def __init__(self, pool: ImpedanceResultPool, parent=None):
        super().__init__(parent)
        self.pool = pool
        self.list = QListWidget(self)
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list.itemSelectionChanged.connect(self._refresh_plot)
        self.plot = pg.PlotWidget(self)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "周期 (s)")
        self.plot.setLabel("left", "视电阻率")
        self.method_filter = QComboBox(self)
        self.method_filter.addItems(["全部", "Robust 估计", "远参考法"])
        self.method_filter.setItemData(1, "robust estimate")
        self.method_filter.setItemData(2, "remote reference")
        self.method_filter.currentIndexChanged.connect(self._refresh_list)
        clear = QPushButton("清空临时池", self)
        clear.clicked.connect(self.pool.clear)
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("结果池："))
        toolbar.addWidget(self.method_filter)
        toolbar.addStretch(1)
        toolbar.addWidget(clear)
        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        body = QHBoxLayout()
        body.addWidget(self.list, 0)
        body.addWidget(self.plot, 1)
        layout.addLayout(body, 1)
        self.pool.changed.connect(self._refresh_list)
        self._refresh_list()

    def _refresh_list(self) -> None:
        selected_names = {item.data( Qt.ItemDataRole.UserRole) for item in self.list.selectedItems()}
        self.list.clear()
        method = self.method_filter.currentData() or "全部"
        for snapshot in self.pool.items():
            if method != "全部" and snapshot.method != method:
                continue
            item = QListWidgetItem(f"{snapshot.name}  [{snapshot.method}]", self.list)
            item.setData(Qt.ItemDataRole.UserRole, snapshot.name)
            item.setSelected(snapshot.name in selected_names)
        self._refresh_plot()

    def _refresh_plot(self) -> None:
        self.plot.clear()
        colors = [(220, 80, 70), (70, 130, 220), (80, 170, 100), (180, 100, 200)]
        by_name = {item.name: item for item in self.pool.items()}
        for index, selected in enumerate(self.list.selectedItems()):
            snapshot = by_name.get(selected.data(Qt.ItemDataRole.UserRole))
            if snapshot is None:
                continue
            result = snapshot.result
            periods = np.asarray(getattr(result, "periods", []), dtype=float)
            rho = np.asarray(getattr(result, "rho_xy", []), dtype=float)
            mask = np.isfinite(periods) & (periods > 0) & np.isfinite(rho)
            if mask.any():
                self.plot.plot(periods[mask], rho[mask], pen=pg.mkPen(colors[index % len(colors)], width=2), name=snapshot.name)
        self.plot.setLogMode(x=True, y=True)
