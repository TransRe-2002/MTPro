from __future__ import annotations

"""线性区域选择混入类，提供基于 LinearRegionItem 的区域管理功能。

需要混入的类必须提供以下属性：
- signal_plot_item: pg.PlotItem
- x_data: np.ndarray
- region_item: Optional[pg.LinearRegionItem]
- selected_indices: list[int]
- selection_status_label: QLabel
"""

from typing import Optional

import numpy as np
import pyqtgraph as pg


class RegionManagementMixin:
    """为处理器窗口提供统一的 LinearRegionItem 区域选择交互。"""

    # 子类必须提供的属性声明（类型标注仅作文档用）
    signal_plot_item: pg.PlotItem
    x_data: np.ndarray
    region_item: Optional[pg.LinearRegionItem]
    selected_indices: list[int]
    selection_status_label: object  # QLabel

    def _on_region_changed(self) -> None:
        """子类可重写的钩子，在区域选择变更后调用。"""
        pass

    def _current_view_region_bounds(self) -> tuple[float, float]:
        if self.x_data.size == 0:
            return 0.0, 1.0
        view_min, view_max = self.signal_plot_item.vb.viewRange()[0]
        data_min = float(self.x_data[0])
        data_max = float(self.x_data[-1])
        view_min = max(data_min, min(view_min, data_max))
        view_max = max(view_min, min(view_max, data_max))
        if view_max <= view_min:
            view_min = data_min
            view_max = data_max
        return float(view_min), float(view_max)

    def _region_bounds_around(self, center_x: Optional[float] = None) -> tuple[float, float]:
        view_min, view_max = self._current_view_region_bounds()
        visible_width = max(view_max - view_min, 1.0)
        region_width = max(1.0, visible_width * 0.2)
        if center_x is None:
            center_x = 0.5 * (view_min + view_max)
        center_x = max(view_min, min(center_x, view_max))
        half_width = 0.5 * region_width
        left = center_x - half_width
        right = center_x + half_width
        if left < view_min:
            right += view_min - left
            left = view_min
        if right > view_max:
            left -= right - view_max
            right = view_max
        left = max(view_min, left)
        right = min(view_max, right)
        if right <= left:
            right = min(view_max, left + 1.0)
        return float(left), float(right)

    def _add_region(self):
        self._set_region_bounds(*self._region_bounds_around())

    def _add_or_move_region_from_x(self, center_x: float):
        self._set_region_bounds(*self._region_bounds_around(center_x))

    def _set_region_bounds(self, left: float, right: float):
        if self.region_item is None:
            self.region_item = pg.LinearRegionItem(values=[left, right], movable=True)
            border_pen = pg.mkPen(color=(65, 105, 225), width=1)
            self.region_item.lines[0].setPen(border_pen)
            self.region_item.lines[1].setPen(border_pen)
            self.region_item.setBrush(pg.mkBrush(135, 206, 250, 70))
            self.signal_plot_item.addItem(self.region_item)
            self.region_item.sigRegionChanged.connect(self._update_region_selection)
            self.region_item.sigRegionChangeFinished.connect(self._update_region_selection)
        else:
            self.region_item.blockSignals(True)
            self.region_item.setRegion((left, right))
            self.region_item.blockSignals(False)
        self._update_region_selection()

    def _remove_region(self):
        if self.region_item is None:
            return
        self.signal_plot_item.removeItem(self.region_item)
        self.region_item = None
        self.selected_indices = []
        self._update_status_label(0)

    def _handle_plot_right_click(self, x_value: float, y_value: float) -> bool:
        del y_value
        if self.region_item is None:
            return False
        left, right = sorted(self.region_item.getRegion())
        if left <= x_value <= right:
            self._remove_region()
            return True
        return False

    def _update_region_selection(self):
        if self.region_item is None:
            self.selected_indices = []
            self._update_status_label(0)
            self._on_region_changed()
            return
        left_value, right_value = sorted(self.region_item.getRegion())
        left = int(np.searchsorted(self.x_data, left_value, side="left"))
        right = int(np.searchsorted(self.x_data, right_value, side="right") - 1)
        left = max(0, min(left, self.x_data.size - 2))
        right = max(left + 1, min(right, self.x_data.size - 1))
        self.selected_indices = list(range(left, right + 1))
        self._update_status_label(len(self.selected_indices))
        self._on_region_changed()

    def _update_status_label(self, count: int) -> None:
        """更新选中状态标签。子类可重写以自定义文本格式。"""
        self.selection_status_label.setText(f"当前选中: {count} 个采样点")
