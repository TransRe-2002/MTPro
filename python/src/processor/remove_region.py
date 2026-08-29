"""连续区域框选去噪工具。

提供两部分能力：
1. RegionPlotWidget:
   一个支持蓝色 LinearRegionItem 框选连续时间段的交互式绘图控件。
   区域内所有数据点将被标记为待删除。
2. RemoveRegion:
   把上面的控件包装成完整工具窗口，用于批量删除连续时间段的尖峰/脉冲/坏点。

与 RemoveSpike 不同，这个工具不需要逐个点选，适合快速处理大范围异常段。
"""

import os
import logging

import numpy as np
import xarray as xr
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore, QtGui
from typing import List, Optional, Any, Tuple

from processor._base_processor import build_pkg_from_series
from processor._display_utils import decimate_for_display, apply_nan_intervals
from processor._modifier_constraint import apply_modifier_axis_constraint
from utils.theme import configure_pyqtgraph_theme, pg_background

os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'

configure_pyqtgraph_theme()

logger = logging.getLogger(__name__)


class RegionPlotWidget(pg.PlotWidget):
    """连续区域选择的时序绘图控件。

    只支持通过蓝色 LinearRegionItem 框选连续时间段，
    不支持离散点框选。区域内所有数据点在删除时一次性置为 NaN。
    """

    selection_changed = QtCore.Signal(object)        # 拖动结束时发完整索引列表
    selection_count_changed = QtCore.Signal(int)      # 拖动中只发选中计数
    data_modified = QtCore.Signal(object)             # 已删除的索引数组
    data_restored = QtCore.Signal(object)             # 已恢复的索引数组
    data_reverted = QtCore.Signal()                   # 全部还原到原始数据

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, axisItems={'bottom': pg.DateAxisItem()})

        # 数据存储
        self.x_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None
        self.origin_y_data: Optional[np.ndarray] = None
        self._display_x: Optional[np.ndarray] = None
        self._display_y: Optional[np.ndarray] = None
        self.deletion_history: List[np.ndarray] = []
        self.deletion_history_ranges: List[List[Tuple[float, float]]] = []  # 每次删除的时间范围

        # 区域管理
        self._regions: List[pg.LinearRegionItem] = []
        self._plot_curve: Optional[pg.PlotCurveItem] = None
        self._current_cursor_shape = QtCore.Qt.CursorShape.ArrowCursor
        self._last_emitted_count: int = -1
        self.is_over_region: bool = False

        self._setup_view()

    # ─── 视图设置 ────────────────────────────────────────────────

    def _setup_view(self) -> None:
        self.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.getViewBox().setMouseEnabled(x=True, y=True)
        configure_pyqtgraph_theme(self)
        self.setBackground(pg_background(self))
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def _refresh_display_data(self) -> None:
        if self.x_data is None or self.y_data is None:
            return
        self._display_x, self._display_y = decimate_for_display(
            self.x_data, self.y_data,
        )
        if self._plot_curve is not None:
            self._plot_curve.setData(self._display_x, self._display_y)

    # ─── 数据管理 ────────────────────────────────────────────────

    def set_data(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        self.x_data = np.array(x_data, dtype=np.float64)
        self.origin_y_data = np.array(y_data, dtype=np.float64)
        self.y_data = self.origin_y_data.copy()

        self._clear_all()

        self._display_x, self._display_y = decimate_for_display(
            self.x_data, self.y_data,
        )
        self._plot_curve = pg.PlotCurveItem(
            self._display_x, self._display_y, pen='b',
            autoDownsample=True, downsampleMethod='peak',
            clipToView=True, connect='finite',
        )
        self.addItem(self._plot_curve)

    def _clear_all(self) -> None:
        self.clear()
        self._regions.clear()
        self._plot_curve = None
        self._display_x = None
        self._display_y = None
        self.deletion_history.clear()
        self.deletion_history_ranges.clear()
        self._last_emitted_count = -1

    def revert_to_original_data(self) -> None:
        if self.origin_y_data is not None and self.x_data is not None:
            self.y_data = self.origin_y_data.copy()
            self._clear_all()
            self._display_x, self._display_y = decimate_for_display(
                self.x_data, self.y_data,
            )
            self._plot_curve = pg.PlotCurveItem(
                self._display_x, self._display_y, pen='b',
                autoDownsample=True, downsampleMethod='peak',
                clipToView=True, connect='finite',
            )
            self.addItem(self._plot_curve)
            self._on_region_changed()
            self.data_reverted.emit()

    def confirm_and_exit(self) -> np.ndarray:
        if self.y_data is not None:
            return self.y_data
        return np.array([])

    # ─── 区域管理 ────────────────────────────────────────────────

    def set_selection_mode(self, mode: str) -> None:
        """兼容接口：RegionPlotWidget 不需要模式切换，空实现。"""
        pass

    def add_region(self, start_x: Optional[float] = None) -> None:
        if self.x_data is None or len(self.x_data) == 0:
            return

        view_box = self.getViewBox()
        current_view_range = view_box.viewRange()
        view_x_min, view_x_max = current_view_range[0]
        visible_range_width = view_x_max - view_x_min
        dynamic_region_width = visible_range_width * 0.1

        if start_x is not None:
            start_x = max(self.x_data.min(), min(self.x_data.max(), start_x))
            region_start = start_x
            region_end = min(self.x_data.max(), start_x + dynamic_region_width)
        else:
            data_range = self.x_data.max() - self.x_data.min()
            region_start = self.x_data.min() + 0.4 * data_range
            region_end = self.x_data.min() + 0.6 * data_range

        min_region_width = visible_range_width * 0.02
        if region_end - region_start < min_region_width:
            region_end = region_start + min_region_width

        region = pg.LinearRegionItem(values=[region_start, region_end], movable=True)
        border_pen = pg.mkPen(color=(65, 105, 225), width=4)
        region.lines[0].setPen(border_pen)
        region.lines[1].setPen(border_pen)
        region.setBrush(pg.mkBrush(135, 206, 250, 80))

        self.addItem(region)
        self._regions.append(region)
        region.sigRegionChanged.connect(self._on_region_changed_lightweight)
        region.sigRegionChangeFinished.connect(self._on_region_changed)

        self._on_region_changed()

    def remove_selected_region(self) -> None:
        if not self._regions:
            return
        region = self._regions.pop()
        self.removeItem(region)
        self._on_region_changed()

    def clear_all_selections(self) -> None:
        for region in self._regions[:]:
            self.removeItem(region)
        self._regions.clear()
        self._on_region_changed()

    def get_region_time_ranges(self) -> List[Tuple[float, float]]:
        """返回所有蓝色区域的时间范围列表 [(start_ts, end_ts), ...]。"""
        ranges = []
        for region in self._regions:
            try:
                xmin, xmax = region.getRegion()
                ranges.append((float(xmin), float(xmax)))
            except Exception as e:
                logger.warning("获取区域时间范围出错: %s", e)
        return ranges

    # ─── 区间计算 ────────────────────────────────────────────────

    def _get_selected_intervals(self) -> List[Tuple[int, int]]:
        if self.x_data is None or len(self.x_data) == 0:
            return []

        intervals: List[Tuple[int, int]] = []
        for region in self._regions:
            try:
                xmin, xmax = region.getRegion()
                left = int(np.searchsorted(self.x_data, xmin, side='left'))
                right = int(np.searchsorted(self.x_data, xmax, side='right'))
                if right > left:
                    intervals.append((left, right))
            except Exception as e:
                logger.warning("计算区域索引时出错: %s", e)
                continue

        if not intervals:
            return []

        intervals.sort()
        merged: List[Tuple[int, int]] = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def _get_selected_count(self) -> int:
        intervals = self._get_selected_intervals()
        return sum(end - start for start, end in intervals)

    def _get_selected_indices(self) -> np.ndarray:
        intervals = self._get_selected_intervals()
        if not intervals:
            return np.array([], dtype=np.int64)
        arrays = [np.arange(start, end, dtype=np.int64) for start, end in intervals]
        return np.concatenate(arrays)

    def _on_region_changed_lightweight(self) -> None:
        count = self._get_selected_count()
        if count != self._last_emitted_count:
            self._last_emitted_count = count
            self.selection_count_changed.emit(count)

    def _on_region_changed(self) -> None:
        selected_indices = self._get_selected_indices()
        self._last_emitted_count = len(selected_indices)
        self.selection_changed.emit(selected_indices)

    # ─── 鼠标事件（简化版：无框选矩形）─────────────────────────

    def _is_mouse_over_region(self, pos: QtCore.QPointF) -> bool:
        view = self.getViewBox()
        if view is None or not self._regions:
            return False
        try:
            scene_pos = view.mapSceneToView(pos)
            for region in self._regions:
                rgn = region.getRegion()
                if rgn[0] <= scene_pos.x() <= rgn[1]:
                    return True
                view_range = view.viewRange()[0]
                tolerance = (view_range[1] - view_range[0]) * 0.03
                if (abs(scene_pos.x() - rgn[0]) < tolerance
                        or abs(scene_pos.x() - rgn[1]) < tolerance):
                    return True
        except Exception:
            pass
        return False

    def _is_mouse_over_auto_scale_button(self, pos: QtCore.QPointF) -> bool:
        try:
            return QtCore.QRectF(1, self.height() - 31, 30, 30).contains(pos)
        except Exception:
            return False

    def _to_scene(self, ev: QtGui.QMouseEvent) -> QtCore.QPointF:
        p = ev.position()
        return self.mapToScene(int(p.x()), int(p.y()))

    def _apply_modifier_axis_constraint(self, modifiers) -> None:
        apply_modifier_axis_constraint(self.getViewBox(), modifiers)

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        modifiers = ev.modifiers()
        vb = self.getViewBox()
        if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            vb.setMouseEnabled(x=True, y=False)
        elif modifiers & QtCore.Qt.KeyboardModifier.AltModifier:
            vb.setMouseEnabled(x=False, y=True)
        else:
            vb.setMouseEnabled(x=True, y=True)
        super().wheelEvent(ev)
        vb.setMouseEnabled(x=True, y=True)

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._is_mouse_over_auto_scale_button(ev.position()):
            super().mousePressEvent(ev)
            return

        scene_pos = self._to_scene(ev)
        self.is_over_region = self._is_mouse_over_region(scene_pos)

        if ev.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._apply_modifier_axis_constraint(ev.modifiers())
            super().mousePressEvent(ev)

        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            view = self.getViewBox()
            if view is not None and self.x_data is not None:
                try:
                    pos = view.mapSceneToView(scene_pos)
                    for region in reversed(self._regions):
                        rgn = region.getRegion()
                        view_range = view.viewRange()[0]
                        tolerance = (view_range[1] - view_range[0]) * 0.01
                        if (rgn[0] <= pos.x() <= rgn[1]
                                or abs(pos.x() - rgn[0]) < tolerance
                                or abs(pos.x() - rgn[1]) < tolerance):
                            self._regions.remove(region)
                            self.removeItem(region)
                            self._on_region_changed()
                            ev.accept()
                            return
                except Exception as e:
                    logger.warning("右键处理错误: %s", e)
            ev.accept()
            return

        elif ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.is_over_region:
                super().mousePressEvent(ev)  # 允许拖动区域边界
                return
            super().mousePressEvent(ev)  # 否则平移
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        scene_pos = self._to_scene(ev)
        self.is_over_region = self._is_mouse_over_region(scene_pos)

        new_cursor = (
            QtCore.Qt.CursorShape.SizeHorCursor if self.is_over_region
            else QtCore.Qt.CursorShape.ArrowCursor
        )
        if new_cursor != self._current_cursor_shape:
            self.setCursor(QtGui.QCursor(new_cursor))
            self._current_cursor_shape = new_cursor

        if ev.buttons() & QtCore.Qt.MouseButton.MiddleButton:
            self._apply_modifier_axis_constraint(ev.modifiers())

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._is_mouse_over_auto_scale_button(ev.position()):
            super().mouseReleaseEvent(ev)
            return
        if ev.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.getViewBox().setMouseEnabled(x=True, y=True)
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._is_mouse_over_auto_scale_button(ev.position()):
            super().mouseDoubleClickEvent(ev)
            return
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            scene_pos = self._to_scene(ev)
            view_box = self.getViewBox()
            if view_box is not None and self.x_data is not None:
                try:
                    data_pos = view_box.mapSceneToView(scene_pos)
                    click_x = data_pos.x()
                    if self.x_data.min() <= click_x <= self.x_data.max():
                        self.add_region(start_x=click_x)
                        ev.accept()
                        return
                except Exception as e:
                    logger.warning("双击创建区域错误: %s", e)
        super().mouseDoubleClickEvent(ev)

    # ─── 数据修改 ────────────────────────────────────────────────

    def modify_selected_data(self) -> None:
        if self.x_data is None or self.y_data is None:
            return

        intervals = self._get_selected_intervals()
        if not intervals:
            return

        idx_arrays = [np.arange(start, end, dtype=np.int64) for start, end in intervals]
        all_indices = np.concatenate(idx_arrays) if idx_arrays else np.array([], dtype=np.int64)

        for start, end in intervals:
            self.y_data[start:end] = np.nan

        self.deletion_history.append(all_indices)
        # 保存时间范围，然后立刻清除区域
        self.deletion_history_ranges.append(self.get_region_time_ranges())
        for region in self._regions:
            self.removeItem(region)
        self._regions.clear()

        self._refresh_display_data()
        self._on_region_changed()
        self.data_modified.emit(all_indices)

    def undo_last_deletion(self) -> None:
        if not self.deletion_history or self.x_data is None or self.y_data is None:
            return

        last_deletion = self.deletion_history.pop()
        ranges = self.deletion_history_ranges.pop() if self.deletion_history_ranges else []

        if self.origin_y_data is not None:
            self.y_data[last_deletion] = self.origin_y_data[last_deletion]

        # 用保存的时间范围重建区域
        for start_ts, end_ts in ranges:
            self._add_region_at(start_ts, end_ts)

        self._refresh_display_data()
        self._on_region_changed()
        self.data_restored.emit(last_deletion)

    def _add_region_at(self, start_x: float, end_x: float) -> None:
        """用精确边界重建 LinearRegionItem。"""
        region = pg.LinearRegionItem(values=[start_x, end_x], movable=True)
        border_pen = pg.mkPen(color=(65, 105, 225), width=4)
        region.lines[0].setPen(border_pen)
        region.lines[1].setPen(border_pen)
        region.setBrush(pg.mkBrush(135, 206, 250, 80))
        self.addItem(region)
        self._regions.append(region)
        region.sigRegionChanged.connect(self._on_region_changed_lightweight)
        region.sigRegionChangeFinished.connect(self._on_region_changed)


class RemoveRegion(QtWidgets.QWidget):
    """连续区域框选去噪的完整工具窗口。

    通过蓝色 LinearRegionItem 框选连续时间段，
    一键将区域内所有数据点置为 NaN。

    不持有 Broker 引用——通过信号与外部通信。
    """

    # 工具信号（data_process 接线到 Broker）
    data_requested = QtCore.Signal(int, str)
    result_submitted = QtCore.Signal(int, str, object)
    result_signal = QtCore.Signal(str, object)
    channel_sync_requested = QtCore.Signal(int, str, object)
    region_sync_requested = QtCore.Signal(object)

    def __init__(
        self,
        data_id: int,
        channel_name: str,
        parent=None,
    ):
        super().__init__(parent)

        if self.parent() is not None:
            self.setWindowFlag(QtCore.Qt.WindowType.Window)

        self.data_id = data_id
        self.ch_name = channel_name

        # 待 _on_data_ready 填充
        self.x_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None
        self.all_channels: list[str] = []
        self.origin_all_channels: dict[str, np.ndarray] = {}
        self._kp_times: Optional[np.ndarray] = None

        self.plot_widget: Optional[RegionPlotWidget] = None
        self.plot_item = None
        self._kp_viewbox: Optional[pg.ViewBox] = None
        self._kp_bars: list = []
        self._kp_auto_restore = False
        self._kp_density_timer = QtCore.QTimer(self)
        self._kp_density_timer.setSingleShot(True)
        self._kp_density_timer.setInterval(150)
        self._kp_density_timer.timeout.connect(self._check_kp_density)

        self.status_label = QtWidgets.QLabel("当前选中: 0 个数据点")
        self.instruction_label = QtWidgets.QLabel(
            "操作说明: 左键双击创建区域 | 拖动区域边界调整范围 | 右键点击区域删除 | "
            "中键移动视图 | Ctrl+Z撤销删除 | Ctrl+S保存 | Delete快捷删除\n"
            "         Shift+滚轮/中键拖动 → 约束X轴 | Alt+滚轮/中键拖动 → 约束Y轴"
        )

        self._init_ui()
        self._connect_signals()
        self._init_shortcuts()

    def wire_data_ready(self, data_ready_signal) -> None:
        """连接外部数据信号。由 data_process 调。"""
        data_ready_signal.connect(self._on_data_ready)

    def load_series(self, channel_name: str, series) -> None:
        """以 ``pd.Series`` 切片直接注入数据（pipeline block 使用）。"""
        self.ch_name = channel_name
        pkg = build_pkg_from_series(channel_name, series)
        self._on_data_ready(self.data_id, channel_name, pkg)

    def _on_data_ready(self, data_id: int, channel: str, pkg: dict) -> None:
        if data_id != self.data_id or channel != self.ch_name:
            return
        self._pkg = pkg
        self.x_data = pkg['x']
        self.y_data = pkg['y']
        self.all_channels = pkg['channel_names']
        self.origin_all_channels = pkg['all_y']
        self._init_plot()

    def _init_shortcuts(self):
        delete_sc = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), self)
        delete_sc.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        delete_sc.activated.connect(self.plot_widget.modify_selected_data)

        save_sc = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        save_sc.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        save_sc.activated.connect(self._confirm_and_close)

        undo_sc = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self)
        undo_sc.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        undo_sc.activated.connect(self.plot_widget.undo_last_deletion)

    def _init_ui(self):
        self.plot_widget = RegionPlotWidget()
        self.plot_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.plot_widget.getAxis('left').setStyle(tickLength=0)
        self.plot_widget.getAxis('bottom').setStyle(tickLength=0)
        self.plot_item = self.plot_widget.getPlotItem()

        btn_layout = QtWidgets.QHBoxLayout()
        self.add_region_btn = QtWidgets.QPushButton("添加连续区域")
        self.remove_region_btn = QtWidgets.QPushButton("移除最后区域")
        self.revert_btn = QtWidgets.QPushButton("还原原始数据")
        self.clear_all_btn = QtWidgets.QPushButton("清空所有区域")
        self.modify_data_btn = QtWidgets.QPushButton("删除选中数据")
        self.undo_btn = QtWidgets.QPushButton("撤销上一次删除")
        self.confirm_btn = QtWidgets.QPushButton("确定")

        btn_layout.addWidget(self.add_region_btn)
        btn_layout.addWidget(self.remove_region_btn)
        btn_layout.addWidget(self.revert_btn)
        btn_layout.addWidget(self.clear_all_btn)
        btn_layout.addWidget(self.modify_data_btn)
        btn_layout.addWidget(self.undo_btn)
        btn_layout.addStretch(1)
        self.sync_checkbox = QtWidgets.QCheckBox("应用到所有通道")
        self.sync_checkbox.setChecked(False)
        self.sync_checkbox.setToolTip("确定时将当前通道的删除点同步到所有其他通道")
        btn_layout.addWidget(self.sync_checkbox)
        self.all_data_checkbox = QtWidgets.QCheckBox("应用到所有数据")
        self.all_data_checkbox.setChecked(False)
        self.all_data_checkbox.setToolTip("确定时将删除操作应用到所有已加载的数据集")
        btn_layout.addWidget(self.all_data_checkbox)
        btn_layout.addWidget(self.confirm_btn)

        glay = QtWidgets.QVBoxLayout()
        glay.addWidget(self.status_label)
        glay.addWidget(self.instruction_label)

        self.kp_checkbox = QtWidgets.QCheckBox("显示Kp")
        self.kp_checkbox.setChecked(False)
        kp_row = QtWidgets.QHBoxLayout()
        kp_row.addWidget(self.kp_checkbox)
        kp_row.addStretch(1)
        glay.addLayout(kp_row)

        glay.addWidget(self.plot_widget)
        glay.addLayout(btn_layout)
        self.setLayout(glay)

    def _connect_signals(self):
        self.add_region_btn.clicked.connect(self.plot_widget.add_region)
        self.remove_region_btn.clicked.connect(self.plot_widget.remove_selected_region)
        self.revert_btn.clicked.connect(self.plot_widget.revert_to_original_data)
        self.undo_btn.clicked.connect(self.plot_widget.undo_last_deletion)
        self.clear_all_btn.clicked.connect(self._clear_all_regions)
        self.modify_data_btn.clicked.connect(self.plot_widget.modify_selected_data)
        self.confirm_btn.clicked.connect(self._confirm_and_close)
        self.kp_checkbox.toggled.connect(self._toggle_kp)
        self.all_data_checkbox.toggled.connect(self._on_all_data_toggled)

        self.plot_widget.selection_count_changed.connect(self._on_selection_count_changed)

    def _on_all_data_toggled(self, checked: bool) -> None:
        """勾选"应用到所有数据"时自动勾选并禁用"应用到所有通道"."""
        if checked:
            self.sync_checkbox.setChecked(True)
            self.sync_checkbox.setEnabled(False)
        else:
            self.sync_checkbox.setEnabled(True)

    def _init_plot(self):
        if self.x_data is None or self.y_data is None:
            self._add_text_placeholder("请先加载数据")
            return

        self.plot_widget.set_data(self.x_data, self.y_data)

        grid_pen = pg.mkPen(
            color=pg.mkColor(100, 100, 100), width=1, alpha=1,
            antialiased=True, style=QtCore.Qt.PenStyle.DashLine,
        )
        self.plot_item.showGrid(x=True, y=True, alpha=0.7)
        self.plot_item.getAxis('bottom').setPen(grid_pen)
        self.plot_item.getAxis('left').setPen(grid_pen)

        kp_data = self._pkg.get('kp_data')
        if kp_data is None or (hasattr(kp_data, 'empty') and kp_data.empty):
            self.kp_checkbox.setEnabled(False)
            self.kp_checkbox.setToolTip("无可用 Kp 数据")
        elif kp_data is not None:
            self._kp_times = kp_data['Kp_datetime'].apply(
                lambda t: t.timestamp()
            ).to_numpy(dtype=np.float64)
            QtCore.QTimer.singleShot(0, self._check_kp_density)
            self.plot_item.vb.sigRangeChanged.connect(self._on_view_range_changed)

    def showEvent(self, ev: QtGui.QShowEvent) -> None:
        super().showEvent(ev)
        self.activateWindow()
        self.plot_widget.setFocus()

    def _clear_all_regions(self):
        self.plot_widget.clear_all_selections()

    def _on_selection_count_changed(self, count: int):
        self.status_label.setText(f"当前选中: {count} 个数据点")

    def _add_text_placeholder(self, text):
        self.plot_item.clear()
        text_item = pg.TextItem(text, color='k', anchor=(0.5, 0.5))
        self.plot_item.addItem(text_item)
        text_item.setPos(0, 0)

    # ─── Kp 柱状图 ────────────────────────────────────────────────

    def _toggle_kp(self, checked: bool) -> None:
        if checked:
            self._kp_auto_restore = True
            if self._kp_density_exceeds():
                self._set_kp_state(auto_off=True)
            else:
                self._show_kp()
        else:
            self._hide_kp()
            self._kp_auto_restore = False
            self.kp_checkbox.setEnabled(True)

    def _set_kp_state(self, auto_off: bool) -> None:
        if auto_off:
            self.kp_checkbox.blockSignals(True)
            self.kp_checkbox.setChecked(False)
            self.kp_checkbox.blockSignals(False)
            self.kp_checkbox.setEnabled(False)
        else:
            self.kp_checkbox.blockSignals(True)
            self.kp_checkbox.setChecked(True)
            self.kp_checkbox.blockSignals(False)
            self.kp_checkbox.setEnabled(True)

    def _show_kp(self) -> None:
        if self._kp_viewbox is not None:
            return
        kp_data = self._pkg.get('kp_data')
        if kp_data is None or (hasattr(kp_data, 'empty') and kp_data.empty):
            self.kp_checkbox.setChecked(False)
            return

        kp_values = kp_data['Kp'].to_numpy(dtype=np.float64)
        bar_width = 3 * 3600

        self.plot_widget.showAxis('right')
        self.plot_widget.setLabel('right', 'Kp')

        self._kp_viewbox = pg.ViewBox()
        self.plot_item.scene().addItem(self._kp_viewbox)
        self.plot_item.getAxis('right').linkToView(self._kp_viewbox)
        self._kp_viewbox.setXLink(self.plot_item)
        self.plot_item.vb.sigResized.connect(self._update_kp_geometry)
        self._update_kp_geometry()

        groups = [
            ('blue', kp_values < 4, pg.mkBrush(30, 144, 255, 180),
             pg.mkPen(color=(30, 144, 255, 180), width=1)),
            ('yellow', (kp_values >= 4) & (kp_values < 5), pg.mkBrush(255, 200, 0, 180),
             pg.mkPen(color=(255, 200, 0, 180), width=1)),
            ('red', kp_values >= 5, pg.mkBrush(255, 60, 60, 180),
             pg.mkPen(color=(255, 60, 60, 180), width=1)),
        ]
        self._kp_bars = []
        for _label, mask, brush, pen in groups:
            if mask.any():
                bar = pg.BarGraphItem(
                    x0=self._kp_times[mask] - bar_width, height=kp_values[mask],
                    width=bar_width, brush=brush, pen=pen,
                )
                self._kp_viewbox.addItem(bar)
                self._kp_bars.append(bar)

        self._kp_viewbox.setYRange(0, 9, padding=0.02)
        self._kp_viewbox.setMouseEnabled(x=False, y=True)

    def _hide_kp(self) -> None:
        if self._kp_viewbox is None:
            return
        self.plot_item.vb.sigResized.disconnect(self._update_kp_geometry)
        self._kp_density_timer.stop()
        for bar in self._kp_bars:
            self._kp_viewbox.removeItem(bar)
        self._kp_bars.clear()
        self.plot_item.scene().removeItem(self._kp_viewbox)
        self._kp_viewbox = None
        self.plot_widget.hideAxis('right')

    def _update_kp_geometry(self) -> None:
        if self._kp_viewbox is not None:
            self._kp_viewbox.setGeometry(self.plot_item.vb.sceneBoundingRect())

    def _on_view_range_changed(self, _vb, range_info) -> None:
        if self._kp_auto_restore or self._kp_viewbox is not None or not self.kp_checkbox.isEnabled():
            self._kp_density_timer.start()

    def _check_kp_density(self) -> None:
        exceeds = self._kp_density_exceeds()
        kp_visible = self._kp_viewbox is not None
        if kp_visible and exceeds:
            self._hide_kp()
            self._set_kp_state(auto_off=True)
        elif not kp_visible and exceeds and not self._kp_auto_restore:
            self.kp_checkbox.setEnabled(False)
            self.kp_checkbox.setToolTip("Kp 数据过密，需放大视口")
        elif not kp_visible and not exceeds:
            if self._kp_auto_restore:
                self._show_kp()
                self._set_kp_state(auto_off=False)
            elif not self.kp_checkbox.isEnabled():
                self.kp_checkbox.setEnabled(True)
                self.kp_checkbox.setToolTip("")

    def _kp_density_exceeds(self) -> bool:
        if self._kp_times is None:
            return False
        vb = self.plot_item.vb
        x_min, x_max = vb.viewRange()[0]
        visible = int(np.sum((self._kp_times >= x_min) & (self._kp_times <= x_max)))
        if visible == 0:
            return False
        px_width = max(vb.width(), 1)
        return visible > px_width / 3

    # ─── 跨通道同步 ──────────────────────────────────────────────

    def _confirm_and_close(self):
        result = self.plot_widget.confirm_and_exit()
        if result.size == 0:
            return

        logger.info("RemoveRegion confirmed, returning %s points", len(result))

        # "应用到所有数据" → 发送本次删除的时间范围
        if self.all_data_checkbox.isChecked():
            time_ranges = self.plot_widget.deletion_history_ranges[-1] if self.plot_widget.deletion_history_ranges else []
            if time_ranges:
                self.region_sync_requested.emit({
                    "time_ranges": time_ranges,
                    "source_data_id": self.data_id,
                })

        # "应用到所有通道" → 信号
        if self.sync_checkbox.isChecked():
            self.channel_sync_requested.emit(self.data_id, self.ch_name, result)

        # 当前通道结果 → 信号
        self.result_submitted.emit(self.data_id, self.ch_name, result)
        self.result_signal.emit(self.ch_name, result)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    from io_utils.mat_io import MatLoader
    em_data = MatLoader().load(
        "/home/transen5/Project/atm_rpc/039BE-20240501-20240515-dt5_struct.mat"
    )
    app = QApplication([])
    window = RemoveRegion(em_data=em_data, channel_name='Ex1')
    window.show()
    app.exec()
