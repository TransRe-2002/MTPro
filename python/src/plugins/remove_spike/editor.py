"""离散点框选去噪工具。

提供两部分能力：
1. SpikePlotWidget:
   一个支持矩形框选离散点的交互式绘图控件。
   左键框选添加点，右键框选取消点，选中点以红色小圆点标记。
2. RemoveSpike:
   把上面的控件包装成完整工具窗口，用于逐个删除尖峰、脉冲等孤立坏点。

与 RemoveRegion 不同，这个工具适合精确框选少量异常点，不适合大范围连续删除。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from plugins.common.base_processor import build_pkg_from_series
from plugins.common.display import decimate_for_display
from plugins.common.modifier_constraint import apply_modifier_axis_constraint
from utils.theme import configure_pyqtgraph_theme, pg_background

configure_pyqtgraph_theme()

logger = logging.getLogger(__name__)


class SpikePlotWidget(pg.PlotWidget):
    """离散点框选的时序绘图控件。

    只支持矩形框选离散点（红色小圆点标记），
    不支持 LinearRegionItem 连续区域操作。
    两种模式：框选放大（zoom）和框选选点（select）。
    """

    selection_changed = QtCore.Signal(int)  # 选中点数变化（轻量）

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, axisItems={'bottom': pg.DateAxisItem()})

        # 数据存储
        self.x_data: np.ndarray | None = None
        self.y_data: np.ndarray | None = None
        self.origin_y_data: np.ndarray | None = None
        self._display_x: np.ndarray | None = None
        self._display_y: np.ndarray | None = None
        self.undo_stack: list[np.ndarray] = []  # 每次删除的索引数组

        # 离散选点
        self.selected_points: set[int] = set()

        # 状态
        self.selection_mode: str = 'zoom'  # 'zoom' | 'select'
        self.is_selecting: bool = False
        self.start_point: QtCore.QPointF | None = None
        self.end_point: QtCore.QPointF | None = None

        # 图形项
        self.selection_rect_item: QtWidgets.QGraphicsRectItem | None = None
        self._highlight_scatter: pg.ScatterPlotItem | None = None
        self._highlight_size: int = 5
        self._plot_curve: pg.PlotCurveItem | None = None
        self._rect_color: QtGui.QColor = QtGui.QColor(30, 144, 255, 180)

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
        self._highlight_scatter = None
        self._plot_curve = None
        self._display_x = None
        self._display_y = None
        self.selected_points.clear()
        self.undo_stack.clear()

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

    def confirm_and_exit(self) -> np.ndarray:
        if self.y_data is not None:
            return self.y_data
        return np.array([])

    def set_selection_mode(self, mode: str) -> None:
        self.selection_mode = mode
        self._clear_selection_rect()

    # ─── 高亮（红色小圆点）─────────────────────────────────────

    def _clear_highlights(self) -> None:
        if self._highlight_scatter is not None:
            self.removeItem(self._highlight_scatter)
            self._highlight_scatter = None

    def _update_highlights(self) -> None:
        self._clear_highlights()
        if not self.selected_points or self.x_data is None or self.y_data is None:
            return

        idx_arr = np.fromiter(self.selected_points, dtype=np.int64,
                              count=len(self.selected_points))
        # 过滤越界 + NaN
        valid_mask = (idx_arr >= 0) & (idx_arr < len(self.x_data)) & (idx_arr < len(self.y_data))
        idx_arr = idx_arr[valid_mask]
        if len(idx_arr) == 0:
            return

        nan_mask = ~np.isnan(self.y_data[idx_arr])
        valid = idx_arr[nan_mask]
        if len(valid) == 0:
            return

        self._highlight_scatter = pg.ScatterPlotItem(
            x=self.x_data[valid],
            y=self.y_data[valid],
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 60, 60, 200),
            size=self._highlight_size,
            symbol='o',
        )
        self.addItem(self._highlight_scatter)

    # ─── 数据修改 ────────────────────────────────────────────────

    def modify_selected_data(self) -> None:
        if self.x_data is None or self.y_data is None or not self.selected_points:
            return

        idx_arr = np.fromiter(self.selected_points, dtype=np.int64,
                              count=len(self.selected_points))
        self.y_data[idx_arr] = np.nan

        self.undo_stack.append(idx_arr)
        self.selected_points.clear()
        self._clear_highlights()
        self._refresh_display_data()
        self.selection_changed.emit(0)

    def undo_last_deletion(self) -> None:
        if not self.undo_stack or self.x_data is None or self.y_data is None:
            return

        idx_arr = self.undo_stack.pop()
        if self.origin_y_data is not None:
            self.y_data[idx_arr] = self.origin_y_data[idx_arr]

        self.selected_points.clear()
        self._clear_highlights()
        self._refresh_display_data()
        self.selection_changed.emit(0)

    # ─── 鼠标事件 ──────────────────────────────────────────────

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

    def _clear_selection_rect(self) -> None:
        if self.selection_rect_item is not None:
            self.getViewBox().removeItem(self.selection_rect_item)
            self.selection_rect_item = None

    def _update_selection_rect(self) -> None:
        if self.start_point is None or self.end_point is None:
            return
        view_box = self.getViewBox()
        if view_box is None:
            return
        try:
            view_rect = view_box.viewRect()
            start_view = view_box.mapSceneToView(self.start_point)
            end_view = view_box.mapSceneToView(self.end_point)

            x_min = max(min(start_view.x(), end_view.x()), view_rect.left())
            x_max = min(max(start_view.x(), end_view.x()), view_rect.right())
            y_min = min(start_view.y(), end_view.y())
            y_max = max(start_view.y(), end_view.y())

            width = x_max - x_min
            height = y_max - y_min

            if self.selection_rect_item is None:
                self.selection_rect_item = QtWidgets.QGraphicsRectItem(
                    QtCore.QRectF(x_min, y_min, width, height)
                )
                pen = QtGui.QPen(self._rect_color)
                pen.setWidth(0)
                pen.setStyle(QtCore.Qt.PenStyle.DashLine)
                self.selection_rect_item.setPen(pen)
                fill_color = QtGui.QColor(self._rect_color)
                fill_color.setAlpha(30)
                self.selection_rect_item.setBrush(QtGui.QBrush(fill_color))
                self.selection_rect_item.setZValue(10)
                self.getViewBox().addItem(self.selection_rect_item, ignoreBounds=True)
            else:
                self.selection_rect_item.setRect(QtCore.QRectF(x_min, y_min, width, height))
        except Exception as e:
            logger.warning("更新选择矩形错误: %s", e)

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        vb = self.getViewBox()
        apply_modifier_axis_constraint(vb, ev.modifiers())
        super().wheelEvent(ev)
        vb.setMouseEnabled(x=True, y=True)

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._is_mouse_over_auto_scale_button(ev.position()):
            super().mousePressEvent(ev)
            return

        scene_pos = self._to_scene(ev)

        if ev.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._apply_modifier_axis_constraint(ev.modifiers())
            super().mousePressEvent(ev)

        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            if self.selection_mode == 'select':
                self._rect_color = QtGui.QColor(255, 60, 60, 180)  # 红色 = 取消
                self.start_point = scene_pos
                self.is_selecting = True
                ev.accept()
                return
            super().mousePressEvent(ev)

        elif ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self._rect_color = (
                QtGui.QColor(255, 200, 0, 200) if self.selection_mode == 'zoom'
                else QtGui.QColor(30, 144, 255, 180)
            )
            self.start_point = scene_pos
            self.is_selecting = True
            ev.accept()
            return
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self.is_selecting and self.start_point is not None:
            self.end_point = self._to_scene(ev)
            self._update_selection_rect()
            ev.accept()
            return

        if ev.buttons() & QtCore.Qt.MouseButton.MiddleButton:
            self._apply_modifier_axis_constraint(ev.modifiers())

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._is_mouse_over_auto_scale_button(ev.position()):
            super().mouseReleaseEvent(ev)
            return

        if self.is_selecting and self.start_point is not None:
            self.end_point = self._to_scene(ev)
            self.is_selecting = False

            if self.selection_mode == 'zoom':
                self._process_zoom_rect(ev.button() == QtCore.Qt.MouseButton.LeftButton)
            else:
                self._process_rectangle_selection(
                    ev.button() == QtCore.Qt.MouseButton.RightButton
                )

            self._clear_selection_rect()
            ev.accept()
            return

        if ev.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.getViewBox().setMouseEnabled(x=True, y=True)

        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._is_mouse_over_auto_scale_button(ev.position()):
            super().mouseDoubleClickEvent(ev)
            return
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.getViewBox().autoRange()
            ev.accept()
            return
        super().mouseDoubleClickEvent(ev)

    # ─── 矩形处理 ──────────────────────────────────────────────

    def _process_rectangle_selection(self, is_right_click: bool = False) -> None:
        if self.start_point is None or self.end_point is None:
            return
        view_box = self.getViewBox()
        if view_box is None or self.x_data is None or self.y_data is None:
            return
        try:
            start_view = view_box.mapSceneToView(self.start_point)
            end_view = view_box.mapSceneToView(self.end_point)
            x_min, x_max = sorted([start_view.x(), end_view.x()])
            y_min, y_max = sorted([start_view.y(), end_view.y()])

            mask = (
                (self.x_data >= x_min) & (self.x_data <= x_max)
                & (self.y_data >= y_min) & (self.y_data <= y_max)
            )
            indices = set(np.where(mask)[0])

            if is_right_click:
                self.selected_points -= indices
            else:
                self.selected_points.update(indices)

            self._update_highlights()
            self.selection_changed.emit(len(self.selected_points))
        except Exception as e:
            logger.warning("框选计算错误: %s", e)

    def _process_zoom_rect(self, is_left: bool = True) -> None:
        if not is_left or self.start_point is None or self.end_point is None:
            return
        vb = self.getViewBox()
        start = vb.mapSceneToView(self.start_point)
        end = vb.mapSceneToView(self.end_point)
        x_min, x_max = sorted([start.x(), end.x()])
        y_min, y_max = sorted([start.y(), end.y()])
        if x_max - x_min < 1e-6 or y_max - y_min < 1e-6:
            return
        vb.setRange(xRange=(x_min, x_max), yRange=(y_min, y_max), padding=0)


class RemoveSpike(QtWidgets.QWidget):
    """离散点框选去噪的完整工具窗口。

    通过矩形框选标记离散异常点（红色小圆点），
    Delete 键或按钮一键删除。

    不持有 Broker 引用——通过信号与外部通信。
    """

    # 工具信号（data_process 接线到 Broker）
    data_requested = QtCore.Signal(int, str)
    result_submitted = QtCore.Signal(int, str, object)
    result_signal = QtCore.Signal(str, object)
    channel_sync_requested = QtCore.Signal(int, str, object)

    def __init__(
        self,
        source: int | pd.Series,
        channel_name: str,
        parent=None,
    ):
        super().__init__(parent)

        if self.parent() is not None:
            self.setWindowFlag(QtCore.Qt.WindowType.Window)

        self.data_id = source if isinstance(source, int) else 0
        self._series_source = source if isinstance(source, pd.Series) else None
        self._is_pipeline_editor = self._series_source is not None
        self.ch_name = channel_name

        # 待 _on_data_ready 填充
        self.x_data: np.ndarray | None = None
        self.y_data: np.ndarray | None = None
        self.all_channels: list[str] = []
        self.origin_all_channels: dict[str, np.ndarray] = {}

        self.plot_widget: SpikePlotWidget | None = None
        self.plot_item = None

        self.status_label = QtWidgets.QLabel("当前选中: 0 个数据点")
        self.instruction_label = QtWidgets.QLabel()
        self.instruction_label.setWordWrap(True)

        self._init_ui()
        self._update_instruction_label()
        self._connect_signals()
        self._init_shortcuts()
        if self._series_source is not None:
            self.load_series(channel_name, self._series_source)

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

        switch_sc = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Tab), self)
        switch_sc.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        switch_sc.activated.connect(self._toggle_selection_mode)

    def _init_ui(self):
        self.plot_widget = SpikePlotWidget()
        self.plot_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.plot_widget.getAxis('left').setStyle(tickLength=0)
        self.plot_widget.getAxis('bottom').setStyle(tickLength=0)
        self.plot_item = self.plot_widget.getPlotItem()

        btn_layout = QtWidgets.QHBoxLayout()
        self.mode_toggle_btn = QtWidgets.QPushButton("切换为选点模式")
        self.mode_label = QtWidgets.QLabel("当前模式: 框选放大")

        self.revert_btn = QtWidgets.QPushButton("还原原始数据")
        self.modify_data_btn = QtWidgets.QPushButton("删除选中数据")
        self.undo_btn = QtWidgets.QPushButton("撤销上一次删除")
        self.confirm_btn = QtWidgets.QPushButton(
            "保存步骤并关闭" if self._is_pipeline_editor else "确定"
        )

        btn_layout.addWidget(self.mode_toggle_btn)
        btn_layout.addWidget(self.mode_label)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.revert_btn)
        btn_layout.addWidget(self.modify_data_btn)
        btn_layout.addWidget(self.undo_btn)
        btn_layout.addStretch(1)
        sync_label = "应用到当前 Pipeline 的全部通道" if self._is_pipeline_editor else "应用到所有通道"
        self.sync_checkbox = QtWidgets.QCheckBox(sync_label)
        self.sync_checkbox.setChecked(False)
        self.sync_checkbox.setToolTip(
            "确认后为当前 Pipeline 的其他通道添加同一 Remove Spike 步骤；"
            "不会立刻写回原始数据。"
            if self._is_pipeline_editor
            else "确定时将当前通道的删除点同步到所有其他通道"
        )
        btn_layout.addWidget(self.sync_checkbox)
        btn_layout.addWidget(self.confirm_btn)

        glay = QtWidgets.QVBoxLayout()
        glay.addWidget(self.status_label)
        glay.addWidget(self.instruction_label)
        glay.addWidget(self.plot_widget)
        glay.addLayout(btn_layout)
        self.setLayout(glay)

    def _connect_signals(self):
        self.mode_toggle_btn.clicked.connect(self._toggle_selection_mode)
        self.revert_btn.clicked.connect(self.plot_widget.revert_to_original_data)
        self.undo_btn.clicked.connect(self.plot_widget.undo_last_deletion)
        self.modify_data_btn.clicked.connect(self.plot_widget.modify_selected_data)
        self.confirm_btn.clicked.connect(self._confirm_and_close)
        self.plot_widget.selection_changed.connect(self._on_selection_changed)

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

    def showEvent(self, ev: QtGui.QShowEvent) -> None:
        super().showEvent(ev)
        self.activateWindow()
        self.plot_widget.setFocus()

    def _toggle_selection_mode(self):
        current = self.plot_widget.selection_mode
        if current == 'zoom':
            self.plot_widget.set_selection_mode('select')
            self.mode_toggle_btn.setText("切换为放大模式")
            self.mode_label.setText("当前模式: 框选选点")
        else:
            self.plot_widget.set_selection_mode('zoom')
            self.mode_toggle_btn.setText("切换为选点模式")
            self.mode_label.setText("当前模式: 框选放大")
        self._update_instruction_label()

    def _update_instruction_label(self) -> None:
        if self.plot_widget.selection_mode == 'select':
            mode_help = "框选选点模式：左键框选添加点 | 右键框选取消点 | Tab/按钮切换为放大模式"
        else:
            mode_help = "框选放大模式：左键框选放大 | 右键拖动缩放 | Tab/按钮切换为选点模式"
        self.instruction_label.setText(
            f"操作说明（{mode_help}） | 中键移动视图 | 双击自适应视图\n"
            "         Ctrl+滚轮/中键拖动 → 仅操作X轴 | Shift+滚轮/中键拖动 → 仅操作Y轴 | "
            "Ctrl+Z撤销删除 | Ctrl+S保存 | Delete删除选中点"
        )

    def _on_selection_changed(self, count: int):
        self.status_label.setText(f"当前选中: {count} 个数据点")

    def _add_text_placeholder(self, text):
        self.plot_item.clear()
        text_item = pg.TextItem(text, color='k', anchor=(0.5, 0.5))
        self.plot_item.addItem(text_item)
        text_item.setPos(0, 0)

    def _confirm_and_close(self):
        result = self.plot_widget.confirm_and_exit()
        if result.size == 0:
            return

        logger.info("RemoveSpike confirmed, returning %s points", len(result))

        if self.sync_checkbox.isChecked() and not self._is_pipeline_editor:
            self.channel_sync_requested.emit(self.data_id, self.ch_name, result)

        self.result_submitted.emit(self.data_id, self.ch_name, result)
        self.result_signal.emit(self.ch_name, result)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    from io_utils.mat_io import MatLoader
    em_data = MatLoader().load(
        "/home/transen5/Project/atm_rpc/039BE-20240501-20240515-dt5_struct.mat"
    )
    app = QApplication([])
    window = RemoveSpike(em_data=em_data, channel_name='Ex1')
    window.show()
    app.exec()
