import os
import logging

from utils.series import dti_to_numpy

os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore, QtGui
from typing import List, Set, Optional, Any

from core.em_data import Channel

pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

logger = logging.getLogger(__name__)


class RegionSelectionPlotWidget(pg.PlotWidget):
    selection_changed = QtCore.Signal(object)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, axisItems={'bottom': pg.DateAxisItem()})

        # 数据存储
        self.x_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None
        self.origin_y_data: Optional[np.ndarray] = None
        self.deletion_history: List[List[int]] = []
        self.selected_points: Set[int] = set()

        # 状态变量
        self.selection_mode: str = 'zoom'
        self.is_selecting: bool = False
        self.start_point: Optional[QtCore.QPointF] = None
        self.end_point: Optional[QtCore.QPointF] = None
        self.is_over_region: bool = False

        # UI元素
        self.selection_rect_item: Optional[QtWidgets.QGraphicsRectItem] = None

        # 私有变量
        self._regions: List[pg.LinearRegionItem] = []
        self._rect_color: QtGui.QColor = QtGui.QColor(30, 144, 255, 180)
        self._highlight_scatter: Optional[pg.ScatterPlotItem] = None
        self._plot_curve: Optional[pg.PlotDataItem] = None
        self._current_cursor_shape = QtCore.Qt.CursorShape.ArrowCursor

        # 初始化设置
        self._setup_view()

    def _setup_view(self) -> None:
        """设置视图基本参数"""
        self.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.getViewBox().setMouseEnabled(x=True, y=True)
        self.setBackground('w')
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    # 数据管理相关函数
    def set_data(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """设置绘图数据"""
        self.x_data = np.array(x_data, dtype=np.float64)
        self.origin_y_data = np.array(y_data, dtype=np.float64)
        self.y_data = self.origin_y_data.copy()

        self._clear_all()
        self._plot_curve = self.plot(self.x_data, self.y_data, pen='b')

    def _clear_all(self) -> None:
        """清除所有状态"""
        self.clear()
        self._regions.clear()
        self._plot_curve = None
        self.selected_points.clear()
        self.deletion_history.clear()

    def revert_to_original_data(self) -> None:
        """还原到原始数据"""
        if self.origin_y_data is not None and self.x_data is not None:
            self.y_data = self.origin_y_data.copy()
            self._clear_all()
            self._plot_curve = self.plot(self.x_data, self.y_data, pen='b')
            self._on_region_changed()

    def confirm_and_exit(self) -> np.ndarray:
        """确认修改并返回处理后的数据"""
        if self.y_data is not None:
            return self.y_data
        return np.array([])

    def set_selection_mode(self, mode: str) -> None:
        """设置选择模式"""
        self.selection_mode = mode
        self._clear_selection_rect()
        self._highlight_selected_points()

    # 区域选择相关函数
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
        """移除当前选中的区域"""
        if not self._regions:
            return

        region = self._regions.pop()
        self.removeItem(region)
        self._on_region_changed()

    def clear_all_selections(self) -> None:
        """清空所有选择"""
        for region in self._regions[:]:
            self.removeItem(region)
        self._regions.clear()
        self.selected_points.clear()
        self._on_region_changed()

    def _on_region_changed_lightweight(self) -> None:
        """拖动中只更新选中计数，不重建高亮散点"""
        selected_indices = self._get_selected_indices()
        self.selection_changed.emit(selected_indices)
        # 不调用 _highlight_selected_points()

    # 选择处理函数
    def _on_region_changed(self) -> None:
        """当区域发生变化时调用"""
        selected_indices = self._get_selected_indices()  # 只算1次
        self.selection_changed.emit(selected_indices)
        self._highlight_selected_points(selected_indices)

    def _get_selected_indices(self) -> List[int]:
        """获取所有被选择的数据索引"""
        if self.x_data is None or len(self.x_data) == 0:
            return []

        all_indices_set = set()

        for region in self._regions:
            try:
                xmin, xmax = region.getRegion()
                mask = (self.x_data >= xmin) & (self.x_data <= xmax)
                indices = np.where(mask)[0]
                all_indices_set.update(indices)
            except Exception as e:
                logger.warning("计算区域索引时出错: %s", e)
                continue

        all_indices_set.update(self.selected_points)
        return sorted(all_indices_set)

    # 鼠标交互相关函数
    def _is_mouse_over_region(self, pos: QtCore.QPointF) -> bool:
        """检查鼠标是否在区域上"""
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
                x_range = view_range[1] - view_range[0]
                tolerance = x_range * 0.03

                if (abs(scene_pos.x() - rgn[0]) < tolerance or
                        abs(scene_pos.x() - rgn[1]) < tolerance):
                    return True
        except Exception:
            pass

        return False

    def _is_mouse_over_auto_scale_button(self, pos: QtCore.QPointF) -> bool:
        """检查鼠标是否在自动缩放按钮上"""
        view_box = self.getViewBox()
        if view_box is None:
            return False

        try:
            button_rect = QtCore.QRectF(1, self.height() - 31, 30, 30)
            return button_rect.contains(pos)
        except Exception:
            return False

    def _to_scene(self, ev: QtGui.QMouseEvent) -> QtCore.QPointF:
        """将鼠标事件的 widget 局部坐标转换为 QGraphicsScene 坐标"""
        p = ev.position()
        return self.mapToScene(int(p.x()), int(p.y()))

    def _apply_modifier_axis_constraint(self, modifiers) -> None:
        """根据当前修饰键实时设置轴向约束"""
        vb = self.getViewBox()
        if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            vb.setMouseEnabled(x=True, y=False)
        elif modifiers & QtCore.Qt.KeyboardModifier.AltModifier:
            vb.setMouseEnabled(x=False, y=True)
        else:
            vb.setMouseEnabled(x=True, y=True)

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        """滚轮事件：根据实时修饰键约束轴向"""
        modifiers = ev.modifiers()
        vb = self.getViewBox()
        if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            vb.setMouseEnabled(x=True, y=False)
        elif modifiers & QtCore.Qt.KeyboardModifier.AltModifier:
            vb.setMouseEnabled(x=False, y=True)
        else:
            vb.setMouseEnabled(x=True, y=True)
        super().wheelEvent(ev)
        # 滚轮结束后恢复两轴（避免状态残留）
        vb.setMouseEnabled(x=True, y=True)

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._is_mouse_over_auto_scale_button(ev.position()):
            super().mousePressEvent(ev)
            return

        scene_pos = self._to_scene(ev)
        self.is_over_region = self._is_mouse_over_region(scene_pos)

        # 中键按下时立即根据当前modifier约束轴向
        if ev.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._apply_modifier_axis_constraint(ev.modifiers())

        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            view = self.getViewBox()

            if view is not None and self.x_data is not None:
                try:
                    pos = view.mapSceneToView(scene_pos)

                    for region in reversed(self._regions):
                        rgn = region.getRegion()
                        view_range = view.viewRange()[0]
                        x_range = view_range[1] - view_range[0]
                        tolerance = x_range * 0.01

                        if (rgn[0] <= pos.x() <= rgn[1] or
                                abs(pos.x() - rgn[0]) < tolerance or
                                abs(pos.x() - rgn[1]) < tolerance):
                            self._regions.remove(region)
                            self.removeItem(region)
                            self._on_region_changed()
                            ev.accept()
                            return

                    if self.selection_mode == 'select' and not self.is_over_region:
                        self._rect_color = QtGui.QColor(255, 60, 60, 180)  # 红色
                        self.start_point = scene_pos
                        self.is_selecting = True
                        ev.accept()
                        return

                except Exception as e:
                    logger.warning("右键处理错误: %s", e)

            ev.accept()
            return
        elif ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.is_over_region:
                super().mousePressEvent(ev)
                return
            if self.selection_mode in ('select', 'zoom'):  # 两个模式都自己画框
                # 按模式设置颜色
                if self.selection_mode == 'zoom':
                    self._rect_color = QtGui.QColor(255, 200, 0, 200)  # 黄色
                else:
                    self._rect_color = QtGui.QColor(30, 144, 255, 180)  # 蓝色
                self.start_point = scene_pos
                self.is_selecting = True
                ev.accept()
                return
            super().mousePressEvent(ev)

        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        scene_pos = self._to_scene(ev)
        self.is_over_region = self._is_mouse_over_region(scene_pos)

        new_cursor = QtCore.Qt.CursorShape.SizeHorCursor if self.is_over_region else QtCore.Qt.CursorShape.ArrowCursor
        if new_cursor != self._current_cursor_shape:
            self.setCursor(QtGui.QCursor(new_cursor))
            self._current_cursor_shape = new_cursor

        if self.is_selecting and self.start_point is not None and not self.is_over_region:
            self.end_point = scene_pos
            self._update_selection_rect()
            ev.accept()
            return

        # 中键拖动时，根据当前实时modifier限制轴向
        if ev.buttons() & QtCore.Qt.MouseButton.MiddleButton:
            self._apply_modifier_axis_constraint(ev.modifiers())

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._is_mouse_over_auto_scale_button(ev.position()):
            super().mouseReleaseEvent(ev)
            return

        if self.is_selecting and self.start_point is not None and not self.is_over_region:
            self.end_point = self._to_scene(ev)
            self.is_selecting = False

            if self.selection_mode == 'zoom':
                self._process_zoom_rect()  # zoom 模式 → 缩放
            else:
                self._process_rectangle_selection(  # select 模式 → 标记点
                    ev.button() == QtCore.Qt.MouseButton.RightButton
                )

            self._clear_selection_rect()
            ev.accept()
            return

        # 中键松开时恢复两轴约束
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

    # 矩形选择相关函数
    def _clear_selection_rect(self) -> None:
        """清除选择矩形"""
        if self.selection_rect_item is not None:
            self.getViewBox().removeItem(self.selection_rect_item)
            self.selection_rect_item = None

    def _update_selection_rect(self) -> None:
        """更新选择矩形显示"""
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

            if start_view.y() < end_view.y():
                y_min = start_view.y()
                y_max = end_view.y()
            else:
                y_min = end_view.y()
                y_max = start_view.y()

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

                brush = QtGui.QBrush(fill_color)
                self.selection_rect_item.setBrush(brush)
                self.selection_rect_item.setZValue(10)
                self.getViewBox().addItem(self.selection_rect_item, ignoreBounds=True)
            else:
                self.selection_rect_item.setRect(QtCore.QRectF(x_min, y_min, width, height))

        except Exception as e:
            logger.warning("更新选择矩形错误: %s", e)

    def _process_rectangle_selection(self, is_right_click: bool = False) -> None:
        """处理矩形选择逻辑"""
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
                    (self.x_data >= x_min) &
                    (self.x_data <= x_max) &
                    (self.y_data >= y_min) &
                    (self.y_data <= y_max)
            )

            indices = set(np.where(mask)[0])

            if is_right_click:
                self.selected_points -= indices
            else:
                self.selected_points.update(indices)

            self._highlight_selected_points()
            self._on_region_changed()

        except Exception as e:
            logger.warning("框选计算错误: %s", e)

    def _process_zoom_rect(self) -> None:
        """zoom 模式：松手后将视图缩放到框选范围"""
        if self.start_point is None or self.end_point is None:
            return

        vb = self.getViewBox()
        start = vb.mapSceneToView(self.start_point)
        end = vb.mapSceneToView(self.end_point)

        x_min, x_max = sorted([start.x(), end.x()])
        y_min, y_max = sorted([start.y(), end.y()])

        # 框太小时忽略（误触保护）
        if x_max - x_min < 1e-6 or y_max - y_min < 1e-6:
            return

        vb.setRange(
            xRange=(x_min, x_max),
            yRange=(y_min, y_max),
            padding=0
        )

    # 高亮显示函数
    def _highlight_selected_points(self, selected_indices: List[int] = None)  -> None:
        """高亮显示选中的数据点"""

        if selected_indices is None:
            selected_indices = self._get_selected_indices()

        if self._highlight_scatter is not None:
            self.removeItem(self._highlight_scatter)
            self._highlight_scatter = None

        if selected_indices and self.x_data is not None and self.y_data is not None:
            valid_indices = [i for i in selected_indices if i < len(self.x_data) and i < len(self.y_data)]

            if valid_indices:
                self._highlight_scatter = pg.ScatterPlotItem(
                    x=self.x_data[valid_indices],
                    y=self.y_data[valid_indices],
                    pen=pg.mkPen(color=(0, 0, 0), width=1),
                    brush=pg.mkBrush(255, 150, 0, 200),
                    size=10,
                    symbol='o'
                )
                self.addItem(self._highlight_scatter)

    # 数据修改函数
    def modify_selected_data(self) -> None:
        """修改选中的数据点（设置为NaN）"""
        if self.x_data is None or self.y_data is None:
            return

        selected_indices = self._get_selected_indices()

        if selected_indices:
            self.deletion_history.append(selected_indices.copy())
            y_modified = self.y_data.copy()
            y_modified[selected_indices] = np.nan
            self.y_data = y_modified

            for regin in self._regions:
                self.removeItem(regin)
            self._regions.clear()
            self.selected_points.clear()
            if self._highlight_scatter is not None:
                self.removeItem(self._highlight_scatter)
                self._highlight_scatter = None

            if self._plot_curve is not None:
                self._plot_curve.setData(self.x_data, self.y_data)
            self._on_region_changed()

    def undo_last_deletion(self) -> None:
        """撤销上一次删除操作"""
        if not self.deletion_history or self.x_data is None or self.y_data is None:
            return

        last_deletion = self.deletion_history.pop()

        if self.origin_y_data is not None:
            self.y_data[last_deletion] = self.origin_y_data[last_deletion]

        for regin in self._regions:
            self.removeItem(regin)
        self._regions.clear()
        self.selected_points.clear()
        if self._highlight_scatter is not None:
            self.removeItem(self._highlight_scatter)
            self._highlight_scatter = None

        if self._plot_curve is not None:
            self._plot_curve.setData(self.x_data, self.y_data)

        self._on_region_changed()

class RemoveSpike(QtWidgets.QWidget):
    result_signal = QtCore.Signal(str, np.ndarray)

    def __init__(
            self,
            channel: Channel,
            parent=None,
    ):
        super().__init__(parent)

        if self.parent() is not None:
            self.setWindowFlag(QtCore.Qt.WindowType.Window)

        x = channel.datetime_index()
        self.x_data = dti_to_numpy(x)
        self.y_data = channel.cts.to_numpy()
        self.ch_name = channel.name
        self.plot_widget = None
        self.plot_item = None

        self.status_label = QtWidgets.QLabel("当前选中: 0 个数据点")
        self.instruction_label = QtWidgets.QLabel(
            "操作说明: 左键双击创建区域 | 右键点击区域删除 | 左键框选添加点 | 右键框选取消点 | 中键移动视图 | Ctrl+Z撤销删除 | Ctrl+S保存 \n"
            "         Delete快捷删除选中点 | Tab切换模式 | Shift+鼠标滚轮放缩/中键移动操作X轴 | Alt+鼠标滚轮放缩/中键移动操作Y轴"
        )

        self._init_ui()
        self._connect_signals()
        self._init_plot()
        self._init_shortcuts()

    def _init_shortcuts(self):
        """初始化快捷键"""
        delete_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), self)
        delete_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        delete_shortcut.activated.connect(self.plot_widget.modify_selected_data)

        save_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        save_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        save_shortcut.activated.connect(self._confirm_and_close)

        undo_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self)
        undo_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        undo_shortcut.activated.connect(self.plot_widget.undo_last_deletion)

        switch_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Tab), self)
        switch_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        switch_shortcut.activated.connect(self._toggle_selection_mode)

    def _init_ui(self):
        """初始化用户界面"""
        self.plot_widget = RegionSelectionPlotWidget()
        self.plot_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        self.plot_widget.getAxis('left').setStyle(tickLength=0)
        self.plot_widget.getAxis('bottom').setStyle(tickLength=0)
        self.plot_item = self.plot_widget.getPlotItem()

        button_layout = QtWidgets.QHBoxLayout()

        self.mode_toggle_btn = QtWidgets.QPushButton("切换为选点模式")
        self.mode_label = QtWidgets.QLabel("当前模式: 框选放大")

        self.add_region_btn = QtWidgets.QPushButton("添加连续选择区域")
        self.remove_region_btn = QtWidgets.QPushButton("移除最后连续区域")
        self.revert_btn = QtWidgets.QPushButton("还原原始数据")
        self.clear_all_btn = QtWidgets.QPushButton("清空所有选择")
        self.modify_data_btn = QtWidgets.QPushButton("删除选中数据")
        self.undo_btn = QtWidgets.QPushButton("撤销上一次删除")
        self.confirm_btn = QtWidgets.QPushButton("确定")

        button_layout.addWidget(self.mode_toggle_btn)
        button_layout.addWidget(self.mode_label)
        button_layout.addStretch(1)
        button_layout.addWidget(self.add_region_btn)
        button_layout.addWidget(self.remove_region_btn)
        button_layout.addWidget(self.revert_btn)
        button_layout.addWidget(self.clear_all_btn)
        button_layout.addWidget(self.modify_data_btn)
        button_layout.addWidget(self.undo_btn)
        button_layout.addStretch(1)
        button_layout.addWidget(self.confirm_btn)

        glay = QtWidgets.QVBoxLayout()
        glay.addWidget(self.status_label)
        glay.addWidget(self.instruction_label)
        glay.addWidget(self.plot_widget)
        glay.addLayout(button_layout)
        self.setLayout(glay)

    def _connect_signals(self):
        """连接信号和槽函数"""

        self.mode_toggle_btn.clicked.connect(self._toggle_selection_mode)
        self.add_region_btn.clicked.connect(self.plot_widget.add_region)
        self.remove_region_btn.clicked.connect(self.plot_widget.remove_selected_region)
        self.revert_btn.clicked.connect(self.plot_widget.revert_to_original_data)
        self.undo_btn.clicked.connect(self.plot_widget.undo_last_deletion)
        self.clear_all_btn.clicked.connect(self._clear_all_selections)
        self.modify_data_btn.clicked.connect(self.plot_widget.modify_selected_data)
        self.confirm_btn.clicked.connect(self._confirm_and_close)

        self.plot_widget.selection_changed.connect(self._on_selection_changed)

    def _init_plot(self):
        """初始化绘图"""
        if self.x_data is None or self.y_data is None:
            self._add_text_placeholder("请先加载数据")
            return

        self.plot_widget.set_data(self.x_data, self.y_data)

        grid_pen = pg.mkPen(
            color=pg.mkColor(100, 100, 100),
            width=1,
            alpha=1,
            antialiased=True,
            style=QtCore.Qt.PenStyle.DashLine
        )
        self.plot_item.showGrid(x=True, y=True, alpha=0.7)
        self.plot_item.getAxis('bottom').setPen(grid_pen)
        self.plot_item.getAxis('left').setPen(grid_pen)

    def showEvent(self, ev: QtGui.QShowEvent) -> None:
        super().showEvent(ev)
        self.activateWindow()
        self.plot_widget.setFocus()

    # 选择模式管理
    def _toggle_selection_mode(self):
        """切换选择模式"""
        current_mode = self.plot_widget.selection_mode
        if current_mode == 'zoom':
            new_mode = 'select'
            self.mode_toggle_btn.setText("切换为放大模式")
            self.mode_label.setText("当前模式: 框选选点")
        else:
            new_mode = 'zoom'
            self.mode_toggle_btn.setText("切换为选点模式")
            self.mode_label.setText("当前模式: 框选放大")
        self.plot_widget.set_selection_mode(new_mode)

    def _clear_all_selections(self):
        """清空所有选择"""
        self.plot_widget.clear_all_selections()

    def _on_selection_changed(self, selected_indices: List[int]):
        """当选择发生变化时更新状态"""
        self.status_label.setText(f"当前选中: {len(selected_indices)} 个数据点")

    # 工具函数
    def _add_text_placeholder(self, text):
        """添加文本占位符"""
        self.plot_item.clear()
        text_item = pg.TextItem(text, color='k', anchor=(0.5, 0.5))
        self.plot_item.addItem(text_item)
        text_item.setPos(0, 0)

    def set_label(self, xlabel, ylabel):
        """设置坐标轴标签"""
        self.plot_item.setLabel('bottom', xlabel)
        self.plot_item.setLabel('left', ylabel)

    def _confirm_and_close(self):
        """确认修改并关闭窗口"""
        result = self.plot_widget.confirm_and_exit()
        if result.size > 0:
            logger.info("RemoveSpike confirmed, returning %s points", len(result))
            self.result_signal.emit(self.ch_name, result)
            self.hide()
            self.close()
            if self.parent() is not None:
                self.parent().close()

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    from io_utils.mat_io import MatLoader
    em_data = MatLoader().load("/home/transen5/Project/atm_rpc/039BE-20240501-20240515-dt5_struct.mat")
    app = QApplication([])
    window = RemoveSpike(channel=em_data.data['Ex1'])
    window.show()
    app.exec()
