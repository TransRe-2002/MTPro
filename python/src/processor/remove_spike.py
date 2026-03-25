import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore, QtGui
from typing import List, Set, Optional, Any
import pandas as pd

from core.em_data import EMData, Channel

pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class RegionSelectionPlotWidget(pg.PlotWidget):
    selection_changed = QtCore.Signal(object)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, axisItems={'bottom': pg.DateAxisItem()})

        # 数据存储
        self.x_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None
        self.origin_y_data: Optional[np.ndarray] = None
        self.deletion_history: List[List[int]] = []
        self.regions: List[pg.LinearRegionItem] = []
        self.selected_points: Set[int] = set()

        # 状态变量
        self.selection_mode: str = 'zoom'
        self.visible_points: int = 0
        self.is_selecting: bool = False
        self.start_point: Optional[QtCore.QPointF] = None
        self.end_point: Optional[QtCore.QPointF] = None
        self.is_over_region: bool = False

        # UI元素
        self.selection_rect_item: Optional[QtWidgets.QGraphicsRectItem] = None
        self.highlight_color = (255, 150, 0, 200)
        self.region_highlight_color = (255, 100, 100, 150)

        # 初始化设置
        self._setup_view()
        self._setup_signals()

    def _setup_view(self) -> None:
        """设置视图基本参数"""
        self.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.setBackground('w')
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

    def _setup_signals(self) -> None:
        """连接信号"""
        self.getViewBox().sigRangeChanged.connect(self._on_view_range_changed)

    # 数据管理相关函数
    def set_data(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """设置绘图数据"""
        self.x_data = np.array(x_data, dtype=np.float64)
        self.origin_y_data = np.array(y_data, dtype=np.float64)
        self.y_data = self.origin_y_data.copy()

        self._clear_all()
        self.plot(self.x_data, self.y_data, pen='b')
        self._update_selection_mode_based_on_data()

    def _clear_all(self) -> None:
        """清除所有状态"""
        self.clear()
        self.regions.clear()
        self.selected_points.clear()
        self.deletion_history.clear()

    def revert_to_original_data(self) -> None:
        """还原到原始数据"""
        if self.origin_y_data is not None and self.x_data is not None:
            self.y_data = self.origin_y_data.copy()
            self.deletion_history.clear()
            self._clear_all()
            self.plot(self.x_data, self.y_data, pen='b')
            self._update_selection_mode_based_on_data()
            self._on_region_changed()
            print("已还原到原始数据")

    def confirm_and_exit(self) -> np.ndarray:
        """确认修改并返回处理后的数据"""
        if self.y_data is not None:
            print("修改已确认并保存")
            return self.y_data
        return np.array([])

    # 选择模式管理
    def _update_selection_mode_based_on_data(self) -> None:
        """根据当前可见数据点数更新选择模式"""
        self.visible_points = self._get_visible_point_count()

        if self.visible_points > 50000:
            self.set_selection_mode('zoom', force=True)
        else:
            self.set_selection_mode(self.selection_mode)

    def set_selection_mode(self, mode: str, force: bool = False) -> None:
        """设置选择模式"""
        if self.x_data is not None and self.visible_points > 50000 and not force:
            self.selection_mode = 'zoom'
            self.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        else:
            self.selection_mode = mode
            if mode == 'zoom':
                self.getViewBox().setMouseMode(pg.ViewBox.RectMode)
            else:
                self.getViewBox().setMouseMode(pg.ViewBox.PanMode)

        self._clear_selection_rect()
        self._highlight_selected_points()

    def toggle_selection_mode(self) -> None:
        """切换选择模式"""
        if self.selection_mode == 'zoom':
            self.set_selection_mode('select')
        else:
            self.set_selection_mode('zoom')

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
        border_pen = pg.mkPen(color=(65, 105, 225), width=7)
        region.lines[0].setPen(border_pen)
        region.lines[1].setPen(border_pen)
        region.setBrush(pg.mkBrush(135, 206, 250, 80))

        self.addItem(region)
        self.regions.append(region)
        region.sigRegionChanged.connect(self._on_region_changed)

        self._on_region_changed()
        print(f"创建区域: [{region_start:.2f}, {region_end:.2f}]")

    def remove_selected_region(self) -> None:
        """移除当前选中的区域"""
        if not self.regions:
            return

        region = self.regions.pop()
        self.removeItem(region)
        self._on_region_changed()

    def clear_all_selections(self) -> None:
        """清空所有选择"""
        for region in self.regions[:]:
            self.removeItem(region)
        self.regions.clear()
        self.selected_points.clear()
        self._on_region_changed()

    # 选择处理函数
    def _on_region_changed(self) -> None:
        """当区域发生变化时调用"""
        selected_indices = self._get_selected_indices()
        self.selection_changed.emit(selected_indices)
        self._highlight_selected_points()

    def _get_selected_indices(self) -> List[int]:
        """获取所有被选择的数据索引"""
        if self.x_data is None or len(self.x_data) == 0:
            return []

        all_indices_set = set()

        for region in self.regions:
            try:
                xmin, xmax = region.getRegion()
                mask = (self.x_data >= xmin) & (self.x_data <= xmax)
                indices = np.where(mask)[0]
                all_indices_set.update(indices.tolist())
            except Exception as e:
                print(f"计算区域索引时出错: {e}")
                continue

        all_indices_set.update(self.selected_points)
        return sorted(all_indices_set)

    def output_selected_indices(self) -> List[int]:
        """输出所有被选择的索引"""
        indices = self._get_selected_indices()
        print(f"选中的索引数量: {len(indices)}")
        print(f"具体索引: {indices}")
        return indices

    def _get_visible_point_count(self) -> int:
        """获取当前视图中可见的数据点数量"""
        if self.x_data is None or self.y_data is None:
            return 0

        view_box = self.getViewBox()
        if view_box is None:
            return len(self.x_data)

        try:
            view_range = view_box.viewRange()[0]
            x_min, x_max = view_range
            mask = (self.x_data >= x_min) & (self.x_data <= x_max)
            return np.sum(mask)
        except Exception:
            return len(self.x_data)

    # 鼠标交互相关函数
    def _is_mouse_over_region(self, pos: QtCore.QPointF) -> bool:
        """检查鼠标是否在区域上"""
        view = self.getViewBox()
        if view is None or not self.regions:
            return False

        try:
            scene_pos = view.mapSceneToView(pos)

            for region in self.regions:
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

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._is_mouse_over_auto_scale_button(ev.position()):
            super().mousePressEvent(ev)
            return

        self.is_over_region = self._is_mouse_over_region(ev.scenePosition())

        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            scene_pos = ev.scenePosition()
            view = self.getViewBox()

            if view is not None and self.x_data is not None:
                try:
                    pos = view.mapSceneToView(scene_pos)

                    for region in reversed(self.regions):
                        rgn = region.getRegion()
                        view_range = view.viewRange()[0]
                        x_range = view_range[1] - view_range[0]
                        tolerance = x_range * 0.01

                        if (rgn[0] <= pos.x() <= rgn[1] or
                                abs(pos.x() - rgn[0]) < tolerance or
                                abs(pos.x() - rgn[1]) < tolerance):
                            self.regions.remove(region)
                            self.removeItem(region)
                            self._on_region_changed()
                            ev.accept()
                            return

                    if self.selection_mode == 'select' and not self.is_over_region:
                        self.start_point = ev.position()
                        self.is_selecting = True
                        ev.accept()
                        return

                except Exception as e:
                    print(f"右键处理错误: {e}")

            ev.accept()
            return

        elif ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.is_over_region:
                super().mousePressEvent(ev)
                return

            if self.selection_mode == 'select':
                self.start_point = ev.position()
                self.is_selecting = True
                ev.accept()
                return
            else:
                super().mousePressEvent(ev)
                return

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.is_over_region = self._is_mouse_over_region(ev.scenePosition())

        if self.is_over_region:
            self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.SizeHorCursor))
        else:
            self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))

        if self.is_selecting and self.start_point is not None and not self.is_over_region:
            self.end_point = ev.position()
            self._update_selection_rect()
            ev.accept()
            return

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._is_mouse_over_auto_scale_button(ev.position()):
            super().mouseReleaseEvent(ev)
            return

        if self.is_selecting and self.start_point is not None and not self.is_over_region:
            self.end_point = ev.position()
            self.is_selecting = False
            self._process_rectangle_selection(ev.button() == QtCore.Qt.RightButton)
            self._clear_selection_rect()
            ev.accept()
            return

        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._is_mouse_over_auto_scale_button(ev.position()):
            super().mouseDoubleClickEvent(ev)
            return

        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            scene_pos = ev.scenePosition()
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
                    print(f"双击创建区域错误: {e}")

        super().mouseDoubleClickEvent(ev)

    # 键盘事件处理
    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        if ev.key() == QtCore.Qt.Key_Z and ev.modifiers() == QtCore.Qt.ControlModifier:
            self.undo_last_deletion()
            ev.accept()
            return
        super().keyPressEvent(ev)

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
                pen = QtGui.QPen(QtGui.QColor(30, 144, 255, 180))
                pen.setWidth(0)
                pen.setStyle(QtCore.Qt.PenStyle.DashLine)
                self.selection_rect_item.setPen(pen)

                brush = QtGui.QBrush(QtGui.QColor(135, 206, 250, 30))
                self.selection_rect_item.setBrush(brush)
                self.selection_rect_item.setZValue(10)
                self.getViewBox().addItem(self.selection_rect_item)
            else:
                self.selection_rect_item.setRect(QtCore.QRectF(x_min, y_min, width, height))

        except Exception as e:
            print(f"更新选择矩形错误: {e}")

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

            indices = set(int(i) for i in np.where(mask)[0])

            if is_right_click:
                self.selected_points -= indices
            else:
                self.selected_points.update(indices)

            self._highlight_selected_points()
            self._on_region_changed()

        except Exception as e:
            print(f"框选计算错误: {e}")

    # 高亮显示函数
    def _highlight_selected_points(self) -> None:
        """高亮显示选中的数据点"""
        for item in self.items():
            if hasattr(item, 'is_highlight') and item.is_highlight:
                self.removeItem(item)

        selected_indices = self._get_selected_indices()

        if selected_indices and self.x_data is not None and self.y_data is not None:
            valid_indices = [i for i in selected_indices if i < len(self.x_data) and i < len(self.y_data)]

            if valid_indices:
                scatter = pg.ScatterPlotItem(
                    x=self.x_data[valid_indices],
                    y=self.y_data[valid_indices],
                    pen=pg.mkPen(color=(0, 0, 0), width=1),
                    brush=pg.mkBrush(255, 150, 0, 200),
                    size=10,
                    symbol='o'
                )
                scatter.is_highlight = True
                self.addItem(scatter)
                print(f"高亮显示 {len(valid_indices)} 个点")

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

            self.clear()
            self.plot(self.x_data, self.y_data, pen='b')
            self.regions.clear()
            self.selected_points.clear()
            self._update_selection_mode_based_on_data()
            self._on_region_changed()
            print(f"已将 {len(selected_indices)} 个数据点设置为 NaN")

    def undo_last_deletion(self) -> None:
        """撤销上一次删除操作"""
        if not self.deletion_history or self.x_data is None or self.y_data is None:
            print("没有可撤销的操作")
            return

        last_deletion = self.deletion_history.pop()

        if self.origin_y_data is not None:
            self.y_data[last_deletion] = self.origin_y_data[last_deletion]

        self.clear()
        self.plot(self.x_data, self.y_data, pen='b')
        self.regions.clear()
        self.selected_points.clear()
        self._on_region_changed()
        print(f"已撤销上一次删除操作，恢复了 {len(last_deletion)} 个点")

    # 视图更新函数
    def _on_view_range_changed(self) -> None:
        """当视图范围变化时调用"""
        self._update_selection_mode_based_on_data()


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
        self.x_data = np.array([x[i].timestamp() for i in range(len(x))])
        self.y_data = channel.cts.to_numpy()
        self.duration: int = len(self.x_data) if self.x_data is not None else 0
        self.channel = channel.name
        self.plot_widget = None
        self.plot_item = None

        self.x_viewport_start: Optional[int] = None
        self.x_viewport_size: Optional[int] = None
        self.y_viewport_start: Optional[int] = None
        self.y_viewport_size: Optional[int] = None

        self.status_label = QtWidgets.QLabel("当前选中: 0 个数据点")
        self.instruction_label = QtWidgets.QLabel(
            "操作说明: 左键双击创建区域 | 右键点击区域删除 | 左键框选添加点 | 右键框选取消点 | 中键移动视图 | Ctrl+Z撤销删除"
        )

        self.zoom_ratio = 1.0

        self._init_ui()
        self._connect_signals()
        self._init_plot()

    def _init_ui(self):
        """初始化用户界面"""
        toolbar1 = QtWidgets.QToolBar()
        self.zoom_in_action = QtGui.QAction("放大", self)
        self.zoom_in_action.setIcon(QtGui.QIcon.fromTheme("zoom-in"))
        self.zoom_in_action.setIconText("放大")
        toolbar1.addAction(self.zoom_in_action)

        self.zoom_out_action = QtGui.QAction("缩小", self)
        self.zoom_out_action.setIcon(QtGui.QIcon.fromTheme("zoom-out"))
        self.zoom_out_action.setIconText("缩小")
        toolbar1.addAction(self.zoom_out_action)

        toolbar1.addSeparator()
        toolbar1.addWidget(QtWidgets.QLabel("绘制时间长度："))

        self.slide_region = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slide_region.setFixedWidth(200)
        self.slide_region.setEnabled(False)
        toolbar1.addWidget(self.slide_region)

        self.plot_widget = RegionSelectionPlotWidget()
        self.plot_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        self.plot_widget.getAxis('left').setStyle(tickLength=0)
        self.plot_widget.getAxis('bottom').setStyle(tickLength=0)
        self.plot_item = self.plot_widget.getPlotItem()

        self.scroll_x = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
        self.scroll_x.setEnabled(False)
        self.scroll_y = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical)
        self.scroll_y.setEnabled(False)
        self.scroll_y.setInvertedControls(True)
        self.scroll_y.setInvertedAppearance(True)

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

        glay = QtWidgets.QGridLayout()
        glay.addWidget(toolbar1, 0, 0, 1, 2)
        glay.addWidget(self.status_label, 1, 0, 1, 2)
        glay.addWidget(self.instruction_label, 2, 0, 1, 2)
        glay.addWidget(self.plot_widget, 3, 1)
        glay.addWidget(self.scroll_x, 4, 1)
        glay.addWidget(self.scroll_y, 3, 2)
        glay.addLayout(button_layout, 5, 0, 1, 2)
        self.setLayout(glay)

    def _connect_signals(self):
        """连接信号和槽函数"""
        self.scroll_x.valueChanged.connect(self._on_scroll_x_changed)
        self.scroll_y.valueChanged.connect(self._on_scroll_y_changed)
        self.slide_region.valueChanged.connect(self._on_slide_region_changed)

        self.zoom_in_action.triggered.connect(self._on_zoom_in_triggered)
        self.zoom_out_action.triggered.connect(self._on_zoom_out_triggered)

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

        if isinstance(self.x_data, pd.Series):
            x_values = self.x_data.to_numpy()
        else:
            x_values = np.array(self.x_data)

        if isinstance(self.y_data, pd.Series):
            y_values = self.y_data.to_numpy()
        else:
            y_values = np.array(self.y_data)

        self.plot_widget.set_data(x_values, y_values)

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

        self._setup_viewport_parameters()

    def _setup_viewport_parameters(self):
        """设置视口参数"""
        self.x_viewport_size = self.duration
        self.x_viewport_start = 0

        self.scroll_x.setEnabled(True)
        self.scroll_x.setMinimum(0)
        self.scroll_x.setMaximum(max(0, self.duration - self.x_viewport_size))
        self.scroll_x.setPageStep(self.x_viewport_size)
        self.scroll_x.setValue(self.x_viewport_start)

        y_min = np.nanmin(self.y_data)
        y_max = np.nanmax(self.y_data)
        y_center = (y_max + y_min) / 2
        y_diff = (y_max - y_min) / 2

        ymin = y_center - y_diff * 1.05
        ymax = y_center + y_diff * 1.05
        yrange = ymax - ymin
        self.y_viewport_start = ymin
        self.y_viewport_size = yrange * self.zoom_ratio

        self.scroll_y.setEnabled(True)
        self.scroll_y.setMinimum(int(ymin * 10000))
        self.scroll_y.setMaximum(max(int(ymin * 10000), int((ymax - self.y_viewport_size) * 10000)))
        self.scroll_y.setPageStep(int(self.y_viewport_size * 10000))
        self.scroll_y.setValue(int(self.y_viewport_start * 10000))

        self.slide_region.setEnabled(True)
        self.slide_region.blockSignals(True)
        self.slide_region.setMinimum(200)
        self.slide_region.setMaximum(max(1, self.duration))
        self.slide_region.setValue(self.x_viewport_size)
        self.slide_region.blockSignals(False)

        self._update_display()

    # 视图控制函数
    def _on_scroll_x_changed(self):
        """滚动条值改变时更新绘图"""
        self._update_x_viewport()

    def _on_scroll_y_changed(self):
        """滚动条值改变时更新绘图"""
        self._update_y_viewport()

    def _on_slide_region_changed(self):
        """滑动条时间长度改变时更新绘图"""
        self._update_x_viewport_size()

    def _on_zoom_in_triggered(self):
        """放大信号"""
        if 1 / 64.0 < self.zoom_ratio <= 64.0:
            self.zoom_out_action.setEnabled(True)
            self.zoom_ratio /= 2.0
        self._update_y_viewport_size()
        if self.zoom_ratio <= 1 / 64.0:
            self.zoom_in_action.setDisabled(True)

    def _on_zoom_out_triggered(self):
        """缩小信号"""
        if 1 / 64.0 <= self.zoom_ratio < 64.0:
            self.zoom_in_action.setEnabled(True)
            self.zoom_ratio *= 2.0
        self._update_y_viewport_size()
        if self.zoom_ratio >= 64.0:
            self.zoom_out_action.setDisabled(True)

    def _update_x_viewport(self):
        """更新横坐标视口"""
        if self.x_data is None:
            return
        self.x_viewport_start = self.scroll_x.value()
        self._update_display()

    def _update_y_viewport(self):
        """更新纵坐标视口"""
        self.y_viewport_start = self.scroll_y.value() / 10000.0
        self._update_display()

    def _update_x_viewport_size(self):
        """更新横坐标视口大小"""
        if self.x_data is None:
            return

        self.x_viewport_size = self.slide_region.value()
        max_start = self.duration - self.x_viewport_size

        self.scroll_x.setMaximum(max(0, max_start))
        self.scroll_x.setPageStep(self.x_viewport_size)

        if self.x_viewport_start > max_start:
            self.x_viewport_start = max_start
            self.scroll_x.setValue(self.x_viewport_start)

        self._update_display()

    def _update_y_viewport_size(self):
        """更新纵坐标视口大小"""
        if self.x_data is None:
            return

        y_min = np.nanmin(self.y_data)
        y_max = np.nanmax(self.y_data)

        if self.zoom_ratio < 1.0:
            yrange = y_max - y_min
            self.y_viewport_size = yrange * self.zoom_ratio
            max_start = y_max - self.y_viewport_size
            center = (y_max + y_min - self.y_viewport_size) / 2
            self.y_viewport_start = center

            self.scroll_y.setMinimum(int(y_min * 10000))
            self.scroll_y.setMaximum(int(max_start * 10000))
            self.scroll_y.setPageStep(int(self.y_viewport_size * 10000))
            self.scroll_y.setValue(int(self.y_viewport_start * 10000))
            self.scroll_y.setEnabled(True)
        elif self.zoom_ratio == 1.0:
            self.y_viewport_size = y_max - y_min
            self.y_viewport_start = y_min
            self.scroll_y.setMinimum(int(y_min * 10000))
            self.scroll_y.setMaximum(int((y_max - self.y_viewport_size) * 10000))
            self.scroll_y.setValue(int(self.y_viewport_start * 10000))
            self.scroll_y.setPageStep(int(self.y_viewport_size * 10000))
            self.scroll_y.setEnabled(False)
        else:
            self.scroll_y.setEnabled(False)

        self._update_display()

    def _update_display(self):
        """更新显示"""
        if self.x_data is None or self.y_data is None:
            return

        x_start = max(0, self.x_viewport_start)
        x_end = min(x_start + self.x_viewport_size, len(self.x_data) - 1)

        y_min = self.y_viewport_start
        y_max = y_min + self.y_viewport_size

        if self.zoom_ratio > 1.0:
            y_center = (y_min + y_max) / 2
            y_diff = (y_max - y_min) / 2
            y_min = y_center - y_diff * self.zoom_ratio
            y_max = y_center + y_diff * self.zoom_ratio

        self.plot_item.setXRange(self.x_data[x_start], self.x_data[x_end], padding=0)
        self.plot_item.setYRange(y_min, y_max, padding=0)

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
            print(f"确认修改，返回 {len(result)} 个数据点")
            self.result_signal.emit(self.channel, result)
            self.hide()
            self.close()
            if self.parent() is not None:
                self.parent().close()
