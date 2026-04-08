import os

os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'

import sys
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore, QtGui
from typing import List, Tuple, Set, Optional, Any

pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class RegionSelectionPlotWidget(pg.PlotWidget):
    selectionChanged = QtCore.Signal(object)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # 存储数据
        self.x_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None

        # 存储所有选择区域
        self.regions: List[pg.LinearRegionItem] = []

        # 存储框选的点（离散选择）
        self.selected_points: Set[int] = set()

        # 选择模式：'zoom' 或 'select'
        self.selection_mode: str = 'zoom'

        # 鼠标状态跟踪
        self.is_selecting: bool = False
        self.start_point: Optional[QtCore.QPointF] = None
        self.end_point: Optional[QtCore.QPointF] = None
        self.is_moving_region: bool = False

        # 选择矩形图形项
        self.selection_rect: Optional[pg.RectROI] = None

        # 双击创建区域的默认宽度（数据范围的10%）
        self.double_click_region_width: float = 0.1

        # 设置视图
        self.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.setBackground('w')

        # 连接区域移动信号
        self.getViewBox().sigRangeChanged.connect(self.onViewRangeChanged)

    def setData(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """设置绘图数据"""
        self.x_data = np.array(x_data, dtype=np.float64)
        self.y_data = np.array(y_data, dtype=np.float64)

        # 清除并重新绘制
        self.clear()
        self.regions.clear()
        self.selected_points.clear()
        self.plot(self.x_data, self.y_data, pen='b')

        # 根据数据点数设置选择模式
        self.updateSelectionModeBasedOnData()

    def updateSelectionModeBasedOnData(self) -> None:
        """根据数据点数更新选择模式"""
        if self.x_data is not None and len(self.x_data) > 2000:
            self.setSelectionMode('zoom')
        else:
            # 保持当前模式，但确保按钮状态正确
            self.setSelectionMode(self.selection_mode)

    def setSelectionMode(self, mode: str) -> None:
        """设置选择模式"""
        if self.x_data is not None and len(self.x_data) > 2000:
            # 强制为放大模式
            self.selection_mode = 'zoom'
            self.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        else:
            self.selection_mode = mode
            if mode == 'zoom':
                self.getViewBox().setMouseMode(pg.ViewBox.RectMode)
            else:  # select mode
                self.getViewBox().setMouseMode(pg.ViewBox.PanMode)

        # 清除选择矩形
        if self.selection_rect is not None:
            self.removeItem(self.selection_rect)
            self.selection_rect = None

        # 更新高亮显示
        self.highlightSelectedPoints()

    def removeSelectedRegion(self) -> None:
        """移除当前选中的区域"""
        if not self.regions:
            return

        # 移除最后一个区域
        region: pg.LinearRegionItem = self.regions.pop()
        self.removeItem(region)
        self.onRegionChanged()

    def onRegionChanged(self) -> None:
        """当区域发生变化时调用"""
        selected_indices: List[int] = self.getSelectedIndices()
        self.selectionChanged.emit(selected_indices)

    def getSelectedIndices(self) -> List[int]:
        """获取所有被选择的数据索引（自动去重）"""
        if self.x_data is None or len(self.x_data) == 0:
            return []

        all_indices_set: Set[int] = set()

        # 添加区域选择的点
        for region in self.regions:
            try:
                xmin, xmax = region.getRegion()
                # 使用布尔索引找到在区域内的点
                mask: np.ndarray = (self.x_data >= xmin) & (self.x_data <= xmax)
                indices: np.ndarray = np.where(mask)[0]
                all_indices_set.update(indices.tolist())
            except Exception as e:
                print(f"计算区域索引时出错: {e}")
                continue

        # 添加框选的点
        all_indices_set.update(self.selected_points)

        # 将集合转换回排序后的列表
        all_indices: List[int] = sorted(all_indices_set)
        return all_indices

    def outputSelectedIndices(self) -> List[int]:
        """输出所有被选择的索引"""
        indices: List[int] = self.getSelectedIndices()
        print(f"选中的索引数量: {len(indices)}")
        print(f"具体索引: {indices}")
        return indices

    def addRegion(self, start_x: Optional[float] = None) -> None:
        """添加新的选择区域（整合按钮和双击创建）"""
        if self.x_data is None or len(self.x_data) == 0:
            return

        # 获取当前视图范围用于计算区域宽度
        view_box = self.getViewBox()
        current_view_range = view_box.viewRange()
        view_x_min, view_x_max = current_view_range[0]
        visible_range_width = view_x_max - view_x_min

        # 计算基于当前视图宽度的10%作为区域宽度
        dynamic_region_width = visible_range_width * 0.1

        if start_x is not None:
            # 双击创建：以点击位置作为起始边界
            # 确保起始位置在数据范围内
            start_x = max(self.x_data.min(), min(self.x_data.max(), start_x))
            region_start = start_x
            region_end = min(self.x_data.max(), start_x + dynamic_region_width)
        else:
            # 按钮创建：默认在数据中间20%位置
            data_range = self.x_data.max() - self.x_data.min()
            region_start = self.x_data.min() + 0.4 * data_range
            region_end = self.x_data.min() + 0.6 * data_range

        # 确保区域有最小宽度
        min_region_width = visible_range_width * 0.02
        if region_end - region_start < min_region_width:
            region_end = region_start + min_region_width

        # 创建并配置区域（统一的创建逻辑）
        region = pg.LinearRegionItem(values=[region_start, region_end], movable=True)

        # 设置区域样式
        border_pen = pg.mkPen(color=(200, 50, 50), width=3)
        region.lines[0].setPen(border_pen)
        region.lines[1].setPen(border_pen)
        region.setBrush(pg.mkBrush(255, 100, 100, 80))

        # 添加到图形和管理列表
        self.addItem(region)
        self.regions.append(region)
        region.sigRegionChanged.connect(self.onRegionChanged)

        # 连接区域移动信号
        for line in region.lines:
            line.sigDragged.connect(self.onRegionLineDragged)
            line.sigDragStart.connect(self.onRegionDragStart)
            line.sigDragFinish.connect(self.onRegionDragFinish)

        self.onRegionChanged()

        print(f"创建区域: [{region_start:.2f}, {region_end:.2f}]")

    def onRegionDragStart(self) -> None:
        """区域拖动开始"""
        self.is_moving_region = True

    def onRegionDragFinish(self) -> None:
        """区域拖动结束"""
        self.is_moving_region = False

    def onRegionLineDragged(self) -> None:
        """区域线条被拖动时更新选择"""
        self.onRegionChanged()

    def onViewRangeChanged(self) -> None:
        """视图范围改变时更新选择矩形"""
        if self.selection_rect is not None and self.is_selecting:
            # 更新选择矩形位置
            self.updateSelectionRect()

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        """处理鼠标点击事件"""
        if ev.button() == QtCore.Qt.RightButton:
            # 右键处理：选择模式下取消选择点
            if self.selection_mode == 'select' and not self.is_moving_region:
                scene_pos = ev.scenePosition()
                view = self.getViewBox()

                if view is not None and self.x_data is not None:
                    try:
                        pos = view.mapSceneToView(scene_pos)

                        # 检查是否点击了某个区域（优先处理区域删除）
                        for region in reversed(self.regions):
                            rgn = region.getRegion()
                            if rgn[0] <= pos.x() <= rgn[1]:
                                self.regions.remove(region)
                                self.removeItem(region)
                                self.onRegionChanged()
                                ev.accept()
                                return

                        # 不是点击区域，则进行框选取消选择
                        self.start_point = ev.position()
                        self.is_selecting = True
                        ev.accept()
                        return

                    except Exception as e:
                        print(f"右键处理错误: {e}")

            ev.accept()
            return

        elif ev.button() == QtCore.Qt.LeftButton:
            # 左键处理
            if self.selection_mode == 'select' and not self.is_moving_region:
                # 选择模式下进行框选
                self.start_point = ev.position()
                self.is_selecting = True
                ev.accept()
                return
            else:
                # 放大模式或其他情况，交给父类处理
                super().mousePressEvent(ev)
                return

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        """处理鼠标移动事件"""
        if self.is_selecting and self.start_point is not None:
            self.end_point = ev.position()
            self.updateSelectionRect()
            ev.accept()
            return

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        """处理鼠标释放事件"""
        if self.is_selecting and self.start_point is not None:
            self.end_point = ev.position()
            self.is_selecting = False

            # 处理框选逻辑
            self.processRectangleSelection(ev.button() == QtCore.Qt.RightButton)

            # 清除选择矩形
            if self.selection_rect is not None:
                self.removeItem(self.selection_rect)
                self.selection_rect = None

            ev.accept()
            return

        super().mouseReleaseEvent(ev)

    def updateSelectionRect(self) -> None:
        """更新选择矩形显示"""
        if self.start_point is None or self.end_point is None:
            return

        view_box = self.getViewBox()
        if view_box is None:
            return

        try:
            start_view = view_box.mapToView(self.start_point)
            end_view = view_box.mapToView(self.end_point)

            x_min = min(start_view.x(), end_view.x())
            x_max = max(start_view.x(), end_view.x())
            y_min = min(start_view.y(), end_view.y())
            y_max = max(start_view.y(), end_view.y())

            width = x_max - x_min
            height = y_max - y_min

            # 创建或更新选择矩形
            if self.selection_rect is None:
                self.selection_rect = pg.RectROI(
                    [x_min, y_min], [width, height],
                    pen=pg.mkPen(color=(0, 100, 255), width=2),
                    movable=False
                )
                self.selection_rect.setZValue(10)
                self.addItem(self.selection_rect)
            else:
                self.selection_rect.setPos([x_min, y_min])
                self.selection_rect.setSize([width, height])

        except Exception as e:
            print(f"更新选择矩形错误: {e}")

    def processRectangleSelection(self, is_right_click: bool = False) -> None:
        """处理矩形选择逻辑"""
        if self.start_point is None or self.end_point is None:
            return

        view_box = self.getViewBox()
        if view_box is None or self.x_data is None:
            return

        try:
            start_view = view_box.mapToView(self.start_point)
            end_view = view_box.mapToView(self.end_point)

            x_min, x_max = sorted([start_view.x(), end_view.x()])
            y_min, y_max = sorted([start_view.y(), end_view.y()])

            # 找到在矩形内的点
            mask = ((self.x_data >= x_min) & (self.x_data <= x_max) &
                    (self.y_data >= y_min) & (self.y_data <= y_max))
            indices = set(np.where(mask)[0])

            if is_right_click:
                # 右键取消选择：从已选点中移除
                self.selected_points -= indices
            else:
                # 左键选择：累积添加到已选点
                self.selected_points.update(indices)

            self.onRegionChanged()
            self.highlightSelectedPoints()

        except Exception as e:
            print(f"框选计算错误: {e}")

    def highlightSelectedPoints(self) -> None:
        """高亮显示选中的数据点"""
        plot_item = self.getPlotItem()

        # 获取所有数据项并移除之前的高亮点
        data_items = plot_item.listDataItems()
        for item in data_items[:]:
            if hasattr(item, 'is_highlight') and item.is_highlight:
                plot_item.removeItem(item)

        # 显示框选的高亮点
        if self.selected_points and self.x_data is not None:
            indices = list(self.selected_points)
            scatter = pg.ScatterPlotItem(
                x=self.x_data[indices],
                y=self.y_data[indices],
                pen=pg.mkPen(None),
                brush=pg.mkBrush(255, 200, 0, 200),  # 橙色高亮
                size=10,
                symbol='o'
            )
            scatter.is_highlight = True
            self.addItem(scatter)

        # 显示区域选择的高亮点
        if self.regions and self.x_data is not None:
            region_indices = set()
            for region in self.regions:
                try:
                    xmin, xmax = region.getRegion()
                    mask = (self.x_data >= xmin) & (self.x_data <= xmax)
                    indices = np.where(mask)[0]
                    region_indices.update(indices)
                except:
                    continue

            if region_indices:
                indices = list(region_indices - self.selected_points)  # 避免重复
                scatter = pg.ScatterPlotItem(
                    x=self.x_data[indices],
                    y=self.y_data[indices],
                    pen=pg.mkPen(None),
                    brush=pg.mkBrush(255, 100, 100, 150),  # 红色高亮
                    size=8,
                    symbol='s'
                )
                scatter.is_highlight = True
                self.addItem(scatter)

    def modifySelectedData(self) -> None:
        """修改选中的数据点（设置为NaN）"""
        if self.x_data is None or self.y_data is None:
            return

        selected_indices = self.getSelectedIndices()

        if selected_indices:
            # 将选中的点设置为NaN
            y_modified = self.y_data.copy()
            y_modified[selected_indices] = np.nan
            self.y_data = y_modified

            # 清除所有图形元素
            self.clear()

            # 重绘曲线
            self.plot(self.x_data, self.y_data, pen='b')

            # 清理选择状态
            self.regions.clear()
            self.selected_points.clear()

            # 更新选择模式
            self.updateSelectionModeBasedOnData()

            self.onRegionChanged()

            print(f"已将 {len(selected_indices)} 个数据点设置为 NaN")

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:
        """处理鼠标双击事件"""
        if ev.button() == QtCore.Qt.LeftButton and self.selection_mode != 'zoom':
            # 获取场景坐标并转换为数据坐标
            scene_pos = ev.scenePosition()
            view_box = self.getViewBox()

            if view_box is not None and self.x_data is not None:
                try:
                    data_pos = view_box.mapSceneToView(scene_pos)
                    click_x = data_pos.x()

                    # 验证点击位置在数据范围内
                    if self.x_data.min() <= click_x <= self.x_data.max():
                        self.addRegion(start_x=click_x)

                except Exception as e:
                    print(f"双击创建区域错误: {e}")

            ev.accept()
            return

        super().mouseDoubleClickEvent(ev)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PyQtGraph 区域选择示例（支持框选和区域选择）")
        self.setGeometry(100, 100, 1000, 700)

        # 创建中央部件
        central_widget: QtWidgets.QWidget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout(central_widget)

        # 创建按钮布局
        button_layout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()

        # 创建模式切换按钮和标签
        self.mode_toggle_btn: QtWidgets.QPushButton = QtWidgets.QPushButton("切换为选点模式")
        self.mode_label: QtWidgets.QLabel = QtWidgets.QLabel("当前模式: 框选放大")
        self.mode_label.setStyleSheet("font-weight: bold; color: blue;")

        # 创建功能按钮
        self.add_region_btn: QtWidgets.QPushButton = QtWidgets.QPushButton("添加选择区域")
        self.remove_region_btn: QtWidgets.QPushButton = QtWidgets.QPushButton("移除最后区域")
        self.output_btn: QtWidgets.QPushButton = QtWidgets.QPushButton("输出选中索引")
        self.clear_all_btn: QtWidgets.QPushButton = QtWidgets.QPushButton("清空所有选择")
        self.modify_data_btn: QtWidgets.QPushButton = QtWidgets.QPushButton("修改选中数据")

        # 添加按钮到布局
        button_layout.addWidget(self.mode_toggle_btn)
        button_layout.addWidget(self.mode_label)
        button_layout.addStretch(1)
        button_layout.addWidget(self.add_region_btn)
        button_layout.addWidget(self.remove_region_btn)
        button_layout.addWidget(self.output_btn)
        button_layout.addWidget(self.clear_all_btn)
        button_layout.addWidget(self.modify_data_btn)

        # 创建状态标签和说明
        self.status_label: QtWidgets.QLabel = QtWidgets.QLabel("当前选中: 0 个数据点")
        self.instruction_label: QtWidgets.QLabel = QtWidgets.QLabel(
            "操作说明: 左键双击创建区域 | 右键点击区域删除 | 左键框选添加点 | 右键框选取消点"
        )
        self.instruction_label.setStyleSheet("color: #666; font-size: 10px;")

        # 创建绘图组件
        self.plot_widget: RegionSelectionPlotWidget = RegionSelectionPlotWidget()

        # 添加到主布局
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.instruction_label)
        main_layout.addWidget(self.plot_widget)

        # 连接信号槽
        self.mode_toggle_btn.clicked.connect(self.toggleSelectionMode)
        self.add_region_btn.clicked.connect(self.plot_widget.addRegion)
        self.remove_region_btn.clicked.connect(self.plot_widget.removeSelectedRegion)
        self.output_btn.clicked.connect(self.plot_widget.outputSelectedIndices)
        self.clear_all_btn.clicked.connect(self.clearAllSelections)
        self.modify_data_btn.clicked.connect(self.plot_widget.modifySelectedData)
        self.plot_widget.selectionChanged.connect(self.onSelectionChanged)

        # 生成示例数据
        self.generateSampleData()

    def generateSampleData(self) -> None:
        """生成示例数据"""
        # 创建不同数据量的测试数据
        data_points = 1500  # 测试小于2000点的情况

        x: np.ndarray = np.linspace(0, 10, data_points)
        y: np.ndarray = (50 * np.sin(0.1 * x) +
                         25 * np.cos(0.3 * x) +
                         10 * np.random.normal(0, 1, len(x)) + 50)

        self.plot_widget.setData(x, y)

    def toggleSelectionMode(self) -> None:
        """切换选择模式"""
        current_mode = self.plot_widget.selection_mode
        if current_mode == 'zoom':
            new_mode = 'select'
            self.mode_toggle_btn.setText("切换为放大模式")
            self.mode_label.setText("当前模式: 框选选点")
            self.mode_label.setStyleSheet("font-weight: bold; color: red;")
        else:
            new_mode = 'zoom'
            self.mode_toggle_btn.setText("切换为选点模式")
            self.mode_label.setText("当前模式: 框选放大")
            self.mode_label.setStyleSheet("font-weight: bold; color: blue;")

        self.plot_widget.setSelectionMode(new_mode)

    def clearAllSelections(self) -> None:
        """清空所有选择"""
        # 清空区域选择
        for region in self.plot_widget.regions[:]:
            self.plot_widget.removeItem(region)
        self.plot_widget.regions.clear()

        # 清空框选点
        self.plot_widget.selected_points.clear()

        self.plot_widget.onRegionChanged()

    def onSelectionChanged(self, selected_indices: List[int]) -> None:
        """当选择发生变化时更新状态"""
        self.status_label.setText(f"当前选中: {len(selected_indices)} 个数据点")

        # 更新按钮状态（数据量大于2000时禁用某些功能）
        data_points = len(self.plot_widget.x_data) if self.plot_widget.x_data is not None else 0
        if data_points > 2000:
            self.mode_toggle_btn.setEnabled(False)
            self.add_region_btn.setEnabled(False)
            self.mode_label.setText("当前模式: 框选放大（数据量>2000，强制放大模式）")
        else:
            self.mode_toggle_btn.setEnabled(True)
            self.add_region_btn.setEnabled(True)


if __name__ == '__main__':
    app: QtWidgets.QApplication = QtWidgets.QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle('Fusion')

    window: MainWindow = MainWindow()
    window.show()

    sys.exit(app.exec())