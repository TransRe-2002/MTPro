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

        # 添加原始数据存储
        self.origin_y_data: Optional[np.ndarray] = None

        # 添加删除历史记录
        self.deletion_history: List[List[int]] = []  # 存储每次删除的索引列表

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
        self.is_over_region: bool = False  # 鼠标是否在区域上

        # 选择矩形图形项
        self.selection_rect_item: Optional[QtWidgets.QGraphicsRectItem] = None

        # 统一的高亮点颜色
        self.highlight_color = (255, 150, 0, 200)  # 橙色
        self.region_highlight_color = (255, 100, 100, 150)  # 红色区域高亮

        # 设置视图
        self.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.setBackground('w')

        # 启用鼠标跟踪
        self.setMouseTracking(True)

        # 启用键盘焦点
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

    def setData(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """设置绘图数据"""
        self.x_data = np.array(x_data, dtype=np.float64)
        self.origin_y_data = np.array(y_data, dtype=np.float64)  # 保存原始数据
        self.y_data = self.origin_y_data.copy()  # 使用副本进行操作

        # 清除并重新绘制
        self.clear()
        self.regions.clear()
        self.selected_points.clear()
        self.deletion_history.clear()  # 清空历史记录
        self.plot(self.x_data, self.y_data, pen='b')

        # 根据数据点数设置选择模式
        self.updateSelectionModeBasedOnData()


    def updateSelectionModeBasedOnData(self) -> None:
        """根据数据点数更新选择模式"""
        if self.x_data is not None and len(self.x_data) > 2000:
            # 数据点大于2000，强制放大模式
            self.setSelectionMode('zoom', force=True)
        else:
            # 保持当前模式
            self.setSelectionMode(self.selection_mode)


    def setSelectionMode(self, mode: str, force: bool = False) -> None:
        """设置选择模式"""
        if self.x_data is not None and len(self.x_data) > 2000 and not force:
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
        self.clearSelectionRect()

        # 更新高亮显示
        self.highlightSelectedPoints()


    def clearSelectionRect(self) -> None:
        """清除选择矩形"""
        if self.selection_rect_item is not None:
            self.getViewBox().removeItem(self.selection_rect_item)
            self.selection_rect_item = None


    def removeSelectedRegion(self) -> None:
        """移除当前选中的区域"""
        if not self.regions:
            return

        # 移除最后一个区域
        region = self.regions.pop()
        self.removeItem(region)
        self.onRegionChanged()


    def onRegionChanged(self) -> None:
        """当区域发生变化时调用"""
        selected_indices = self.getSelectedIndices()
        self.selectionChanged.emit(selected_indices)

        # 自动更新高亮显示
        self.highlightSelectedPoints()


    def getSelectedIndices(self) -> List[int]:
        """获取所有被选择的数据索引（自动去重）"""
        if self.x_data is None or len(self.x_data) == 0:
            return []

        all_indices_set = set()

        # 添加区域选择的点
        for region in self.regions:
            try:
                xmin, xmax = region.getRegion()
                # 使用布尔索引找到在区域内的点
                mask = (self.x_data >= xmin) & (self.x_data <= xmax)
                indices = np.where(mask)[0]
                all_indices_set.update(indices.tolist())
            except Exception as e:
                print(f"计算区域索引时出错: {e}")
                continue

        # 添加框选的点
        all_indices_set.update(self.selected_points)

        # 将集合转换回排序后的列表
        all_indices = sorted(all_indices_set)
        return all_indices


    def outputSelectedIndices(self) -> List[int]:
        """输出所有被选择的索引"""
        indices = self.getSelectedIndices()
        print(f"选中的索引数量: {len(indices)}")
        print(f"具体索引: {indices}")
        return indices


    def addRegion(self, start_x: Optional[float] = None) -> None:
        """添加新的选择区域"""
        # 检查数据点数量限制
        if self.x_data is not None and len(self.x_data) > 2000:
            print("数据点超过2000，禁止添加区域")
            return

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

        # 创建并配置区域
        region = pg.LinearRegionItem(values=[region_start, region_end], movable=True)

        # 设置区域样式 - 使用更美观的颜色
        border_pen = pg.mkPen(color=(65, 105, 225), width=7)  # 皇家蓝
        region.lines[0].setPen(border_pen)
        region.lines[1].setPen(border_pen)
        region.setBrush(pg.mkBrush(135, 206, 250, 80))  # 浅天蓝填充

        # 添加到图形和管理列表
        self.addItem(region)
        self.regions.append(region)

        # 连接区域变化信号
        region.sigRegionChanged.connect(self.onRegionChanged)

        self.onRegionChanged()
        print(f"创建区域: [{region_start:.2f}, {region_end:.2f}]")


    def isMouseOverRegion(self, pos: QtCore.QPointF) -> bool:
        """检查鼠标是否在区域上（包括区域线条）"""
        view = self.getViewBox()
        if view is None or not self.regions:
            return False

        try:
            scene_pos = view.mapSceneToView(pos)

            for region in self.regions:
                rgn = region.getRegion()
                # 检查是否在区域内
                if rgn[0] <= scene_pos.x() <= rgn[1]:
                    return True

                # 检查是否在区域线条附近（容差范围）
                view_range = view.viewRange()[0]
                x_range = view_range[1] - view_range[0]
                tolerance = x_range * 0.01  # 1%的视图宽度作为容差

                if (abs(scene_pos.x() - rgn[0]) < tolerance or
                        abs(scene_pos.x() - rgn[1]) < tolerance):
                    return True
        except Exception:
            pass

        return False

    def isMouseOverAutoScaleButton(self, pos: QtCore.QPointF) -> bool:
        """检查鼠标是否在自动缩放按钮（A键）上"""
        view_box = self.getViewBox()
        if view_box is None:
            return False

        try:
            # 获取自动缩放按钮的位置和大小
            # 通常A键在左下角，大小约为30x30像素
            button_rect = QtCore.QRectF(1, self.height() - 31, 30, 30)

            # 检查鼠标位置是否在按钮区域内
            return button_rect.contains(pos)
        except Exception:
            return False

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        # 检查是否在自动缩放按钮上（最高优先级）
        if self.isMouseOverAutoScaleButton(ev.position()):
            # 交给父类处理（自动缩放功能）
            super().mousePressEvent(ev)
            return

        # 检查是否在区域上 - 这个检查要提前
        self.is_over_region = self.isMouseOverRegion(ev.scenePosition())

        if ev.button() == QtCore.Qt.RightButton:
            # 右键处理：优先处理区域删除
            scene_pos = ev.scenePosition()
            view = self.getViewBox()

            if view is not None and self.x_data is not None:
                try:
                    pos = view.mapSceneToView(scene_pos)

                    # 检查是否点击了某个区域（优先处理区域删除）
                    for region in reversed(self.regions):
                        rgn = region.getRegion()
                        # 检查是否在区域内或边界线附近
                        view_range = view.viewRange()[0]
                        x_range = view_range[1] - view_range[0]
                        tolerance = x_range * 0.01  # 1%的视图宽度作为容差

                        # 检查是否在区域内或边界线容差范围内
                        if (rgn[0] <= pos.x() <= rgn[1] or
                                abs(pos.x() - rgn[0]) < tolerance or
                                abs(pos.x() - rgn[1]) < tolerance):
                            self.regions.remove(region)
                            self.removeItem(region)
                            self.onRegionChanged()
                            ev.accept()
                            return

                    # 如果没有点击在区域上，且在选择模式下，进行框选取消选择
                    if self.selection_mode == 'select' and not self.is_over_region:
                        self.start_point = ev.position()
                        self.is_selecting = True
                        ev.accept()
                        return

                except Exception as e:
                    print(f"右键处理错误: {e}")

            ev.accept()
            return

        elif ev.button() == QtCore.Qt.LeftButton:
            # 左键处理：如果鼠标在区域上，交给区域处理
            if self.is_over_region:
                # 让区域处理拖动事件
                super().mousePressEvent(ev)
                return

            # 选择模式下进行框选
            if self.selection_mode == 'select':
                self.start_point = ev.position()
                self.is_selecting = True
                ev.accept()
                return
            else:
                # 放大模式，交给父类处理
                super().mousePressEvent(ev)
                return

        super().mousePressEvent(ev)


    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        """处理鼠标移动事件"""
        # 更新鼠标是否在区域上的状态
        self.is_over_region = self.isMouseOverRegion(ev.scenePosition())

        # 当鼠标在区域上时改变光标形状
        if self.is_over_region:
            self.setCursor(QtGui.QCursor(QtCore.Qt.SizeHorCursor))
        else:
            self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

        if self.is_selecting and self.start_point is not None and not self.is_over_region:
            self.end_point = ev.position()
            self.updateSelectionRect()
            ev.accept()
            return

        super().mouseMoveEvent(ev)


    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        """处理鼠标释放事件"""
        # 检查是否在自动缩放按钮上
        if self.isMouseOverAutoScaleButton(ev.position()):
            # 交给父类处理
            super().mouseReleaseEvent(ev)
            return

        if self.is_selecting and self.start_point is not None and not self.is_over_region:
            self.end_point = ev.position()
            self.is_selecting = False

            # 处理框选逻辑
            self.processRectangleSelection(ev.button() == QtCore.Qt.RightButton)

            # 清除选择矩形
            self.clearSelectionRect()

            ev.accept()
            return

        super().mouseReleaseEvent(ev)

    def updateSelectionRect(self) -> None:
        """更新选择矩形显示 - 修复坐标转换问题"""
        if self.start_point is None or self.end_point is None:
            return

        view_box = self.getViewBox()
        if view_box is None:
            return

        try:
            # 修复：使用正确的坐标转换方法
            # 获取视图的边界矩形
            view_rect = view_box.viewRect()

            # 将鼠标位置转换为视图坐标（确保在正确坐标系中）
            start_view = view_box.mapSceneToView(self.start_point)
            end_view = view_box.mapSceneToView(self.end_point)

            # 确保坐标在视图范围内
            x_min = max(min(start_view.x(), end_view.x()), view_rect.left())
            x_max = min(max(start_view.x(), end_view.x()), view_rect.right())

            # 修复：注意y轴方向，视图坐标中y向上增加，而屏幕坐标中y向下增加
            # 所以我们需要确保y坐标的正确顺序
            if start_view.y() < end_view.y():
                y_min = start_view.y()
                y_max = end_view.y()
            else:
                y_min = end_view.y()
                y_max = start_view.y()

            width = x_max - x_min
            height = y_max - y_min

            # 创建或更新选择矩形
            if self.selection_rect_item is None:
                self.selection_rect_item = QtWidgets.QGraphicsRectItem(
                    QtCore.QRectF(x_min, y_min, width, height)
                )
                # 使用更美观的样式
                pen = QtGui.QPen(QtGui.QColor(30, 144, 255, 180))  # 半透明道奇蓝
                pen.setWidth(0)
                pen.setStyle(QtCore.Qt.DashLine)
                self.selection_rect_item.setPen(pen)

                # 半透明填充
                brush = QtGui.QBrush(QtGui.QColor(135, 206, 250, 30))
                self.selection_rect_item.setBrush(brush)
                self.selection_rect_item.setZValue(10)
                self.getViewBox().addItem(self.selection_rect_item)
            else:
                self.selection_rect_item.setRect(QtCore.QRectF(x_min, y_min, width, height))

        except Exception as e:
            print(f"更新选择矩形错误: {e}")

    def processRectangleSelection(self, is_right_click: bool = False) -> None:
        """处理矩形选择逻辑 - 统一索引类型为int"""
        if self.start_point is None or self.end_point is None:
            return

        view_box = self.getViewBox()
        if view_box is None or self.x_data is None or self.y_data is None:
            return

        try:
            # 将鼠标位置转换为视图坐标
            start_view = view_box.mapSceneToView(self.start_point)
            end_view = view_box.mapSceneToView(self.end_point)

            # 确保坐标正确排序
            x_min, x_max = sorted([start_view.x(), end_view.x()])
            y_min, y_max = sorted([start_view.y(), end_view.y()])

            # 找到在矩形内的点
            mask = (
                    (self.x_data >= x_min) &
                    (self.x_data <= x_max) &
                    (self.y_data >= y_min) &
                    (self.y_data <= y_max)
            )

            # 统一索引类型：将np.int64转换为Python int
            indices = set(int(i) for i in np.where(mask)[0])

            if is_right_click:
                # 右键取消选择：从已选点中移除
                self.selected_points -= indices
            else:
                # 左键选择：累积添加到已选点
                self.selected_points.update(indices)

            # 立即更新显示
            self.highlightSelectedPoints()
            self.onRegionChanged()

        except Exception as e:
            print(f"框选计算错误: {e}")

    def highlightSelectedPoints(self) -> None:
        """高亮显示选中的数据点 - 修复显示问题"""
        # 获取所有数据项并移除之前的高亮点
        for item in self.items():
            if hasattr(item, 'is_highlight') and item.is_highlight:
                self.removeItem(item)

        # 显示所有选中的点（包括区域选择和框选）
        selected_indices = self.getSelectedIndices()

        if selected_indices and self.x_data is not None and self.y_data is not None:
            # 确保索引不超出数据范围
            valid_indices = [i for i in selected_indices if i < len(self.x_data) and i < len(self.y_data)]

            if valid_indices:
                # 使用统一的颜色标记所有选中的点
                scatter = pg.ScatterPlotItem(
                    x=self.x_data[valid_indices],
                    y=self.y_data[valid_indices],
                    pen=pg.mkPen(color=(0, 0, 0), width=1),  # 黑色边框
                    brush=pg.mkBrush(255, 150, 0, 200),  # 橙色填充
                    size=10,
                    symbol='o'
                )
                scatter.is_highlight = True
                self.addItem(scatter)

                # 打印高亮点信息
                print(f"高亮显示 {len(valid_indices)} 个点")

    def modifySelectedData(self) -> None:
        """修改选中的数据点（设置为NaN）"""
        if self.x_data is None or self.y_data is None:
            return

        selected_indices = self.getSelectedIndices()

        if selected_indices:
            # 记录删除历史（用于撤销）
            self.deletion_history.append(selected_indices.copy())

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

    def revertToOriginalData(self) -> None:
        """还原到原始数据"""
        if self.origin_y_data is not None and self.x_data is not None:
            self.y_data = self.origin_y_data.copy()
            self.deletion_history.clear()

            # 清除并重新绘制
            self.clear()
            self.regions.clear()
            self.selected_points.clear()
            self.plot(self.x_data, self.y_data, pen='b')

            self.updateSelectionModeBasedOnData()
            self.onRegionChanged()
            print("已还原到原始数据")

    def undoLastDeletion(self) -> None:
        """撤销上一次删除操作"""
        if not self.deletion_history or self.x_data is None or self.y_data is None:
            print("没有可撤销的操作")
            return

        # 获取最后一次删除的索引
        last_deletion = self.deletion_history.pop()

        # 还原这些点的数据（从原始数据中恢复）
        if self.origin_y_data is not None:
            self.y_data[last_deletion] = self.origin_y_data[last_deletion]

        # 重新绘制
        self.clear()
        self.plot(self.x_data, self.y_data, pen='b')

        # 清除选择状态
        self.regions.clear()
        self.selected_points.clear()

        self.onRegionChanged()
        print(f"已撤销上一次删除操作，恢复了 {len(last_deletion)} 个点")

    def confirmAndExit(self) -> np.ndarray:
        """确认修改并返回处理后的数据"""
        if self.y_data is not None:
            # 更新原始数据为当前修改后的数据
            self.origin_y_data = self.y_data.copy()
            print("修改已确认并保存")
            return self.y_data
        return np.array([])

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:
        """处理鼠标双击事件 - 在两种模式下都允许创建区域"""
        # 检查是否在自动缩放按钮上
        if self.isMouseOverAutoScaleButton(ev.position()):
            # 交给父类处理
            super().mouseDoubleClickEvent(ev)
            return

        if ev.button() == QtCore.Qt.LeftButton:
            # 检查数据点数量限制
            if self.x_data is not None and len(self.x_data) > 2000:
                print("数据点超过2000，禁止创建区域")
                ev.accept()
                return

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
                        ev.accept()
                        return

                except Exception as e:
                    print(f"双击创建区域错误: {e}")

        super().mouseDoubleClickEvent(ev)

    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        """处理键盘事件"""
        if ev.key() == QtCore.Qt.Key_Z and ev.modifiers() == QtCore.Qt.ControlModifier:
            self.undoLastDeletion()
            ev.accept()
            return
        super().keyPressEvent(ev)
    

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PyQtGraph 区域选择示例（支持框选和区域选择）")
        self.setGeometry(100, 100, 1000, 700)

        # 创建中央部件
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # 创建按钮布局
        button_layout = QtWidgets.QHBoxLayout()

        # 创建模式切换按钮和标签
        self.mode_toggle_btn = QtWidgets.QPushButton("切换为选点模式")
        self.mode_label = QtWidgets.QLabel("当前模式: 框选放大")

        # 创建功能按钮
        self.add_region_btn = QtWidgets.QPushButton("添加选择区域")
        self.remove_region_btn = QtWidgets.QPushButton("移除最后区域")
        self.revert_btn = QtWidgets.QPushButton("还原数据")  # 新增还原按钮
        self.undo_btn = QtWidgets.QPushButton("撤销删除")  # 新增撤销按钮
        self.clear_all_btn = QtWidgets.QPushButton("清空所有选择")
        self.modify_data_btn = QtWidgets.QPushButton("修改选中数据")
        self.confirm_btn = QtWidgets.QPushButton("确定")  # 修改为确定按钮

        # 添加按钮到布局
        button_layout.addWidget(self.mode_toggle_btn)
        button_layout.addWidget(self.mode_label)
        button_layout.addStretch(1)
        button_layout.addWidget(self.add_region_btn)
        button_layout.addWidget(self.remove_region_btn)
        button_layout.addWidget(self.revert_btn)  # 添加还原按钮
        button_layout.addWidget(self.undo_btn)  # 添加撤销按钮
        button_layout.addWidget(self.clear_all_btn)
        button_layout.addWidget(self.modify_data_btn)
        button_layout.addWidget(self.confirm_btn)  # 添加确定按钮到最右边

        # 创建状态标签和说明
        self.status_label = QtWidgets.QLabel("当前选中: 0 个数据点")
        self.instruction_label = QtWidgets.QLabel(
            "操作说明: 左键双击创建区域 | 右键点击区域删除 | 左键框选添加点 | 右键框选取消点 | 中键移动视图 | Ctrl+Z撤销删除"
        )
        # 创建绘图组件
        self.plot_widget = RegionSelectionPlotWidget()

        # 添加到主布局
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.instruction_label)
        main_layout.addWidget(self.plot_widget)

        # 连接信号槽
        self.mode_toggle_btn.clicked.connect(self.toggleSelectionMode)
        self.add_region_btn.clicked.connect(self.plot_widget.addRegion)
        self.remove_region_btn.clicked.connect(self.plot_widget.removeSelectedRegion)
        self.revert_btn.clicked.connect(self.plot_widget.revertToOriginalData)  # 连接还原按钮
        self.undo_btn.clicked.connect(self.plot_widget.undoLastDeletion)  # 连接撤销按钮
        self.clear_all_btn.clicked.connect(self.clearAllSelections)
        self.modify_data_btn.clicked.connect(self.plot_widget.modifySelectedData)
        self.confirm_btn.clicked.connect(self.confirmAndClose)  # 连接确定按钮
        self.plot_widget.selectionChanged.connect(self.onSelectionChanged)

        # 生成示例数据
        self.generateSampleData()

    def toggleSelectionMode(self) -> None:
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

        # 更新按钮状态
        data_points = len(self.plot_widget.x_data) if self.plot_widget.x_data is not None else 0
        if data_points > 2000:
            # 数据点超过2000，强制放大模式
            self.mode_toggle_btn.setEnabled(False)
            self.add_region_btn.setEnabled(False)
            self.mode_label.setText("当前模式: 框选放大（数据量>2000，强制放大模式）")
        else:
            # 数据点少于2000，允许所有功能
            self.mode_toggle_btn.setEnabled(True)
            self.add_region_btn.setEnabled(True)

    def confirmAndClose(self) -> None:
        """确认修改并关闭窗口"""
        result = self.plot_widget.confirmAndExit()
        if result.size > 0:
            # 可以在这里处理返回的数据，比如保存到文件或传递给其他模块
            print(f"确认修改，返回 {len(result)} 个数据点")
            self.close()  # 关闭窗口

    def generateSampleData(self) -> None:
        """生成示例数据"""
        # 创建测试数据
        data_points = 1500  # 测试小于2000点的情况

        x = np.linspace(0, 10, data_points)
        y = (50 * np.sin(0.1 * x) +
             25 * np.cos(0.3 * x) +
             10 * np.random.normal(0, 1, len(x)))

        self.plot_widget.setData(x, y)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())