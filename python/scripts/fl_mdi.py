import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QMdiArea, QMdiSubWindow,
    QTreeView, QWidget, QVBoxLayout, QLabel, QTextEdit,
    QToolBar, QStatusBar, QDockWidget
)
from PySide6.QtGui import QStandardItemModel, QStandardItem, QAction, QDrag
from PySide6.QtCore import Qt, QMimeData, QByteArray


class DraggableTreeView(QTreeView):
    """支持拖拽的自定义树视图"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)  # 启用视图的拖拽功能
        self.setEditTriggers(QTreeView.NoEditTriggers)  # 禁用编辑触发
        self.setSelectionMode(QTreeView.SingleSelection)  # 单选模式更符合拖拽直觉

    def startDrag(self, supportedActions):
        """重写开始拖拽的方法 - 修复版本"""
        index = self.currentIndex()
        if not index.isValid():
            return

        # 获取被拖拽项的数据
        item = self.model().itemFromIndex(index)
        # 这里我们假设项的数据角色（Qt.UserRole + 1）存储了窗口类型标识符
        widget_type = item.data(Qt.UserRole + 1)

        if widget_type:
            # 创建MIME数据，用于在拖拽过程中传递信息
            mime_data = QMimeData()
            # 使用自定义MIME类型来传递窗口类型
            mime_data.setData("application/x-widgettype", QByteArray(widget_type.encode()))

            # 修复：使用 QDrag 而不是 dragObject()
            drag = QDrag(self)
            drag.setMimeData(mime_data)

            # 开始执行拖拽操作
            drag.exec(supportedActions, Qt.CopyAction)


class CustomMdiArea(QMdiArea):
    """支持接收拖拽放置的自定义MDI区域"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)  # 启用放置接受

    def dragEnterEvent(self, event):
        """当拖拽进入区域时，判断数据是否可接受"""
        if event.mimeData().hasFormat("application/x-widgettype"):
            event.acceptProposedAction()  # 接受此操作
        else:
            event.ignore()  # 忽略不相关的拖拽

    def dropEvent(self, event):
        """当放置发生时，创建对应的子窗口"""
        if event.mimeData().hasFormat("application/x-widgettype"):
            # 获取传递过来的窗口类型标识符
            widget_type = event.mimeData().data("application/x-widgettype").data().decode()
            # 获取鼠标释放的位置，作为新窗口的初始位置
            position = event.position().toPoint()

            # 根据类型创建不同的子窗口
            sub_window = self.create_subwindow(widget_type, position)
            if sub_window:
                self.addSubWindow(sub_window)
                sub_window.show()
                # 将活动窗口设置为新创建的窗口
                self.setActiveSubWindow(sub_window)
                event.acceptProposedAction()

    def create_subwindow(self, widget_type, position):
        """根据类型标识符创建子窗口和内容控件"""
        sub_window = QMdiSubWindow()
        sub_window.setWindowTitle(f"窗口 - {widget_type}")
        # 设置子窗口的初始位置和大小
        sub_window.setGeometry(position.x(), position.y(), 400, 300)

        content_widget = None
        if widget_type == "TextEditor":
            content_widget = QTextEdit()
            content_widget.setPlainText("这是一个文本编辑器窗口。\n由拖拽创建。")
        elif widget_type == "PianoRoll":
            content_widget = QWidget()
            layout = QVBoxLayout(content_widget)
            label = QLabel("🎹 钢琴卷帘窗口 (模拟界面)")
            layout.addWidget(label)
        elif widget_type == "Mixer":
            content_widget = QWidget()
            layout = QVBoxLayout(content_widget)
            label = QLabel("🔊 混音器窗口 (模拟界面)")
            layout.addWidget(label)
        else:
            content_widget = QTextEdit()
            content_widget.setPlainText(f"未知工具类型: {widget_type}")

        if content_widget:
            sub_window.setWidget(content_widget)
            return sub_window
        return None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FL Studio 风格界面：可拖拽树形视图 + MDI")
        self.resize(1000, 700)

        # 创建MDI区域（主工作区）
        self.mdi_area = CustomMdiArea()
        self.setCentralWidget(self.mdi_area)

        # 创建树形视图（功能库导航）
        self.setup_tree_view()

        # 创建工具栏和状态栏
        self.setup_ui()

    def setup_tree_view(self):
        """创建并设置树形视图（作为 dock widget 放在左侧）"""
        # 创建可拖拽的树视图
        self.tree_view = DraggableTreeView()

        # 创建数据模型
        self.tree_model = QStandardItemModel()
        self.tree_model.setHorizontalHeaderLabels(["功能库"])

        # 构建树形结构数据
        # 第一层：分类
        cat_sound = QStandardItem("音源")
        cat_effects = QStandardItem("效果器")
        cat_midi = QStandardItem("MIDI 工具")

        # 设置分类项的标志，允许拖拽
        for category in [cat_sound, cat_effects, cat_midi]:
            category.setFlags(category.flags() | Qt.ItemIsDragEnabled)

        # 第二层：具体功能项，并为每个项设置自定义数据（窗口类型标识符）
        # 音源分类下的项
        item_vst = QStandardItem("VST 乐器")
        item_vst.setData("VSTHost", Qt.UserRole + 1)  # 设置数据类型
        item_vst.setToolTip("拖拽到右侧创建 VST 宿主窗口")
        cat_sound.appendRow(item_vst)

        item_sample = QStandardItem("采样器")
        item_sample.setData("Sampler", Qt.UserRole + 1)
        cat_sound.appendRow(item_sample)

        # 效果器分类下的项
        item_eq = QStandardItem("均衡器")
        item_eq.setData("EQ", Qt.UserRole + 1)
        cat_effects.appendRow(item_eq)

        item_reverb = QStandardItem("混响")
        item_reverb.setData("Reverb", Qt.UserRole + 1)
        cat_effects.appendRow(item_reverb)

        item_delay = QStandardItem("延迟")
        item_delay.setData("Delay", Qt.UserRole + 1)
        cat_effects.appendRow(item_delay)

        # MIDI 工具分类下的项
        item_piano = QStandardItem("钢琴卷帘")
        item_piano.setData("PianoRoll", Qt.UserRole + 1)
        cat_midi.appendRow(item_piano)

        item_arp = QStandardItem("琶音器")
        item_arp.setData("Arpeggiator", Qt.UserRole + 1)
        cat_midi.appendRow(item_arp)

        # 将分类添加到根节点
        root = self.tree_model.invisibleRootItem()
        root.appendRow(cat_sound)
        root.appendRow(cat_effects)
        root.appendRow(cat_midi)

        # 将模型设置到视图
        self.tree_view.setModel(self.tree_model)
        # 默认展开所有第一级分类
        self.tree_view.expandAll()

        # 创建一个左侧停靠窗口来容纳树视图
        dock_widget = QDockWidget("功能面板", self)
        dock_widget.setWidget(self.tree_view)
        dock_widget.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_widget)

    def setup_ui(self):
        """设置工具栏和状态栏"""
        # 工具栏
        toolbar = QToolBar("主工具栏")
        self.addToolBar(toolbar)
        new_action = QAction("新建", self)
        toolbar.addAction(new_action)

        # 状态栏
        status_bar = QStatusBar(self)
        self.setStatusBar(status_bar)
        status_bar.showMessage("就绪：从左侧功能库拖拽项目到右侧工作区")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()