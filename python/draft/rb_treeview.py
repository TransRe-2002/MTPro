from PySide6.QtWidgets import (
    QApplication, QTreeView, QMenu, QMainWindow
)
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt
import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tree = QTreeView()
        self.setCentralWidget(self.tree)

        # 构建示例数据
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["名称"])
        for i in range(3):
            parent = QStandardItem(f"父节点 {i}")
            for j in range(3):
                child = QStandardItem(f"子节点 {i}-{j}")
                parent.appendRow(child)
            model.appendRow(parent)
        self.tree.setModel(model)

        # ① 开启自定义右键菜单模式
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        # ② 连接信号
        self.tree.customContextMenuRequested.connect(self.on_context_menu)

    def on_context_menu(self, pos):
        index = self.tree.indexAt(pos)

        # ③ 判断是否为有效索引
        if not index.isValid():
            return

        # ④ 判断是否为父节点（hasChildren 或 parent 无效均可）
        is_parent = self.tree.model().hasIndex(0, 0, index)  # 有子项 → 父节点

        if is_parent:
            menu = QMenu(self)
            action_add    = menu.addAction("➕ 添加子节点")
            action_rename = menu.addAction("✏️ 重命名")
            menu.addSeparator()
            action_delete = menu.addAction("🗑️ 删除")

            action = menu.exec(self.tree.viewport().mapToGlobal(pos))

            if action == action_add:
                self.handle_add(index)
            elif action == action_rename:
                self.handle_rename(index)
            elif action == action_delete:
                self.handle_delete(index)

    def handle_add(self, index):
        model = self.tree.model()
        parent_item = model.itemFromIndex(index)
        parent_item.appendRow(QStandardItem("新子节点"))
        self.tree.expand(index)

    def handle_rename(self, index):
        self.tree.edit(index)  # 触发内联编辑

    def handle_delete(self, index):
        model = self.tree.model()
        model.removeRow(index.row(), index.parent())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())