import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QTreeView, QVBoxLayout, QWidget, QHeaderView
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt, QDateTime


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("带复选框的目录树 - PySide6")
        self.resize(400, 300)

        # 1. 创建模型(Model)和视图(View)
        self.model = QStandardItemModel()
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.model)

        # 2. 设置表头（可选）
        self.model.setHorizontalHeaderLabels(['数据名称', '开始时间', '结束时间', '采集点数'])
        # 如果你不需要表头，可以取消下面这行的注释
        # self.tree_view.setHeaderHidden(True)

        # 3. 创建根节点
        self.create_tree_structure()

        # 4. 布局
        layout = QVBoxLayout()
        layout.addWidget(self.tree_view)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def create_tree_structure(self):
        # --- 创建第一层级：目录 ---
        parent1_item = QStandardItem("项目文件夹")
        parent2_item = QStandardItem("资料库")

        # --- 创建第二层级：具体的文件/单元 ---
        # 为第一个目录添加子项
        child1_item = QStandardItem("文件 A")
        child1_item.setCheckable(True)  # 关键：开启复选框
        child1_item.setCheckState(Qt.Unchecked)  # 默认未选中

        child2_item = QStandardItem("文件 B")
        child2_item.setCheckable(True)
        child2_item.setCheckState(Qt.PartiallyChecked)  # 半选状态（如果需要）

        # 将子项添加到父项
        parent1_item.appendRow([child1_item, QStandardItem(QDateTime.currentDateTime().toString())])
        parent1_item.appendRow([child2_item, QStandardItem(QDateTime.currentDateTime().toString())])

        # --- 为第二个目录添加子项 ---
        child3_item = QStandardItem("文件 C")
        child3_item.setCheckable(True)
        child3_item.setCheckState(Qt.Checked)  # 默认选中

        parent2_item.appendRow([child3_item, QStandardItem("完成")])

        # --- 将顶层目录添加到模型 ---
        self.model.appendRow([parent1_item, QStandardItem("主目录")])
        self.model.appendRow([parent2_item, QStandardItem("备份")])

        # --- 展开所有节点 ---
        self.tree_view.expandAll()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())