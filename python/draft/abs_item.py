from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt
from PySide6.QtWidgets import QTreeView, QApplication
import sys


class EmployeeModel(QAbstractItemModel):
    def __init__(self, data_dict, parent=None):
        super().__init__(parent)
        # 将字典转换为列表以便按行索引
        self._data = list(data_dict.items())  # [(id1, data1), (id2, data2), ...]
        self._headers = ["ID", "姓名", "部门", "工资"]

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0  # 这里示例是平面列表，无子项
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return 4  # ID列 + 3个数据列

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        row, col = index.row(), index.column()
        emp_id, emp_data = self._data[row]

        if role == Qt.DisplayRole:
            if col == 0:  # ID列
                return str(emp_id)
            elif col == 1:  # 姓名
                return emp_data.get("name", "")
            elif col == 2:  # 部门
                return emp_data.get("dept", "")
            elif col == 3:  # 工资
                return str(emp_data.get("salary", ""))

        # 将id存储在UserRole中，供外部访问
        elif role == Qt.UserRole + 1 and col == 0:
            return emp_id

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._headers[section]
        return None

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        return self.createIndex(row, column)

    def parent(self, index):
        return QModelIndex()  # 平面结构，无父项

    def get_data_by_index(self, index):
        """通过模型索引获取完整数据"""
        if not index.isValid():
            return None
        emp_id = self.data(self.index(index.row(), 0), Qt.UserRole + 1)
        row = index.row()
        if 0 <= row < len(self._data):
            return self._data[row]  # 返回 (id, data) 元组
        return None


# 使用示例
if __name__ == "__main__":
    employee_data = {
        1001: {"name": "张三", "dept": "研发部", "salary": 15000},
        1002: {"name": "李四", "dept": "市场部", "salary": 12000},
    }

    app = QApplication(sys.argv)
    model = EmployeeModel(employee_data)

    view = QTreeView()
    view.setModel(model)
    view.clicked.connect(lambda idx:
                         print(f"点击了: {model.get_data_by_index(idx)}"))
    view.show()

    sys.exit(app.exec())
