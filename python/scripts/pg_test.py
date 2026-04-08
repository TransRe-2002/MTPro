import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'

import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph import DateAxisItem
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pandas as pd

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 创建主窗口和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # 创建绘图部件
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

        # 生成示例数据
        x = np.linspace(0, 100, 10000)
        y = 2 * np.sin(np.pi * x) + np.cos(2 * np.pi * x) \
            + 0.5 * np.sin(3 * np.pi * x) + 0.2 * np.cos(4 * np.pi * x) \
            + np.log(5 * x + 1) + np.random.rand(10000) \
            + 2


        # 绘制数据
        self.plot_widget.plot(
            x,
            y,
            pen='b',
        )

        # 创建线性区域选择项
        self.region = pg.LinearRegionItem(values=[3, 7], movable=True)
        self.plot_widget.addItem(self.region)

        # 连接区域变化信号
        self.region.sigRegionChanged.connect(self.region_changed)

        # 显示选中的数据范围
        self.text_item = pg.TextItem("选中区域: 3.00 - 7.00", color='r')
        self.text_item.setPos(3, 0.8)
        self.plot_widget.addItem(self.text_item)

        self.setWindowTitle("PyQtGraph 区域选择示例")
        self.resize(800, 600)

    def region_changed(self):
        # 获取选定区域的范围
        min_val, max_val = self.region.getRegion()

        # 更新文本显示
        self.text_item.setText(f"选中区域: {min_val:.2f} - {max_val:.2f}")
        self.text_item.setPos(min_val, 0.8)

        # 在实际应用中，可以在这里处理选定区域的数据
        print(f"选定的X轴范围: {min_val:.2f} - {max_val:.2f}")


def test_choose_region():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    assert app.exec() == 0