import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'

import pyqtgraph as pg
from PySide6 import QtWidgets

app = QtWidgets.QApplication([])

# 创建 PlotWidget
plot_widget = pg.PlotWidget()

# 获取 ViewBox
vb = plot_widget.getViewBox()

# 禁用 x 轴的鼠标交互（缩放和平移）
vb.setMouseEnabled(x=False, y=True)  # x 轴禁止，y 轴允许

# 添加示例曲线
curve = plot_widget.plot([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])

plot_widget.show()
app.exec()