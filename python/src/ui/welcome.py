import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QSizePolicy, QProgressDialog
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class Welcome(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 窗口基本设置
        self.setWindowTitle("欢迎界面")
        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)  # 控件间距
        main_layout.setContentsMargins(50, 50, 50, 50)  # 边距(左,上,右,下)

        # 创建欢迎标签
        self.welcome_label = QLabel("欢迎！")
        self.welcome_label.setFont(QFont("Arial", 24, QFont.Bold))  # 字体设置
        self.welcome_label.setAlignment(Qt.AlignCenter)  # 文本居中[7](@ref)

        # 创建文件打开按钮
        self.open_button = QPushButton("打开文件")
        self.open_button.setFont(QFont("Arial", 12))
        self.open_button.setMinimumWidth(120)  # 设置最小宽度
        self.open_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.continue_button = QPushButton("继续任务")
        self.continue_button.setFont(QFont("Arial", 12))
        self.continue_button.setMinimumWidth(120)
        self.continue_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # 添加控件到布局
        main_layout.addStretch(1)
        main_layout.addWidget(self.welcome_label, alignment=Qt.AlignHCenter)
        main_layout.addWidget(self.open_button, alignment=Qt.AlignHCenter)
        main_layout.addWidget(self.continue_button, alignment=Qt.AlignHCenter)
        main_layout.addStretch(1)

        # 设置窗口布局
        self.setLayout(main_layout)


import pytest
@pytest.mark.parametrize("title", ["欢迎！"])
def test_welcome(title):
    app = QApplication(sys.argv)
    window = Welcome()
    window.show()
    assert app.exec() == 0