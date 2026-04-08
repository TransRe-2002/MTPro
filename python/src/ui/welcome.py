import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QSizePolicy, QProgressDialog, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class Welcome(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("欢迎界面")
        self.setObjectName("WelcomeRoot")
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(50, 50, 50, 50)

        panel = QFrame(self)
        panel.setObjectName("WelcomePanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(18)
        panel_layout.setContentsMargins(42, 38, 42, 38)

        self.welcome_label = QLabel("MTPro")
        self.welcome_label.setFont(QFont("Arial", 25))
        self.welcome_label.setObjectName("WelcomeTitle")
        self.welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.subtitle_label = QLabel(
            "面向大地电磁法时间序列查看、人工清洗与后续处理的桌面工作台。"
        )
        self.subtitle_label.setObjectName("WelcomeSubtitle")
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setFixedHeight(40)
        self.subtitle_label.setWordWrap(True)

        self.open_button = QPushButton("打开 MAT 数据")
        # self.open_button.setFont(QFont("Arial", 12))
        self.open_button.setMinimumWidth(180)
        self.open_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.continue_button = QPushButton("进入工作区")
        # self.continue_button.setFont(QFont("Arial", 12))
        self.continue_button.setMinimumWidth(180)
        self.continue_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        panel_layout.addWidget(self.welcome_label, alignment=Qt.AlignHCenter)
        panel_layout.addWidget(self.subtitle_label, alignment=Qt.AlignHCenter)
        panel_layout.addSpacing(8)
        panel_layout.addWidget(self.open_button, alignment=Qt.AlignHCenter)
        panel_layout.addWidget(self.continue_button, alignment=Qt.AlignHCenter)

        main_layout.addStretch(1)
        main_layout.addWidget(panel, alignment=Qt.AlignHCenter)
        main_layout.addStretch(1)

        self.setLayout(main_layout)


import pytest
@pytest.mark.parametrize("title", ["欢迎！"])
def test_welcome(title):
    app = QApplication(sys.argv)
    window = Welcome()
    window.show()
    assert app.exec() == 0
