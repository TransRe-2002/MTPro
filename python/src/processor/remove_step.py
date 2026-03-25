from PyQt6.QtWidgets import QGridLayout
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter,
    QSizePolicy, QSlider, QDateTimeEdit, QScrollBar,
    QToolBar, QLabel, QPushButton, QMessageBox
)
from PySide6.QtCore import Signal
import numpy as np


class RemoveStep(QWidget):
    result_signal = Signal(str, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout = QGridLayout(self)

