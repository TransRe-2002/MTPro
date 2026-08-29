"""Probe: import ui.mainwindow + construct MainWindow (no show/exec)."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

print("U1 start", flush=True)
from PySide6.QtWidgets import QApplication

print("U2 QtWidgets", flush=True)
import ui.mainwindow  # noqa: F401

print("U3 import ui.mainwindow ok", flush=True)
app = QApplication(sys.argv)
print("U4 app ok", flush=True)
from base.data_manager import DataManager

dm = DataManager()
print("U5 DataManager ok", flush=True)
win = mw = None
from ui.mainwindow import MainWindow

win = MainWindow(dm)
print("U6 MainWindow constructed", flush=True)
print("UI_SMOKE_OK", flush=True)
sys.exit(0)