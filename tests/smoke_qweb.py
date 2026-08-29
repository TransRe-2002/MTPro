"""Probe: does QtWebEngine import crash in a Nuitka-compiled Windows exe?"""
import sys

print("Q1 start", flush=True)
from PySide6.QtWidgets import QApplication

print("Q2 QtWidgets", flush=True)
from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: F401

print("Q3 QtWebEngineWidgets imported", flush=True)
app = QApplication(sys.argv)
print("Q4 app ok", flush=True)
print("QWEB_SMOKE_OK", flush=True)
sys.exit(0)