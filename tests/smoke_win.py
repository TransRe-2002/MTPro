"""Minimal PySide6 smoke target for cross-compiling to Windows (validation only)."""
import sys

print("START", flush=True)
from PySide6.QtWidgets import QApplication, QLabel

print("QtWidgets imported", flush=True)
app = QApplication(sys.argv)
print("QApplication created", flush=True)
w = QLabel("hello")
print("SMOKE_OK", flush=True)
sys.exit(0)