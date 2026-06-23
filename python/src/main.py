import os
import sys

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QObject

from ui.mainwindow import MainWindow
from ui.log_console import install_standard_stream_logging
from base.data_manager import DataManager

class Main(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_manager = DataManager(self)
        self.window = MainWindow(self.data_manager)

    def main(self):
        self.window.showMaximized()
        app.exec()

if __name__ == "__main__":
    app = QApplication([])
    install_standard_stream_logging()
    main = Main(app)
    main.main()
    sys.exit()
