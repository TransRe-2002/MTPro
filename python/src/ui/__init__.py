import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'

from ui.theme import configure_pyqtgraph_theme

configure_pyqtgraph_theme()