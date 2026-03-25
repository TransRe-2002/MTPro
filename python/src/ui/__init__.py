import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'
import pyqtgraph as pg

pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')