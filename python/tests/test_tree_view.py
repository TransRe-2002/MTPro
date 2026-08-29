import pytest
from pathlib import Path
import sys
from PySide6.QtWidgets import QApplication

from base.data_manager import DataManager
from ui.data_tree_view import DataTreeViewer
_FIXTURES = [
    Path("/home/transen5/Documents/MTData/039BE-20240501-20240515-dt5_struct.mat"),
    Path("/home/transen5/Documents/MTData/031BE-20240501-20240520-dt5_struct.mat"),
    Path("/home/transen5/Documents/MTData/003E-20240501-20240531-dt5_struct.mat"),
]
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in _FIXTURES),
    reason="需要本机测量数据 fixture（~/Documents/MTData/*，仓库未包含）",
)

from io_utils.mat_io import MatLoader

def test_tree_view():
    app = QApplication(sys.argv)
    manager = DataManager()
    window = DataTreeViewer(manager)
    data1 = MatLoader.load('/home/transen5/Documents/MTData/039BE-20240501-20240515-dt5_struct.mat')
    data2 = MatLoader.load('/home/transen5/Documents/MTData/031BE-20240501-20240520-dt5_struct.mat')
    data3 = MatLoader.load('/home/transen5/Documents/MTData/003E-20240501-20240531-dt5_struct.mat')
    manager.add(data1)
    manager.add(data2)
    manager.add(data3)
    window.show()
    assert app.exec() == 0
