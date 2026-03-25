from core.em_data import EMData
from typing import Dict, Optional
from PySide6.QtCore import Signal, QObject

class DataManager(QObject):
    updated_added = Signal(int)
    updated_removed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.em_datas: Dict[int, EMData] = {}
        self.next_key = 1
        self.updated_removed.connect(self.remove)

    def add(self, data: EMData):
        key = self.next_key
        self.em_datas[key] = data
        self.next_key += 1
        self.updated_added.emit(key)

    def remove(self, key: int):
        if key in self.em_datas:
            del self.em_datas[key]

    def get(self, key: int) -> Optional[EMData]:
        if key == 0:
            return None
        return self.em_datas[key]





