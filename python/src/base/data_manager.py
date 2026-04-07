from typing import Dict, Iterator, Optional, Tuple

from PySide6.QtCore import Signal, QObject

from core.em_data import EMData

class DataManager(QObject):
    data_added = Signal(int)
    data_removed = Signal(int)
    active_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._em_datas: Dict[int, EMData] = {}
        self._next_key = 1
        self._active_id = 0

    def add(self, data: EMData) -> int:
        key = self._next_key
        self._em_datas[key] = data
        self._next_key += 1
        self.data_added.emit(key)
        return key

    def remove(self, key: int) -> bool:
        if key not in self._em_datas:
            return False

        if key == self._active_id:
            self.set_active(0)

        del self._em_datas[key]
        self.data_removed.emit(key)
        return True

    def get(self, key: int) -> Optional[EMData]:
        if key == 0:
            return None
        return self._em_datas.get(key)

    def items(self) -> Iterator[Tuple[int, EMData]]:
        return iter(self._em_datas.items())

    def set_active(self, key: int) -> bool:
        if key != 0 and key not in self._em_datas:
            return False
        if self._active_id == key:
            return False

        self._active_id = key
        self.active_changed.emit(key)
        return True

    def active_id(self) -> int:
        return self._active_id

    def active_data(self) -> Optional[EMData]:
        return self.get(self._active_id)




