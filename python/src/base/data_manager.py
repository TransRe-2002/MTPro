from typing import Dict, Iterator, Optional, Tuple

from PySide6.QtCore import Signal, QObject

from core.em_data import EMData

class DataManager(QObject):
    data_added = Signal(int)
    data_removed = Signal(int)
    active_changed = Signal(int)
    # 数据集内容被就地修改（预处理）。第二个参数为 structural：
    #   True  — npts/dt/通道集变化，波形视图需完整重建
    #   False — 仅元数据变化，刷新树与摘要即可
    data_changed = Signal(int, bool)

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

    def notify_changed(self, key: int, structural: bool = False) -> None:
        """通知数据集已被就地修改（预处理后由 UI 层调用）。

        key 不存在时静默忽略。
        """
        if key not in self._em_datas:
            return
        self.data_changed.emit(key, structural)




