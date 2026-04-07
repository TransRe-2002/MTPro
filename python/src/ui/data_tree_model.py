from __future__ import annotations

from typing import Dict, List, Optional, Set

from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt

from base.data_manager import DataManager
from core.em_data import Channel, EMData


class DataTreeModel(QAbstractItemModel):
    KEY_ROLE = Qt.ItemDataRole.UserRole
    CHANNEL_ROLE = Qt.ItemDataRole.UserRole + 1
    NODE_KIND_ROLE = Qt.ItemDataRole.UserRole + 2
    _ROOT_ID = 0

    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self._keys: List[int] = [key for key, _ in self.data_manager.items()]
        self._row_by_key: Dict[int, int] = {
            key: row for row, key in enumerate(self._keys)
        }
        self._checked_channels: Dict[int, Set[str]] = {}

        self.data_manager.data_added.connect(self.on_data_added)
        self.data_manager.data_removed.connect(self.on_data_removed)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 1

    def rowCount(self, parent=QModelIndex()) -> int:
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            return len(self._keys)

        if self.is_dataset_index(parent):
            key = self.dataset_key(parent)
            em_data = self.data_manager.get(key) if key is not None else None
            if em_data is None:
                return 0
            return len(self._channel_names(em_data))

        return 0

    def index(self, row: int, column: int, parent=QModelIndex()) -> QModelIndex:
        if column != 0 or row < 0:
            return QModelIndex()

        if not parent.isValid():
            if row >= len(self._keys):
                return QModelIndex()
            return self.createIndex(row, column, self._ROOT_ID)

        if not self.is_dataset_index(parent):
            return QModelIndex()

        key = self.dataset_key(parent)
        if key is None:
            return QModelIndex()

        em_data = self.data_manager.get(key)
        if em_data is None:
            return QModelIndex()

        channel_names = self._channel_names(em_data)
        if row >= len(channel_names):
            return QModelIndex()

        return self.createIndex(row, column, key)

    def parent(self, index: QModelIndex) -> QModelIndex:
        if not self.is_channel_index(index):
            return QModelIndex()

        key = self.dataset_key(index)
        if key is None:
            return QModelIndex()

        row = self._row_by_key.get(key)
        if row is None:
            return QModelIndex()

        return self.createIndex(row, 0, self._ROOT_ID)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        if self.is_dataset_index(index):
            key = self.dataset_key(index)
            em_data = self.data_manager.get(key) if key is not None else None
            if em_data is None:
                return None

            if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
                return em_data.name
            if role == self.KEY_ROLE:
                return key
            if role == self.NODE_KIND_ROLE:
                return "dataset"
            return None

        if self.is_channel_index(index):
            key = self.dataset_key(index)
            channel_name = self.channel_name(index)
            if key is None or channel_name is None:
                return None

            if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
                return channel_name
            if role == Qt.ItemDataRole.CheckStateRole:
                checked = channel_name in self._checked_channels.get(key, set())
                return Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            if role == self.KEY_ROLE:
                return key
            if role == self.CHANNEL_ROLE:
                return channel_name
            if role == self.NODE_KIND_ROLE:
                return "channel"

        return None

    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole) -> bool:
        if (
            not self.is_channel_index(index)
            or role != Qt.ItemDataRole.CheckStateRole
        ):
            return False

        key = self.dataset_key(index)
        channel_name = self.channel_name(index)
        if key is None or channel_name is None:
            return False

        checked = Qt.CheckState(value) == Qt.CheckState.Checked
        bucket = self._checked_channels.setdefault(key, set())

        if checked:
            if channel_name in bucket:
                return False
            bucket.add(channel_name)
        else:
            if channel_name not in bucket:
                return False
            bucket.remove(channel_name)
            if not bucket:
                self._checked_channels.pop(key, None)

        self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
        return True

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags

        base_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if self.is_channel_index(index):
            return base_flags | Qt.ItemFlag.ItemIsUserCheckable
        return base_flags

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role=Qt.ItemDataRole.DisplayRole,
    ):
        if (
            orientation == Qt.Orientation.Horizontal
            and section == 0
            and role == Qt.ItemDataRole.DisplayRole
        ):
            return "数据名称"
        return None

    def checked_channels(self) -> List[Channel]:
        result: List[Channel] = []
        for key in self._keys:
            em_data = self.data_manager.get(key)
            if em_data is None:
                continue

            checked_names = self._checked_channels.get(key)
            if not checked_names:
                continue

            for channel_name in self._channel_names(em_data):
                if channel_name in checked_names:
                    result.append(em_data.data[channel_name])

        return result

    def is_dataset_index(self, index: QModelIndex) -> bool:
        return index.isValid() and index.internalId() == self._ROOT_ID

    def is_channel_index(self, index: QModelIndex) -> bool:
        return index.isValid() and index.internalId() != self._ROOT_ID

    def dataset_key(self, index: QModelIndex) -> Optional[int]:
        if not index.isValid():
            return None

        if self.is_dataset_index(index):
            return self._key_at_row(index.row())

        if self.is_channel_index(index):
            return int(index.internalId())

        return None

    def index_for_key(self, key: int) -> QModelIndex:
        row = self._row_by_key.get(key)
        if row is None:
            return QModelIndex()
        return self.index(row, 0)

    def channel_name(self, index: QModelIndex) -> Optional[str]:
        if not self.is_channel_index(index):
            return None

        key = self.dataset_key(index)
        em_data = self.data_manager.get(key) if key is not None else None
        if em_data is None:
            return None

        channel_names = self._channel_names(em_data)
        if 0 <= index.row() < len(channel_names):
            return channel_names[index.row()]
        return None

    def on_data_added(self, key: int):
        if key in self._row_by_key:
            return

        row = len(self._keys)
        self.beginInsertRows(QModelIndex(), row, row)
        self._keys.append(key)
        self._row_by_key[key] = row
        self.endInsertRows()

    def on_data_removed(self, key: int):
        row = self._row_by_key.get(key)
        if row is None:
            return

        self.beginRemoveRows(QModelIndex(), row, row)
        del self._keys[row]
        self.endRemoveRows()

        self._row_by_key.pop(key, None)
        self._checked_channels.pop(key, None)
        self._reindex_rows(start=row)

    def _reindex_rows(self, start: int = 0):
        for row in range(start, len(self._keys)):
            self._row_by_key[self._keys[row]] = row

    def _key_at_row(self, row: int) -> Optional[int]:
        if 0 <= row < len(self._keys):
            return self._keys[row]
        return None

    @staticmethod
    def _channel_names(em_data: EMData) -> List[str]:
        if getattr(em_data, "chid", None):
            return list(em_data.chid)
        return list(em_data.data.keys())
