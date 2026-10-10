from collections.abc import Iterator

import pandas as pd
from PySide6.QtCore import Signal, QObject

from core.em_data import EMData


class DataManager(QObject):
    data_added = Signal(int)
    data_removed = Signal(int)
    active_changed = Signal(int)
    # 数据集内容被就地修改。第二个参数为 structural：
    #   True  — npts/dt/通道集变化，波形视图需完整重建
    #   False — 通道数值或元数据变化，无需完整重建波形视图
    data_changed = Signal(int, bool)
    # None 表示整个数据集；字符串表示单个通道。先清理处理状态，再广播数据更新。
    data_restored = Signal(int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._em_datas: dict[int, EMData] = {}
        # One in-memory baseline per channel.  It is captured immediately
        # before the first numerical processing write and deliberately is not
        # serialized: closing the dataset/app starts a new undo cycle.
        self._channel_processing_baselines: dict[tuple[int, str], pd.Series] = {}
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
        self.discard_channel_processing_baselines(key)
        self.data_removed.emit(key)
        return True

    def get(self, key: int) -> EMData | None:
        if key == 0:
            return None
        return self._em_datas.get(key)

    def items(self) -> Iterator[tuple[int, EMData]]:
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

    def active_data(self) -> EMData | None:
        return self.get(self._active_id)

    def notify_changed(self, key: int, structural: bool = False) -> None:
        """通知数据集已被就地修改（预处理后由 UI 层调用）。

        key 不存在时静默忽略。
        """
        if key not in self._em_datas:
            return
        self.data_changed.emit(key, structural)

    def notify_restored(self, key: int, channel: str | None = None) -> None:
        if key not in self._em_datas:
            return
        self.data_restored.emit(key, channel)
        self.notify_changed(key, structural=channel is None)

    # ── Channel processing undo ────────────────────────────────

    def capture_channel_processing_baseline(self, key: int, channel: str) -> bool:
        """Capture the channel once, immediately before its first write.

        Returns ``True`` only when a new baseline was stored.  Later processing
        operations intentionally keep the original snapshot so "undo all"
        restores the state from before the whole processing cycle.
        """
        token = (key, channel)
        if token in self._channel_processing_baselines:
            return False
        data = self.get(key)
        if data is None or not data.has_channel(channel):
            return False
        self._channel_processing_baselines[token] = data.get_channel_series(channel).copy(deep=True)
        return True

    def has_channel_processing_baseline(self, key: int, channel: str) -> bool:
        return (key, channel) in self._channel_processing_baselines

    def has_dataset_processing_baseline(self, key: int) -> bool:
        return any(data_id == key for data_id, _channel in self._channel_processing_baselines)

    def processing_baseline_channels(self, key: int) -> list[str]:
        return [channel for data_id, channel in self._channel_processing_baselines if data_id == key]

    def undo_all_channel_processing(self, key: int, channel: str) -> bool:
        """Restore and consume the first-processing snapshot for one channel."""
        token = (key, channel)
        snapshot = self._channel_processing_baselines.get(token)
        data = self.get(key)
        if snapshot is None or data is None or not data.has_channel(channel):
            return False
        data.set_channel_series(channel, snapshot.copy(deep=True))
        self._channel_processing_baselines.pop(token, None)
        self.notify_restored(key, channel)
        return True

    def undo_all_dataset_processing(self, key: int) -> bool:
        """Restore and consume every first-processing snapshot for one dataset."""
        data = self.get(key)
        if data is None:
            return False
        snapshots = {
            channel: snapshot.copy(deep=True)
            for (data_id, channel), snapshot in self._channel_processing_baselines.items()
            if data_id == key and data.has_channel(channel)
        }
        if not snapshots:
            return False
        current = {
            channel: data.get_channel_series(channel).copy(deep=True)
            for channel in snapshots
        }
        try:
            for channel, snapshot in snapshots.items():
                data.set_channel_series(channel, snapshot)
        except Exception:
            for channel, snapshot in current.items():
                data.set_channel_series(channel, snapshot)
            raise
        for channel in snapshots:
            self._channel_processing_baselines.pop((key, channel), None)
        self.notify_restored(key, None)
        return True

    def discard_channel_processing_baseline(self, key: int, channel: str) -> None:
        self._channel_processing_baselines.pop((key, channel), None)

    def discard_channel_processing_baselines(self, key: int) -> None:
        for token in [token for token in self._channel_processing_baselines if token[0] == key]:
            self._channel_processing_baselines.pop(token, None)

    def rename_channel_processing_baseline(self, key: int, old: str, new: str) -> None:
        snapshot = self._channel_processing_baselines.pop((key, old), None)
        if snapshot is not None:
            snapshot = snapshot.copy(deep=True)
            snapshot.name = new
            self._channel_processing_baselines[(key, new)] = snapshot

    def remap_channel_processing_baselines(
        self,
        key: int,
        renames: dict[str, str],
        deleted: set[str],
    ) -> None:
        """Apply a bulk channel rename/delete atomically to undo snapshots."""
        snapshots = {
            channel: snapshot
            for (data_id, channel), snapshot in self._channel_processing_baselines.items()
            if data_id == key
        }
        self.discard_channel_processing_baselines(key)
        for channel, snapshot in snapshots.items():
            if channel in deleted:
                continue
            target = renames.get(channel, channel)
            copied = snapshot.copy(deep=True)
            copied.name = target
            self._channel_processing_baselines[(key, target)] = copied
