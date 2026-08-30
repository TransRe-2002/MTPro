"""DataManager.data_changed 信号与 notify_changed 的单元测试。"""

from base.data_manager import DataManager
from core.em_data import EMData


class StubData(EMData):
    def __init__(self):
        super().__init__("/tmp/stub.mat")
        self.name = "stub"
        self.kp_data = None

    def restore_data(self, ch: str):
        return None


def test_notify_changed_emits_key_and_structural():
    dm = DataManager()
    key = dm.add(StubData())

    received = []
    dm.data_changed.connect(lambda k, s: received.append((k, s)))

    dm.notify_changed(key, structural=True)
    dm.notify_changed(key, structural=False)

    assert received == [(key, True), (key, False)]


def test_notify_changed_default_non_structural():
    dm = DataManager()
    key = dm.add(StubData())

    received = []
    dm.data_changed.connect(lambda k, s: received.append((k, s)))
    dm.notify_changed(key)

    assert received == [(key, False)]


def test_notify_changed_invalid_key_is_silent():
    dm = DataManager()

    received = []
    dm.data_changed.connect(lambda k, s: received.append((k, s)))
    dm.notify_changed(999, structural=True)

    assert received == []


def test_notify_changed_after_remove_is_silent():
    dm = DataManager()
    key = dm.add(StubData())
    dm.remove(key)

    received = []
    dm.data_changed.connect(lambda k, s: received.append((k, s)))
    dm.notify_changed(key, structural=True)

    assert received == []
