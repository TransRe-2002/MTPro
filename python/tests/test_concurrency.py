import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEventLoop, QObject, QTimer
from PySide6.QtWidgets import QApplication

from utils.concurrency import compute_pool, io_pool, pool_for, submit
from utils.qt_bridge import FutureBridge, run


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _Collector(QObject):
    """主线程接收器：槽是 QObject 方法，信号必走排队连接回主线程。"""

    def __init__(self) -> None:
        super().__init__()
        self.value = None
        self.exc = None
        self.loop: QEventLoop | None = None

    def wait(self, timeout_ms: int = 5000) -> None:
        self.loop = QEventLoop()
        QTimer.singleShot(timeout_ms, self.loop.quit)  # 防挂死守卫
        self.loop.exec()

    def on_done(self, value) -> None:
        self.value = value
        if self.loop is not None:
            self.loop.quit()

    def on_failed(self, exc) -> None:
        self.exc = exc
        if self.loop is not None:
            self.loop.quit()


def test_submit_compute_and_io_return_results():
    fc = submit("compute", sum, range(1000))
    fi = submit("io", lambda: "io-ok")
    assert fc.result(timeout=5) == sum(range(1000))
    assert fi.result(timeout=5) == "io-ok"


def test_pool_sizes_follow_policy():
    expected_compute = max(1, (os.cpu_count() or 4) // 2)
    assert compute_pool._max_workers == expected_compute
    assert io_pool._max_workers == 32


def test_pool_for_rejects_unknown_kind():
    with pytest.raises(ValueError):
        pool_for("gpu")


def test_future_bridge_delivers_result_to_main_thread():
    _app()
    collector = _Collector()
    bridge = FutureBridge()
    bridge.done.connect(collector.on_done)
    bridge.attach(submit("io", lambda: 42))
    collector.wait()
    assert collector.value == 42


def test_future_bridge_delivers_exception_to_main_thread():
    _app()
    collector = _Collector()
    bridge = FutureBridge()
    bridge.failed.connect(collector.on_failed)

    def boom():
        raise RuntimeError("kapow")

    bridge.attach(submit("compute", boom))
    collector.wait()
    assert isinstance(collector.exc, RuntimeError)


def test_run_convenience_wires_done_slot():
    _app()
    collector = _Collector()
    run("io", lambda: "via-run", on_done=collector.on_done)
    collector.wait()
    assert collector.value == "via-run"


def test_run_convenience_wires_error_slot():
    _app()
    collector = _Collector()

    def boom():
        raise ValueError("nope")

    run("compute", boom, on_error=collector.on_failed)
    collector.wait()
    assert isinstance(collector.exc, ValueError)
