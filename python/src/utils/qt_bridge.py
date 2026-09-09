"""把共享线程池的任务结果排队回 Qt 主线程。

池线程里禁止直接碰 Qt 控件——本模块是唯一的 sanctioned 回路：
``FutureBridge`` 在池线程上 emit 信号，Qt 的 AutoConnection 自动
把它排队到接收槽所在的线程（主线程）。

典型用法::

    from utils.concurrency import submit
    from utils.qt_bridge import FutureBridge

    bridge = FutureBridge(owner=self)          # parent 防止提前析构
    bridge.done.connect(self._on_loaded)       # 主线程槽: (result,)
    bridge.failed.connect(self._on_error)      # 主线程槽: (exception,)
    bridge.attach(submit("io", load_mat, path))

或一步到位::

    run("io", load_mat, path, on_done=self._on_loaded,
        on_error=self._on_error, owner=self)
"""

from __future__ import annotations

import logging
from concurrent.futures import CancelledError, Future
from typing import Any
from collections.abc import Callable

from PySide6.QtCore import QObject, Qt, Signal

from utils.concurrency import submit as _pool_submit

logger = logging.getLogger(__name__)


class FutureBridge(QObject):
    """把一个 future 的结果/异常经信号转发到接收槽所在的线程。"""

    done = Signal(object)    # 任务返回值
    failed = Signal(object)  # 异常对象（含 CancelledError）

    def attach(self, future: Future) -> "FutureBridge":
        future.add_done_callback(self._relay)
        return self

    def _relay(self, future: Future) -> None:
        # 在池线程上执行；跨线程 emit 由 Qt 自动排队到槽的线程
        try:
            exc = future.exception()
        except CancelledError as cancelled:
            exc = cancelled
        if exc is not None:
            self.failed.emit(exc)
        else:
            self.done.emit(future.result())


def run(
    kind: str,
    fn: Callable[..., Any],
    *args: Any,
    on_done: Callable[[Any], None] | None = None,
    on_error: Callable[[BaseException], None] | None = None,
    owner: QObject | None = None,
    **kwargs: Any,
) -> Future:
    """提交任务并自动把结果桥接回主线程。

    Args:
        kind: "compute" | "io"。
        fn / args / kwargs: 任务本体（在池线程执行，不得碰 Qt 控件）。
        on_done: 主线程成功槽 (result,)。
        on_error: 主线程失败槽 (exception,)。未提供时异常只记日志。
        owner: 桥对象的 parent，防止任务完成前被 GC。

    Returns:
        任务的 Future（可用于取消等控制）。
    """
    bridge = FutureBridge(owner)
    queued = Qt.ConnectionType.QueuedConnection
    if on_done is not None:
        bridge.done.connect(on_done, type=queued)
    if on_error is not None:
        bridge.failed.connect(on_error, type=queued)
    else:
        bridge.failed.connect(
            lambda exc: logger.error("后台任务失败: %r", exc),
            type=queued,
        )
    return bridge.attach(_pool_submit(kind, fn, *args, **kwargs))
