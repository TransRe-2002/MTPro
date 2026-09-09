"""全局共享线程池（Qt-free，可在无界面环境使用）。

两个池、一个门面：

- ``compute_pool``: CPU 密集任务（区域同步、 decimate 编排等）。
  大小 ``max(1, cpu // 2)`` —— 与区域同步原有策略一致，满载时
  给 UI 留出余量。
- ``io_pool``: 文件读取等 I/O 任务。上限 32、线程惰性创建——
  I/O 等待不吃 CPU，上限放宽不影响计算侧。

铁律（违反会死锁或碰坏 Qt）：

1. **池内任务不得阻塞等待本池的 future**（饥饿死锁）；等待
   **另一个**池的 future 可以（如在 compute 任务里等 io 结果）。
2. **取消是协作式**：把 flag 传进任务，任务自己检查。
3. **池线程里禁止直接操作 Qt 控件**——经 ``utils.qt_bridge``
   的信号排队回主线程。
"""

from __future__ import annotations

import atexit
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)

_CPU = os.cpu_count() or 4

compute_pool = ThreadPoolExecutor(
    max_workers=max(1, _CPU // 2),
    thread_name_prefix="mtpro-compute",
)
io_pool = ThreadPoolExecutor(
    max_workers=32,
    thread_name_prefix="mtpro-io",
)


def pool_for(kind: str) -> ThreadPoolExecutor:
    """按名字取池："compute" | "io"。"""
    if kind == "compute":
        return compute_pool
    if kind == "io":
        return io_pool
    raise ValueError(f"未知池类型: {kind!r}（应为 'compute' 或 'io'）")


def submit(kind: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
    """提交任务到指定池，返回 concurrent.futures.Future。"""
    return pool_for(kind).submit(fn, *args, **kwargs)


def shutdown(wait: bool = False) -> None:
    """关闭两个池。应用退出时调用，默认不等待未完成任务。"""
    compute_pool.shutdown(wait=wait, cancel_futures=True)
    io_pool.shutdown(wait=wait, cancel_futures=True)


atexit.register(shutdown)
