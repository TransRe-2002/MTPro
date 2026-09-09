"""测试会话级 workaround。

pyqtgraph 0.14.0 的 ``LabelItem.sizeHint``（LabelItem.py:132）无保护地访问
``self._sizeHint``，而该属性只在 ``updateMin`` 末尾赋值。离屏测试的 GC
时序下（LegendItem 构建 LabelItem 时被回收触发 sizeHint）会抛
AttributeError 并导致 Qt abort。测试侧给 ``sizeHint`` 加保护。
"""

import pyqtgraph.graphicsItems.LabelItem as _LabelMod
import pytest
from PySide6.QtCore import QSizeF

_orig_size_hint = _LabelMod.LabelItem.sizeHint


def _safe_size_hint(self, hint, constraint):
    if not hasattr(self, "_sizeHint"):
        return QSizeF(0, 0)
    return _orig_size_hint(self, hint, constraint)


_LabelMod.LabelItem.sizeHint = _safe_size_hint


@pytest.fixture(autouse=True)
def _shutdown_shared_worker_pools(request):
    """确保 pytest-forked 子进程退出前不再残留共享工作线程。"""
    yield
    if not request.config.getoption("forked", default=False):
        return

    from utils.concurrency import shutdown

    shutdown(wait=True)
