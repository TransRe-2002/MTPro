from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def block_signals(*widgets):
    """Temporarily block signals on one or more QObjects.

    Usage:
        with block_signals(self.start_time, self.end_time):
            self.start_time.setDateTime(...)
            self.end_time.setDateTime(...)
    """
    for w in widgets:
        w.blockSignals(True)
    try:
        yield
    finally:
        for w in widgets:
            w.blockSignals(False)
