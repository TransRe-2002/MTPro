"""Backward-compatibility shim — Qt time helpers moved to :mod:`ui.widgets.qt_time`.

Both ``pts_to_qdt`` and ``qdt_to_pts`` return/take ``QDateTime``, so they are
Qt-coupled and belong in ``ui/`` not ``utils/``. This shim re-exports them for
backward compatibility.
"""

from ui.widgets.qt_time import (  # noqa: F401
    pts_to_qdt,
    qdt_to_pts,
)
