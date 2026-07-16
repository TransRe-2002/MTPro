from __future__ import annotations

from typing import Optional, Callable

from PySide6.QtWidgets import QLineEdit

# Moved to ui/widgets/validators.py — kept here as a re-export shim for
# backward compatibility with existing importers.
from ui.widgets.validators import (  # noqa: F401
    read_positive_float,
    read_nonnegative_int,
    read_positive_int,
)