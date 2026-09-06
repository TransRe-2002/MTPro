"""Compatibility exports for helpers formerly exposed as plugins.legacy."""
from __future__ import annotations

from plugins.common.base_processor import build_pkg_from_series
from plugins.common.display import decimate_for_display

__all__ = ["build_pkg_from_series", "decimate_for_display"]
