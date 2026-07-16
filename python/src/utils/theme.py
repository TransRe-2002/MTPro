from __future__ import annotations

# Moved to ui/theme.py — kept here as a re-export shim for backward
# compatibility with existing importers.
from ui.theme import (  # noqa: F401
    is_dark_theme,
    palette_color,
    pg_background,
    pg_foreground,
    configure_pyqtgraph_theme,
    transparent_legend_brush,
    subtle_pen,
    configure_matplotlib_figure,
    configure_matplotlib_canvas,
)