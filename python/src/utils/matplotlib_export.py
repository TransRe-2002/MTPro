from __future__ import annotations

# Moved to ui/dialogs/figure_export.py — kept here as a re-export shim for
# backward compatibility with existing importers.
from ui.dialogs.figure_export import (  # noqa: F401
    PIXEL_PRESETS,
    MatplotlibExportDialog,
    default_export_name,
    sanitize_filename,
    export_matplotlib_figure_with_dialog,
    save_matplotlib_figure,
)