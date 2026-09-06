"""Compatibility exports for the shared Pipeline-slice editor bridge."""
from plugins.common.legacy_editor import (
    LegacyPluginEditor,
    SlicePluginEditor,
    base_processor_editor,
    slice_processor_editor,
    slice_standalone_editor,
    standalone_processor_editor,
)

__all__ = [
    "LegacyPluginEditor",
    "SlicePluginEditor",
    "base_processor_editor",
    "slice_processor_editor",
    "slice_standalone_editor",
    "standalone_processor_editor",
]
