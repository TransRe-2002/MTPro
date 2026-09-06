"""Automatic short-pulse spike filtering feature plugin."""

from plugins.auto_spike_filter.editor import AutoSpikeFilter
from plugins.auto_spike_filter.plugin import AutoSpikeFilterPlugin

PLUGIN_CLASS = AutoSpikeFilterPlugin
EDITOR_CLASS = AutoSpikeFilter

__all__ = ["AutoSpikeFilter", "AutoSpikeFilterPlugin", "EDITOR_CLASS", "PLUGIN_CLASS"]
