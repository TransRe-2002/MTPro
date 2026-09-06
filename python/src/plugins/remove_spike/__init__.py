"""Spike removal feature plugin package."""

from plugins.remove_spike.plugin import SpikeRemovalPlugin
from plugins.remove_spike.editor import RemoveSpike

PLUGIN_CLASS = SpikeRemovalPlugin
EDITOR_CLASS = RemoveSpike

__all__ = ["SpikeRemovalPlugin", "RemoveSpike", "PLUGIN_CLASS", "EDITOR_CLASS"]
