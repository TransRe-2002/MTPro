"""Region removal feature plugin package."""

from plugins.remove_region.plugin import RegionRemovalPlugin
from plugins.remove_region.editor import RemoveRegion, RegionPlotWidget

PLUGIN_CLASS = RegionRemovalPlugin
EDITOR_CLASS = RemoveRegion

__all__ = ["RegionRemovalPlugin", "RemoveRegion", "RegionPlotWidget", "PLUGIN_CLASS", "EDITOR_CLASS"]
