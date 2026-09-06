"""Diurnal sinusoid removal feature plugin package."""

from plugins.remove_diurnal_sinusoid.plugin import DiurnalSinusoidPlugin
from plugins.remove_diurnal_sinusoid.editor import RemoveDiurnalSinusoid

PLUGIN_CLASS = DiurnalSinusoidPlugin
EDITOR_CLASS = RemoveDiurnalSinusoid

__all__ = ["DiurnalSinusoidPlugin", "RemoveDiurnalSinusoid", "PLUGIN_CLASS", "EDITOR_CLASS"]
