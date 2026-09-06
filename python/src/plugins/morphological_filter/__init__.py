"""Morphological denoising plugin."""

from plugins.morphological_filter.plugin import MorphologicalFilterPlugin


PLUGIN_CLASS = MorphologicalFilterPlugin

__all__ = ["MorphologicalFilterPlugin", "PLUGIN_CLASS"]
