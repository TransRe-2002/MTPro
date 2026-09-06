"""Median denoising plugin."""

from plugins.median_filter.plugin import MedianFilterPlugin


PLUGIN_CLASS = MedianFilterPlugin

__all__ = ["MedianFilterPlugin", "PLUGIN_CLASS"]
