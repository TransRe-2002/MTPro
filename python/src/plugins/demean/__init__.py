"""Whole-channel mean-removal plugin."""

from plugins.demean.plugin import DemeanPlugin


PLUGIN_CLASS = DemeanPlugin

__all__ = ["DemeanPlugin", "PLUGIN_CLASS"]
