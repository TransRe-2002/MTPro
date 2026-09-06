"""Zero-phase IIR filtering plugin."""

from plugins.iir_filter.plugin import IIRFilterPlugin


PLUGIN_CLASS = IIRFilterPlugin

__all__ = ["IIRFilterPlugin", "PLUGIN_CLASS"]
