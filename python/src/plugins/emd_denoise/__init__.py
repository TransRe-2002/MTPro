"""Empirical mode decomposition denoising plugin."""

from plugins.emd_denoise.plugin import EMDDenoisePlugin


PLUGIN_CLASS = EMDDenoisePlugin

__all__ = ["EMDDenoisePlugin", "PLUGIN_CLASS"]
