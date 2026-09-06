"""Variational mode decomposition denoising plugin."""

from plugins.vmd_denoise.plugin import VMDDenoisePlugin


PLUGIN_CLASS = VMDDenoisePlugin

__all__ = ["VMDDenoisePlugin", "PLUGIN_CLASS"]
