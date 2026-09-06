"""Wavelet denoising feature plugin package."""

from plugins.remove_wavelet_denoise.plugin import WaveletDenoisePlugin
from plugins.remove_wavelet_denoise.editor import RemoveWaveletDenoise

PLUGIN_CLASS = WaveletDenoisePlugin
EDITOR_CLASS = RemoveWaveletDenoise

__all__ = ["WaveletDenoisePlugin", "RemoveWaveletDenoise", "PLUGIN_CLASS", "EDITOR_CLASS"]
