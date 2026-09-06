"""Hankel-SVD denoising plugin."""

from plugins.svd_denoise.plugin import SVDDenoisePlugin


PLUGIN_CLASS = SVDDenoisePlugin

__all__ = ["SVDDenoisePlugin", "PLUGIN_CLASS"]
