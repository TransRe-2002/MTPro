"""First-party MTPro processing plugins, loaded before external plugins."""
from __future__ import annotations

from plugins.auto_spike_filter.plugin import AutoSpikeFilterPlugin
from plugins.detrend.plugin import DetrendPlugin
from plugins.emd_denoise.plugin import EMDDenoisePlugin
from plugins.iir_filter.plugin import IIRFilterPlugin
from plugins.median_filter.plugin import MedianFilterPlugin
from plugins.morphological_filter.plugin import MorphologicalFilterPlugin
from plugins.remove_diurnal_sinusoid.plugin import DiurnalSinusoidPlugin
from plugins.remove_secular_variation.plugin import SecularVariationPlugin
from plugins.remove_region.plugin import RegionRemovalPlugin
from plugins.remove_spike.plugin import SpikeRemovalPlugin
from plugins.remove_step_baseline.plugin import StepBaselinePlugin
from plugins.remove_step_diff.plugin import StepDiffPlugin
from plugins.remove_step_window.plugin import StepWindowPlugin
from plugins.remove_wavelet_denoise.plugin import WaveletDenoisePlugin
from plugins.svd_denoise.plugin import SVDDenoisePlugin
from plugins.vmd_denoise.plugin import VMDDenoisePlugin


BUILTIN_PLUGINS: list[type] = [
    RegionRemovalPlugin,
    SpikeRemovalPlugin,
    AutoSpikeFilterPlugin,
    EMDDenoisePlugin,
    DetrendPlugin,
    MedianFilterPlugin,
    MorphologicalFilterPlugin,
    IIRFilterPlugin,
    StepDiffPlugin,
    StepWindowPlugin,
    StepBaselinePlugin,
    WaveletDenoisePlugin,
    SVDDenoisePlugin,
    SecularVariationPlugin,
    DiurnalSinusoidPlugin,
    VMDDenoisePlugin,
]
