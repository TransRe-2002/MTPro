"""Backward-compatibility shim — algorithms moved to :mod:`core.algorithms.step`.

The pure numerical logic now lives in ``core/algorithms/step.py``; this module
re-exports every public name so existing ``from plugins.common.step_algorithms import ...``
call sites continue to work unchanged.
"""

from core.algorithms.step import (  # noqa: F401
    DEFAULT_AVG_WINDOW,
    DEFAULT_MAX_STEPS,
    RobustBaselineStepResult,
    WaveletDenoiseResult,
    WaveletPulseResult,
    WindowedStepCandidate,
    detect_step_indices,
    remove_diff_steps_by_count,
    remove_diff_steps_by_threshold,
    robust_baseline_destep,
    wavelet_local_depulse,
    wavelet_multiscale_denoise,
    windowed_mean_destep,
    windowed_mean_destep_with_diagnostics,
    zero_diff_indices,
)
