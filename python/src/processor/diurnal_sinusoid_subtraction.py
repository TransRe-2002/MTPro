"""Backward-compatibility shim — algorithms moved to :mod:`core.algorithms.diurnal`.

The pure numerical logic now lives in ``core/algorithms/diurnal.py``; this
module re-exports every public name so existing
``from processor.diurnal_sinusoid_subtraction import ...`` call sites continue
to work unchanged.
"""

from core.algorithms.diurnal import (  # noqa: F401
    SECONDS_PER_DAY,
    PeriodicSubtractionConfig,
    PeriodicSubtractionResult,
    subtract_diurnal_signal_channel,
    subtract_diurnal_signal_series,
    subtract_periodic_signal_channel,
    subtract_periodic_signal_series,
)
