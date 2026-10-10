"""C-response derivation for EMTF-convention impedance results."""

from __future__ import annotations

import numpy as np


IMPEDANCE_COMPONENTS = ("zxx", "zxy", "zyx", "zyy")
C_COMPONENTS = ("cxx", "cxy", "cyx", "cyy")


def c_response_components(periods, *impedances) -> dict[str, np.ndarray]:
    """Convert EMTF impedances (mV/km/nT) to C responses in metres.

    EMTF's numerical impedance convention differs from SI ohms by ``mu0*1000``.
    Applying ``C = Z_SI / (i*omega*mu0)`` therefore reduces to
    ``C = 1000*Z_EMTF / (i*omega)``.
    """
    periods = np.asarray(periods, dtype=np.float64)
    if len(impedances) != 4:
        raise ValueError("four impedance tensor components are required")

    omega = np.full(periods.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(periods) & (periods > 0)
    omega[valid] = 2.0 * np.pi / periods[valid]
    scale = np.full(periods.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    scale[valid] = 1000.0 / (1j * omega[valid])

    return {
        name: np.asarray(impedance, dtype=np.complex128) * scale
        for name, impedance in zip(C_COMPONENTS, impedances)
    }


def c_response_from_bands(bands) -> dict[str, np.ndarray]:
    """Return Cxx/Cxy/Cyx/Cyy arrays from a banded EMTF result."""
    periods = np.asarray(getattr(bands, "periods", []), dtype=np.float64)
    impedances = [getattr(bands, name, None) for name in IMPEDANCE_COMPONENTS]
    if any(value is None for value in impedances):
        return {}
    return c_response_components(periods, *impedances)
