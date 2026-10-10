"""Adapt the native DNFF/FC-band path to MTPro's shared result contract.

Numerics run in emtf-rs. The bundled emtf package supplies only the common
configuration and result types here; this adapter never invokes Fortran.
"""
from __future__ import annotations

import numpy as np


def native_level_specs(levels):
    """Keep decset overlap, FC count and centre-first FIR semantics explicit."""
    specs = []
    for level in levels:
        if level.decimation_offset != 0:
            raise ValueError("Native EMTF alignment currently requires zero clock offsets")
        overlap = int(level.window_length)
        if overlap != level.window_length:
            raise ValueError("Native EMTF alignment requires integer overlap samples")
        coefficients = [level.fil_1, level.fil_2, level.fil_3, level.fil_4]
        if not 1 <= level.nfilt <= len(coefficients):
            raise ValueError("Expected 1–4 centre-first EMTF filter coefficients")
        specs.append((level.nwin, overlap, level.decimation_factor,
                      level.nfft, coefficients[:level.nfilt]))
    return specs


def estimate_native_emtf(*arrays, remote_reference=False, **kwargs):
    """Estimate on the same explicit FC bands as DNFF + tranmt.

    Uses original tranmt defaults (Huber 1.5, 50 iterations, tolerance .002)
    and pwset length 3. There is no second impedance-averaging step.
    """
    import emtf_rs
    from emtf.api import SpectrumResult

    if not hasattr(emtf_rs, "estimate_emtf_bands"):
        raise RuntimeError(
            "emtf-rs 缺少 EMTF 对齐接口，请在项目 python/.venv 中重新构建安装 Rust 扩展"
        )
    expected = 6 if remote_reference else 4
    if len(arrays) != expected:
        raise ValueError(f"Expected {expected} input channels")
    channels = list(arrays[:4])
    if kwargs.get("hz") is not None:
        channels.append(kwargs["hz"])
    if remote_reference:
        channels.extend(arrays[4:])
    bands = kwargs["bands"]
    native = emtf_rs.estimate_emtf_bands(
        np.stack(channels).astype(np.float64, copy=False),
        kwargs["sample_rate_hz"], native_level_specs(kwargs["decimation_levels"]),
        [(b.decimation_level, b.fc_start, b.fc_end) for b in bands],
        remote_reference=remote_reference,
    )
    names = ["frequencies_hz"]
    names += [f"z{component}{suffix}" for component in ("xx", "xy", "yx", "yy")
              for suffix in ("", "_se")]
    names += [f"{quantity}_{component}{suffix}" for quantity in ("rho", "phi")
              for component in ("xy", "yx") for suffix in ("", "_se")]
    n_observations = np.asarray(native.n_observations, dtype=np.int32)
    result = SpectrumResult(
        **{name: getattr(native, name) for name in names},
        n_observations=n_observations,
        segment_count=int(n_observations[0]) // (bands[0].fc_end - bands[0].fc_start + 1),
        station_name=kwargs.get("station_name", "STATION"),
        mode="rr" if remote_reference else "ss",
        processing_scheme="emtf_fc_bands_v1",
        **{name: getattr(native, name) for name in ("tzx", "tzy", "tzx_se", "tzy_se")},
    )
    return result
