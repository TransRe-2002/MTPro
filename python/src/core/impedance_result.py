"""Typed in-memory impedance calculation results."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

import numpy as np

from emtf.io.edi import BandedResult, from_result


def _time_label(value: object) -> str:
    if value is None:
        return ""
    try:
        return value.strftime("%Y%m%d_%H%M%S")
    except AttributeError:
        return str(value).replace("-", "").replace(":", "").replace(" ", "_")[:15]


def default_impedance_result_name(payload: dict, method: str) -> str:
    local = str(payload.get("station_name") or "STATION")
    remote = str(payload.get("remote_station_name") or "")
    station = f"{local}←{remote}" if remote else local
    method_label = "RR" if method == "remote reference" else "Robust"
    start = _time_label(payload.get("range_start"))
    end = _time_label(payload.get("range_end"))
    return f"{station} {method_label}" + (f" {start}–{end}" if start and end else "")


def _detached_bands(payload: dict, method: str) -> BandedResult | None:
    bands = payload.get("bands")
    if bands is None:
        return None
    mode = "rr" if method == "remote reference" else "ss"
    detached = from_result(
        bands,
        station_name=str(payload.get("station_name") or "unknown"),
        mode=mode,
        lat=float(payload.get("latitude") or 0.0),
        lon=float(payload.get("longitude") or 0.0),
        elev=float(payload.get("elevation") or 0.0),
    )
    return deepcopy(detached)


def propagate_complex_standard_error(
    values: object,
    component_error: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate equal independent real/imaginary errors to amplitude and phase.

    The returned phase uncertainty is expressed in degrees.  Phase uncertainty
    is undefined for a zero-amplitude value and is reported as ``nan``.
    """
    complex_values = np.asarray(values, dtype=np.complex128)
    errors = np.abs(np.asarray(component_error, dtype=float))
    complex_values, errors = np.broadcast_arrays(complex_values, errors)
    amplitudes = np.abs(complex_values)
    amplitude_error = np.full(amplitudes.shape, np.nan, dtype=float)
    phase_error = np.full(amplitudes.shape, np.nan, dtype=float)

    valid = (
        np.isfinite(complex_values.real)
        & np.isfinite(complex_values.imag)
        & np.isfinite(errors)
        & (amplitudes > 0)
    )
    if valid.any():
        real = complex_values.real[valid]
        imaginary = complex_values.imag[valid]
        amplitude = amplitudes[valid]
        error = errors[valid]
        amplitude_error[valid] = np.sqrt(
            np.square(real * error / amplitude)
            + np.square(imaginary * error / amplitude)
        )
        phase_error[valid] = np.degrees(np.sqrt(
            np.square(imaginary * error / np.square(amplitude))
            + np.square(real * error / np.square(amplitude))
        ))
    return amplitude_error, phase_error


@dataclass
class ImpedanceResult:
    """Independent saved calculation result; raw time series are excluded."""

    snapshot_id: str
    name: str
    method: str
    bands: BandedResult | None
    spectrum: object | None
    created_at: datetime
    range_start: str = ""
    range_end: str = ""
    station_name: str = ""
    remote_station_name: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    elevation: float = 0.0
    parameters: dict[str, object] = field(default_factory=dict)
    channels: dict[str, object] = field(default_factory=dict)
    exported_formats: set[str] = field(default_factory=set)

    @classmethod
    def from_payload(cls, payload: dict) -> ImpedanceResult:
        method = str(payload.get("method") or "robust estimate")
        return cls(
            snapshot_id=uuid4().hex,
            name=str(payload.get("name") or default_impedance_result_name(payload, method)),
            method=method,
            bands=_detached_bands(payload, method),
            spectrum=deepcopy(payload.get("spectrum")),
            created_at=datetime.now().astimezone(),
            range_start=str(payload.get("range_start") or ""),
            range_end=str(payload.get("range_end") or ""),
            station_name=str(payload.get("station_name") or ""),
            remote_station_name=str(payload.get("remote_station_name") or ""),
            latitude=float(payload.get("latitude") or 0.0),
            longitude=float(payload.get("longitude") or 0.0),
            elevation=float(payload.get("elevation") or 0.0),
            parameters=deepcopy(payload.get("parameters") or {}),
            channels=deepcopy(payload.get("channels") or {}),
        )

    @property
    def result(self) -> BandedResult | None:
        return self.bands

    @property
    def has_tipper(self) -> bool:
        bands = self.bands
        if bands is None:
            return False
        values = [
            np.asarray(getattr(bands, name, []))
            for name in ("periods", "tzx", "tzy", "tzx_se", "tzy_se")
            if getattr(bands, name, None) is not None
        ]
        if len(values) != 5:
            return False
        count = min(map(len, values))
        if count == 0:
            return False
        periods, tzx, tzy, tzx_se, tzy_se = (value[:count] for value in values)
        return bool((
            np.isfinite(periods)
            & (periods > 0)
            & np.isfinite(tzx.real)
            & np.isfinite(tzx.imag)
            & np.isfinite(tzy.real)
            & np.isfinite(tzy.imag)
            & np.isfinite(tzx_se)
            & np.isfinite(tzy_se)
        ).any())

    @property
    def exported(self) -> bool:
        return bool(self.exported_formats & {"edi", "z", "csv"})
