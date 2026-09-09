"""Typed in-memory impedance calculation results."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

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
    def exported(self) -> bool:
        return bool(self.exported_formats & {"edi", "z", "csv"})
