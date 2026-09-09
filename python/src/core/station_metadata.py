from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StationMetadata:
    """Station location fields shared by impedance-output tools."""

    name: str
    latitude: float | None
    longitude: float | None
    elevation: float | None
