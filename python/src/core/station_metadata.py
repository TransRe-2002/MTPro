from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StationMetadata:
    """Station location fields shared by impedance-output tools."""

    name: str
    latitude: Optional[float]
    longitude: Optional[float]
    elevation: Optional[float]
