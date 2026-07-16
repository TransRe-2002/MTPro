from __future__ import annotations

from typing import Literal

import math

from core.em_data import EMData


EARTH_RADIUS_M = 6_371_000.0


def great_circle_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    *,
    unit: Literal["m", "km"] = "km",
) -> float:
    """Return the great-circle distance between two WGS84 lat/lon points."""
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    d_phi = math.radians(float(lat2) - float(lat1))
    d_lambda = math.radians(float(lon2) - float(lon1))

    hav = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    distance_m = 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(hav)))

    if unit == "m":
        return distance_m
    if unit == "km":
        return distance_m / 1000.0
    raise ValueError(f"unsupported unit: {unit}")


def station_distance(
    station_a: EMData,
    station_b: EMData,
    *,
    unit: Literal["m", "km"] = "km",
) -> float:
    """Return the distance between two EMData stations using latitude/longitude."""
    if station_a.latitude is None or station_a.longitude is None:
        raise ValueError(f"station_a has no valid latitude/longitude: {station_a.name}")
    if station_b.latitude is None or station_b.longitude is None:
        raise ValueError(f"station_b has no valid latitude/longitude: {station_b.name}")

    return great_circle_distance(
        station_a.latitude,
        station_a.longitude,
        station_b.latitude,
        station_b.longitude,
        unit=unit,
    )
