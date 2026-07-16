"""Backward-compatibility shim — geo logic moved to :mod:`core.geo`.

``utils.geo`` imported ``core.em_data.EMData``, creating a ``utils → core``
edge. The functions now live in ``core/geo.py`` (intra-``core`` edge only);
this module re-exports them so existing importers work unchanged.
"""

from core.geo import (  # noqa: F401
    EARTH_RADIUS_M,
    great_circle_distance,
    station_distance,
)
