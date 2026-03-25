from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import weakref
import pandas as pd
from typing import List

@dataclass
class Channel:
    name: str
    cts: pd.Series
    npts: int
    start: weakref.ref[pd.Timestamp]
    end: weakref.ref[pd.Timestamp]
    dt: weakref.ref[pd.Timedelta]
    datetime_index: weakref.ref[pd.DatetimeIndex]
    parent: weakref.ref['EMData']

    def __init__(self,
        name: str,
        cts: pd.Series,
        parent: 'EMData'
    ):
        self.name = name
        self.cts = cts
        self.parent = weakref.ref(parent)

        self.npts = parent.npts
        self.start = weakref.ref(parent.start_time)
        self.end = weakref.ref(parent.end_time)
        self.dt = weakref.ref(parent.dt)
        self.datetime_index = weakref.ref(parent.datetime_index)

class EMData(ABC):
    name: str
    path: str
    npts: int
    NCh: int
    chid: List[str]
    data: Dict[str, Channel]
    zero_time: Optional[pd.Timestamp]
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    dt: pd.Timedelta
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation: Optional[float] = None
    e_units: Optional[str] = None
    m_units: Optional[str] = None

    kp_data: Optional[pd.DataFrame]
    datetime_index: pd.DatetimeIndex

    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Channel] = {}
        self.chid: List[str] = []