"""Probe: compiled imports of numpy/pandas/xarray/pyqtgraph (no matplotlib)."""
import sys

print("L1 start", flush=True)
import numpy as np

print("L2 numpy", np.__version__, flush=True)
import pandas as pd

print("L3 pandas", pd.__version__, flush=True)
import xarray as xr

print("L4 xarray", xr.__version__, flush=True)
import pyqtgraph as pg

print("L5 pyqtgraph", pg.__version__, flush=True)
print("LIBS_SMOKE_OK", flush=True)
sys.exit(0)