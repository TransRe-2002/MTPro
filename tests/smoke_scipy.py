"""Probe: compiled import of scipy.signal."""
import sys

print("S1 start", flush=True)
import scipy  # noqa: F401

print("S2 scipy", scipy.__version__, flush=True)
import scipy.signal  # noqa: F401

print("S3 scipy.signal ok", flush=True)
from scipy.signal import detrend  # noqa: F401

print("S4 detrend ok", flush=True)
from scipy import fft, linalg, ndimage, optimize  # noqa: F401

print("S5 submodules ok", flush=True)
print("SCIPY_SMOKE_OK", flush=True)
sys.exit(0)