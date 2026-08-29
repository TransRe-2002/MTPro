"""Probe: compiled import of matplotlib (the classic Nuitka pain point)."""
import sys

print("M1 start", flush=True)
import matplotlib  # noqa: F401

print("M2 matplotlib", matplotlib.__version__, flush=True)
import matplotlib.pyplot as plt  # noqa: F401

print("M3 pyplot ok", flush=True)
import matplotlib.backends.backend_qtagg  # noqa: F401

print("M4 qtagg ok", flush=True)
print("MPL_SMOKE_OK", flush=True)
sys.exit(0)