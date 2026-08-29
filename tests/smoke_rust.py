"""Probe whether rust_lib loads + runs inside a Nuitka-compiled Windows exe."""
import sys

print("R1 start", flush=True)
import numpy as np

print("R2 numpy", flush=True)
import rust_lib  # noqa: F401

print("R3 rust_lib imported", flush=True)
x = np.linspace(0.0, 1.0, 1000)
y = np.sin(x)
res = rust_lib.minmax_decimate_rs(x, y, 32)
print("R4 call ok, type:", type(res), flush=True)
print("RUST_SMOKE_OK", flush=True)
sys.exit(0)