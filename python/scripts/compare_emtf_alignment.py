"""Reproduce native/reference comparisons on legacy tsStruct MAT recordings.

Run with the project venv and PYTHONPATH=python/src. Outputs are diagnostics,
not replacements for the source data. --remote adds an exactly aligned station.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.io import loadmat

from base.emtf_native import estimate_native_emtf
from emtf.time_alignment import matching_remote_slice
from output_tools.shared.emtf import build_spectrum_kwargs, preprocess_channel


def recording(path):
    s = loadmat(path, simplify_cells=True)["tsStruct"]
    data = np.asarray(s["data"], dtype=float)
    names = [str(name) for name in s["chid"]]
    channels = {name: data[i] for i, name in enumerate(names)}
    start = pd.Timestamp(*[int(value) for value in s["startTime"]], tz="UTC")
    index = pd.date_range(start, periods=data.shape[1], freq=pd.Timedelta(seconds=float(s["dt"])))
    return s, channels, index


def compare(local_path, remote_path=None):
    import emtf

    meta, data, index = recording(local_path)
    names = ["Ex1", "Ey1", "Bx", "By", "Bz"]
    channels = [preprocess_channel(data[name], name) for name in names]
    arrays = channels[:4]
    if remote_path:
        _, remote, remote_index = recording(remote_path)
        start, end = matching_remote_slice(index, remote_index, 0, len(index))
        arrays += [preprocess_channel(remote[name][start:end], name) for name in ("Bx", "By")]
    kwargs = build_spectrum_kwargs(
        sample_rate=1.0 / float(meta["dt"]), window_size=2048, step_pct=50,
        huber_threshold=1.5, max_iterations=25, tolerance=1e-8, hz_arr=channels[4],
        backend="emtf_rs", fft_window_size=128, n_bands=40, station_name=str(meta["name"]),
    )
    timings = {}
    start = time.perf_counter()
    native = estimate_native_emtf(*arrays, remote_reference=bool(remote_path), **kwargs)
    timings["native_seconds"] = time.perf_counter() - start
    estimator = emtf.estimate_remote_reference_spectrum if remote_path else emtf.estimate_single_station_spectrum
    start = time.perf_counter()
    reference = estimator(*arrays, **kwargs)
    timings["fortran_seconds"] = time.perf_counter() - start
    if native.periods.shape != reference.periods.shape:
        raise AssertionError(f"Different band counts: {native.periods.size} vs {reference.periods.size}")
    metrics = {}
    for name in ("zxy", "zyx", "rho_xy", "rho_yx", "phi_xy", "phi_yx", "zxy_se", "zyx_se"):
        a, b = getattr(native, name), getattr(reference, name)
        errors = np.abs(a - b)
        if not name.startswith("phi"):
            errors /= np.maximum(np.abs(b), 1e-15)
        metrics[name] = {"median": float(np.median(errors)), "max": float(np.max(errors)),
                         "p90": float(np.quantile(errors, .9)), "per_band": errors.tolist()}
    summary = dict(local=str(Path(local_path).resolve()), remote=str(Path(remote_path).resolve()) if remote_path else None,
                   samples=len(index), band_count=len(native.periods), timings=timings,
                   native_periods=native.periods.tolist(), reference_periods=reference.periods.tolist(),
                   max_relative_period_error=float(np.max(np.abs(native.periods/reference.periods-1))),
                   differences=metrics,
                   notes=["Relative complex-Z / rho / SE errors; phase errors in degrees.",
                          "Native preserves float64 input; Fortran rounds H to .01 nT and E to .001 mV/km in ASC.",
                          "SEs and robust observation-count definitions are not yet proven equivalent.",
                          "Single-run wall times include reference file I/O, not a controlled performance benchmark."])
    return summary, native, reference


def plot(path, native, reference, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, layout="constrained")
    for col, component in enumerate(("xy", "yx")):
        for result, color, label in ((reference, "#39b914", "EMTF (Fortran)"),
                                      (native, "#941dc5", "emtf-rs aligned")):
            axes[0, col].loglog(result.periods, getattr(result, "rho_"+component),
                                "o", ms=4, color=color, label=label)
            axes[1, col].semilogx(result.periods, getattr(result, "phi_"+component),
                                  "o", ms=4, color=color)
        axes[0, col].set_title(component.upper())
        axes[1, col].set_xlabel("Period (s)")
    axes[0, 0].set_ylabel("Apparent resistivity (ohm m)")
    axes[1, 0].set_ylabel("Phase (degrees)")
    for ax in axes.flat:
        ax.grid(True, alpha=.2)
    axes[0, 1].legend(loc="best", frameon=False)
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("local", type=Path)
    parser.add_argument("--remote", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, native, reference = compare(args.local, args.remote)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(args.output / "comparison.png", native, reference, args.local.stem + (" | RR" if args.remote else " | SS"))
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("differences", "native_periods", "reference_periods")}, indent=2))
    print("Z relative error medians:", {k: summary["differences"][k]["median"] for k in ("zxy", "zyx")})


if __name__ == "__main__":
    main()
