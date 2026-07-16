"""Standalone test for robust impedance estimation using emtf_rs.

Reads a MAT-format time series file, runs single-station robust estimation,
and prints results to stdout.

Usage:
    ~/anaconda3/bin/python python/scripts/test_robust_estimate.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend

# Make sure src/ is on the path so we can import atm_rpc modules
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from io_utils.mat_io import MatLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_robust")


def interp_nan(arr: np.ndarray) -> np.ndarray:
    """Fill NaN values with linear interpolation; extrapolate edges with nearest."""
    s = pd.Series(arr)
    if s.isna().any():
        s = s.interpolate(method="linear", limit_direction="both")
        # If any NaN remain at edges after interpolation, ffill/bfill
        s = s.ffill().bfill()
    return s.to_numpy()


def main():
    file_path = "/home/transen5/Project/atm_rpc/039BE-20240501-20240515-dt5_struct.mat"

    logger.info("Loading %s ...", file_path)
    em_data = MatLoader.load(file_path)

    logger.info("Station: %s", em_data.name)
    logger.info("Channels: %s", em_data.channel_names())
    logger.info("Samples: %d", em_data.npts)
    dt_sec = em_data.dt.total_seconds()
    sample_rate = 1.0 / dt_sec
    logger.info("dt: %s (%.4f Hz)", em_data.dt, sample_rate)

    # Channel mapping — this station has Bx/By/Bz/Ex1/Ey1/Ex2/Ey2
    ch_ex = "Ex1"
    ch_ey = "Ey1"
    ch_hx = "Bx"
    ch_hy = "By"
    ch_hz = "Bz"

    for label, ch in [("Ex", ch_ex), ("Ey", ch_ey), ("Hx", ch_hx), ("Hy", ch_hy), ("Hz", ch_hz)]:
        if not em_data.has_channel(ch):
            logger.error("Channel %s (%s) not found", label, ch)
            sys.exit(1)

    ex = em_data.get_channel_series(ch_ex).to_numpy()
    ey = em_data.get_channel_series(ch_ey).to_numpy()
    hx = em_data.get_channel_series(ch_hx).to_numpy()
    hy = em_data.get_channel_series(ch_hy).to_numpy()
    hz = em_data.get_channel_series(ch_hz).to_numpy()

    # Fill NaN gaps and detrend
    for name, arr in [("Ex", ex), ("Ey", ey), ("Hx", hx), ("Hy", hy), ("Hz", hz)]:
        n_nan = np.isnan(arr).sum()
        if n_nan > 0:
            logger.info("%s: filling %d NaN (%.2f%%)", name, n_nan, 100 * n_nan / len(arr))

    ex = interp_nan(ex)
    ey = interp_nan(ey)
    hx = interp_nan(hx)
    hy = interp_nan(hy)
    hz = interp_nan(hz)

    # Detrend to remove large DC offsets
    ex = detrend(ex)
    ey = detrend(ey)
    hx = detrend(hx)
    hy = detrend(hy)
    hz = detrend(hz)

    for name, arr in [("Ex", ex), ("Ey", ey), ("Hx", hx), ("Hy", hy), ("Hz", hz)]:
        logger.info("%s after detrend: min=%.4f, max=%.4f, std=%.4f",
                    name, arr.min(), arr.max(), arr.std())

    # ── estimation parameters ──
    window_size = 1024             # 1024 * 5s = 5120s per window
    step_size = window_size // 2    # 50% overlap
    huber_threshold = 1.5
    max_iterations = 25
    tolerance = 1e-8

    logger.info(
        "Running robust estimation: window=%d, step=%d, huber=%.1f, max_iter=%d, tol=%.0e",
        window_size, step_size, huber_threshold, max_iterations, tolerance,
    )

    import emtf_rs

    result = emtf_rs.estimate_single_station(
        ex, ey, hx, hy,
        sample_rate_hz=sample_rate,
        window_size=window_size,
        hz=hz,
        step_size=step_size,
        huber_threshold=huber_threshold,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    # ── print results ──
    freqs = result.frequencies_hz
    n_freq = len(freqs)
    n_seg = result.estimates[0].segment_count

    print(f"\n{'='*100}")
    print(f"  Robust Single-Station Estimation Results")
    print(f"  Station: {em_data.name}  |  {n_freq} frequencies  |  {n_seg} time windows")
    print(f"{'='*100}")
    header = (f"{'freq (Hz)':>12s}  {'period (s)':>12s}  "
              f"{'Zxx real':>12s} {'Zxx imag':>12s}  "
              f"{'Zxy real':>12s} {'Zxy imag':>12s}  "
              f"{'Zyx real':>12s} {'Zyx imag':>12s}  "
              f"{'Zyy real':>12s} {'Zyy imag':>12s}  "
              f"{'res_Ex':>8s}  {'res_Ey':>8s}")
    print(header)
    print("-" * 100)

    # Show first 20 and last 5 frequency bins
    show_indices = list(range(min(20, n_freq)))
    if n_freq > 25:
        show_indices += list(range(n_freq - 5, n_freq))

    for i in show_indices:
        f = freqs[i]
        period = 1.0 / f if f > 0 else float("inf")
        print(
            f"{f:12.6f}  {period:12.3f}  "
            f"{result.zxx[i].real:12.4e} {result.zxx[i].imag:12.4e}  "
            f"{result.zxy[i].real:12.4e} {result.zxy[i].imag:12.4e}  "
            f"{result.zyx[i].real:12.4e} {result.zyx[i].imag:12.4e}  "
            f"{result.zyy[i].real:12.4e} {result.zyy[i].imag:12.4e}  "
            f"{result.residual_scale_ex[i]:8.4f}  {result.residual_scale_ey[i]:8.4f}"
        )

    if n_freq > 25:
        print(f"{'...':>12s}")

    # ── summary statistics ──
    zxy = result.zxy
    zyx = result.zyx
    finite = np.isfinite(zxy)
    n_finite = finite.sum()

    print(f"\n{'='*100}")
    print("  Summary")
    print(f"{'='*100}")
    print(f"  Valid frequency bins:  {n_finite} / {n_freq}")
    print(f"  Mean residual scale Ex: {np.mean(result.residual_scale_ex[finite]):.4f}")
    print(f"  Mean residual scale Ey: {np.mean(result.residual_scale_ey[finite]):.4f}")
    print(f"  Mean |Zxy|:             {np.mean(np.abs(zxy[finite])):.4e}")
    print(f"  Mean |Zyx|:             {np.mean(np.abs(zyx[finite])):.4e}")
    print(f"  Total time windows:     {n_seg}")
    if hasattr(result, 'magnetic_condition_number'):
        cond = result.magnetic_condition_number
        cond_finite = cond[np.isfinite(cond)]
        if len(cond_finite) > 0:
            print(f"  Mean H cond number:     {np.mean(cond_finite):.2f}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
