from __future__ import annotations

import argparse
import gc
import importlib.util
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "python" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import core.mat_data as mat_data_module
from core.algorithms.svd import svd_denoise_series
from io_utils.bin_io import DAT_CH_FIELDS, DAT_DTYPE, BinLoader
from io_utils.mat_io import MatLoader, new_ts_struct
from processor._display_utils import apply_nan_intervals
from processor.diurnal_sinusoid_subtraction import subtract_diurnal_signal_series
from processor.step_algorithms import (
    remove_diff_steps_by_count,
    robust_baseline_destep,
    wavelet_multiscale_denoise,
    windowed_mean_destep,
)
from utils.decimate import build_lod_pyramid, minmax_decimate


@dataclass(frozen=True)
class Benchmark:
    name: str
    func: Callable[[], object]


def _rust_lib_available() -> bool:
    return importlib.util.find_spec("rust_lib") is not None


def _make_signal(npts: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(npts, dtype=np.float64)
    y = (
        2.0 * np.sin(x / 700.0)
        + 0.4 * np.cos(x / 37.0)
        + 0.00001 * x
    )
    y[npts // 3:] += 3.0
    y[(2 * npts) // 3:] -= 2.0
    y[npts // 5:npts // 5 + 20] += np.linspace(0.0, 8.0, 20, dtype=np.float64)
    return x, y.astype(np.float64, copy=False)


def _make_diurnal_series(days: int, freq: str) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=days * 24 * 6, freq=freq, tz="UTC")
    dt_seconds = (index[1] - index[0]).total_seconds()
    t = np.arange(len(index), dtype=np.float64) * dt_seconds
    baseline = 0.0008 * t + 0.5 * np.sin(2.0 * np.pi * t / (36.0 * 3600.0))
    noise = (
        5.0 * np.sin(2.0 * np.pi * t / (24.0 * 3600.0) + 0.4)
        + 2.0 * np.cos(2.0 * np.pi * t / (12.0 * 3600.0) - 0.2)
    )
    return pd.Series(baseline + noise, index=index, name="Ex")


def _make_dataset(index: pd.DatetimeIndex, npts: int, channel_count: int) -> xr.Dataset:
    x = np.arange(npts, dtype=np.float64)
    data_vars = {
        f"Ch{i + 1}": (("time",), np.sin(x / (50.0 + i)) + i)
        for i in range(channel_count)
    }
    return xr.Dataset(data_vars, coords={"time": index})


def _write_synthetic_dat(path: Path, npts: int) -> None:
    raw = np.zeros(npts, dtype=DAT_DTYPE)
    start = pd.Timestamp("2024-01-01 00:00:00")
    times = start + pd.to_timedelta(np.arange(npts, dtype=np.int64), unit="s")
    raw["year"] = times.year.astype(np.uint16)
    raw["month"] = times.month.astype(np.uint16)
    raw["day"] = times.day.astype(np.uint16)
    raw["hour"] = times.hour.astype(np.uint16)
    raw["minute"] = times.minute.astype(np.uint16)
    raw["second"] = times.second.astype(np.uint16)
    raw["millisecond"] = 0
    x = np.arange(npts, dtype=np.float32)
    for i, field in enumerate(DAT_CH_FIELDS):
        raw[field] = np.sin(x / np.float32(100.0 + i)) + np.float32(i)
    raw.tofile(path)


def _write_synthetic_mat(path: Path, npts: int, channel_count: int) -> None:
    ts = new_ts_struct()
    data = np.vstack(
        [
            np.sin(np.arange(npts, dtype=np.float64) / (80.0 + i)) + i
            for i in range(channel_count)
        ]
    )
    ts["NCh"] = np.array([[channel_count]], dtype=np.float64)
    ts["npts"] = np.array([[npts]], dtype=np.float64)
    ts["data"] = data
    ts["startTime"] = np.array([[2024, 1, 1, 0, 0, 0]], dtype=np.float64)
    ts["zeroTime"] = np.array([[2024, 1, 1, 0, 0, 0]], dtype=np.float64)
    ts["endTime"] = np.array([[2024, 1, 1, 0, 0, npts - 1]], dtype=np.float64)
    ts["dt"] = np.array([[1.0]], dtype=np.float64)
    ts["timeStamp"] = np.linspace(2024.0, 2024.0 + npts / (366.0 * 86400.0), npts).reshape(1, -1)
    ts["name"] = "bench"

    chid = np.empty((1, channel_count), dtype=object)
    for i in range(channel_count):
        chid[0, i] = f"Ch{i + 1}"
    ts["chid"] = chid
    savemat(path, {"tsStruct": ts})


def _time_one(func: Callable[[], object], repeat: int) -> tuple[float, float, object]:
    durations: list[float] = []
    result: object = None
    for _ in range(repeat):
        gc.collect()
        start = perf_counter()
        result = func()
        durations.append(perf_counter() - start)
    return min(durations), statistics.mean(durations), result


def _format_result(result: object) -> str:
    if isinstance(result, tuple):
        parts = []
        for item in result:
            if hasattr(item, "shape"):
                parts.append(str(item.shape))
            elif isinstance(item, list):
                parts.append(f"list[{len(item)}]")
            else:
                parts.append(type(item).__name__)
        return ", ".join(parts)
    if isinstance(result, list):
        return f"list[{len(result)}]"
    if hasattr(result, "shape"):
        return str(result.shape)
    return type(result).__name__


def build_benchmarks(args: argparse.Namespace) -> list[Benchmark]:
    x, y = _make_signal(args.npts)
    index = pd.date_range("2024-01-01", periods=args.npts, freq="1s", tz="UTC")
    dataset = _make_dataset(index, args.npts, args.channels)
    intervals = [
        (args.npts // 10, args.npts // 10 + args.npts // 200),
        (args.npts // 3, args.npts // 3 + args.npts // 100),
        (args.npts // 2, args.npts // 2 + args.npts // 80),
    ]
    diurnal = _make_diurnal_series(args.diurnal_days, args.diurnal_freq)
    signal_series = pd.Series(y, index=index, name="Ex")
    tempdir = tempfile.TemporaryDirectory(prefix="mtpro-profile-")
    tmp_path = Path(tempdir.name)
    dat_path = tmp_path / "synthetic.dat"
    mat_path = tmp_path / "synthetic.mat"
    _write_synthetic_dat(dat_path, min(args.npts, args.io_npts))
    _write_synthetic_mat(mat_path, min(args.npts, args.io_npts), args.channels)

    left = args.npts // 2
    right = min(args.npts - 1, left + 120)

    def bench_nan_intervals() -> np.ndarray:
        arr = y.copy()
        apply_nan_intervals(arr, intervals)
        return arr

    def bench_series_exchange() -> dict[str, np.ndarray]:
        return {
            name: dataset[name].to_pandas().to_numpy(dtype=np.float64, copy=True)
            for name in dataset.data_vars
        }

    def bench_robust_baseline() -> np.ndarray:
        result = robust_baseline_destep(
            y,
            (left, right),
            baseline_degree=1,
            transition_degree=1,
            context_points=args.context_points,
            loss="soft_l1",
        )
        return result.corrected

    def bench_bin_load(_tempdir: tempfile.TemporaryDirectory[str] = tempdir) -> int:
        data = BinLoader.load(
            str(dat_path),
            "Asia/Shanghai",
            channel_configs=[
                {"name": f"Ch{i + 1}", "scale": 1.0, "electrode_pair": ""}
                for i in range(8)
            ],
        )
        return data.npts

    def bench_mat_v5_load(_tempdir: tempfile.TemporaryDirectory[str] = tempdir) -> int:
        old_fetch_kp = mat_data_module.fetch_kp
        mat_data_module.fetch_kp = lambda *_args, **_kwargs: None
        try:
            data = MatLoader.load(str(mat_path), tz="Asia/Shanghai")
            return data.npts
        finally:
            mat_data_module.fetch_kp = old_fetch_kp

    def bench_wavelet() -> np.ndarray:
        result = wavelet_multiscale_denoise(
            y,
            (left, right),
            wavelet_name="db4",
            level=4,
            sample_rate_hz=1.0,
            padding_points=args.wavelet_padding,
            mode="soft",
        )
        return result.corrected

    benchmarks = [
        Benchmark("decimate.minmax", lambda: minmax_decimate(x, y, args.target_bins)),
        Benchmark("decimate.lod_pyramid", lambda: build_lod_pyramid(x, y)),
        Benchmark("display.apply_nan_intervals", bench_nan_intervals),
        Benchmark("data.xarray_series_exchange", bench_series_exchange),
        Benchmark("io.bin_loader_full", bench_bin_load),
        Benchmark("io.mat_v5_loader_full", bench_mat_v5_load),
        Benchmark("step.remove_diff_by_count", lambda: remove_diff_steps_by_count(y, args.step_count)[0]),
        Benchmark("step.windowed_mean_destep", lambda: windowed_mean_destep(y, 1.0, 1000, args.step_count)[0]),
        Benchmark("step.robust_baseline_destep", bench_robust_baseline),
        Benchmark(
            "denoise.hankel_svd",
            lambda: svd_denoise_series(
                signal_series,
                embedding_points=101,
                rank=8,
                min_segment_points=128,
            ).cleaned,
        ),
        Benchmark("diurnal.subtract_periodic", lambda: subtract_diurnal_signal_series(diurnal).cleaned),
    ]
    if not args.skip_wavelet:
        benchmarks.append(Benchmark("wavelet.multiscale_denoise", bench_wavelet))
    return benchmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run synthetic performance probes for MTPro Python/Rust migration candidates.",
    )
    parser.add_argument("--npts", type=int, default=300_000, help="Synthetic time-series length.")
    parser.add_argument("--channels", type=int, default=5, help="Synthetic channel count for data exchange.")
    parser.add_argument("--io-npts", type=int, default=200_000, help="Synthetic file length for IO probes.")
    parser.add_argument("--repeat", type=int, default=3, help="Repeat count; reports best and mean time.")
    parser.add_argument("--target-bins", type=int, default=2000, help="Target bins for min/max decimation.")
    parser.add_argument("--step-count", type=int, default=20, help="Step count for step-removal probes.")
    parser.add_argument("--context-points", type=int, default=256, help="Context points for baseline destep.")
    parser.add_argument("--wavelet-padding", type=int, default=512, help="Padding points for wavelet denoise.")
    parser.add_argument("--skip-wavelet", action="store_true", help="Skip PyWavelets probe.")
    parser.add_argument("--diurnal-days", type=int, default=14, help="Synthetic diurnal series length in days.")
    parser.add_argument("--diurnal-freq", default="10min", help="Synthetic diurnal sample frequency.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.npts < 10_000:
        raise SystemExit("--npts must be at least 10000")
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    print(f"python: {sys.executable}")
    print(f"npts: {args.npts:,}")
    print(f"rust_lib: {'available' if _rust_lib_available() else 'not installed'}")
    print()
    print(f"{'benchmark':34} {'best ms':>10} {'mean ms':>10}  result")
    print("-" * 76)

    for bench in build_benchmarks(args):
        best, mean, result = _time_one(bench.func, args.repeat)
        print(f"{bench.name:34} {best * 1000.0:10.2f} {mean * 1000.0:10.2f}  {_format_result(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
