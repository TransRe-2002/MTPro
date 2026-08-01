from __future__ import annotations

import logging
import math
import posixpath
import re
import threading
from collections import Counter
from concurrent.futures import as_completed
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Callable, Iterable
from xml.etree.ElementTree import fromstring, iterparse
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytz

from core.xlsx_data import XlsxEMData
from io_utils.em_io import EMDataLoader
from utils.concurrency import io_pool

logger = logging.getLogger(__name__)


class XlsxImportCancelled(Exception):
    """批量导入被用户取消。向导据此走失败回滚路径。"""


try:
    import rust_lib
    _HAS_RUST_XLSX = hasattr(rust_lib, "read_xlsx_sheet_inline_str_rs")
except ImportError:
    _HAS_RUST_XLSX = False
    rust_lib = None  # type: ignore

XLSX_MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
XLSX_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
OFFICE_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"

DEFAULT_XLSX_CHANNELS = (
    "Bx_1", "By_2", "Bz_3", "Test_4",
    "Ex1_5", "Ey1_6", "Ex2_7", "Ey2_8",
)

INTERP_NONE = "none"
INTERP_LINEAR = "linear"
INTERP_NEAREST = "nearest"
INTERPOLATION_METHODS = {
    INTERP_NONE: "不插值",
    INTERP_LINEAR: "时间加权线性插值",
    INTERP_NEAREST: "最近邻",
}

# 单通道点数建议上限：超过后波形浏览（即便有 LOD 降采样）与
# 处理工具（全序列拷贝、逐窗拟合）的内存/时间开销都会明显上升。
MAX_POINTS_PER_CHANNEL = 5_000_000


def estimate_channel_points(range_seconds: float, dt_sec: float) -> int:
    """估计给定时间跨度与采样周期下的单通道点数（含首尾两端）。"""
    if range_seconds <= 0 or dt_sec <= 0:
        return 0
    return int(range_seconds // dt_sec) + 1


def recommend_min_dt(
    range_seconds: float,
    max_points: int = MAX_POINTS_PER_CHANNEL,
) -> float:
    """推荐 dt 下限：使单通道点数不超过 max_points。

    结果向上取整到 2 位有效数字，便于人工读数与设置
    （例如一年跨度 → 6.307... → 6.4 s）。
    """
    if range_seconds <= 0 or max_points <= 1:
        return 0.0
    raw = range_seconds / (max_points - 1)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    return math.ceil(raw / magnitude * 10.0) * magnitude / 10.0


def format_points(n: int) -> str:
    """点数的中文紧凑显示：1576.8万 / 1.58亿 / 9,999。"""
    if n >= 100_000_000:
        return f"{n / 1e8:.2f}亿"
    if n >= 10_000:
        return f"{n / 1e4:.1f}万"
    return f"{n:,}"

_INLINE_A_CELL_RE = re.compile(
    rb'<c\s+[^>]*r="A\d+"[^>]*>.*?<t[^>]*>(.*?)</t>.*?</c>',
    re.DOTALL,
)
_INLINE_ROW_RE = re.compile(rb'<row\b[^>]*>(.*?)</row>', re.DOTALL)
_INLINE_CELL_RE = re.compile(
    rb'<c\s+[^>]*r="([A-Z]+)\d+"[^>]*>(.*?)</c>',
    re.DOTALL,
)


@dataclass(frozen=True)
class XlsxChannelSummary:
    sheet_name: str
    channel_name: str
    rows: int
    first_time: pd.Timestamp | None
    last_time: pd.Timestamp | None
    inferred_dt_sec: float | None
    top_intervals: tuple[tuple[float, int], ...]


@dataclass(frozen=True)
class XlsxWorkbookSummary:
    path: str
    file_size_mb: float
    sheets: tuple[XlsxChannelSummary, ...]

    @property
    def inferred_dt_sec(self) -> float | None:
        candidates = [
            sheet.inferred_dt_sec
            for sheet in self.sheets
            if sheet.inferred_dt_sec is not None and sheet.inferred_dt_sec > 0
        ]
        if not candidates:
            return None
        counts = Counter(candidates)
        return counts.most_common(1)[0][0]

    @property
    def first_time(self) -> pd.Timestamp | None:
        values = [sheet.first_time for sheet in self.sheets if sheet.first_time is not None]
        return min(values) if values else None

    @property
    def last_time(self) -> pd.Timestamp | None:
        values = [sheet.last_time for sheet in self.sheets if sheet.last_time is not None]
        return max(values) if values else None

    @property
    def max_rows(self) -> int:
        return max((sheet.rows for sheet in self.sheets), default=0)


@dataclass(frozen=True)
class XlsxFileSummary:
    path: str
    prefix: str
    file_size_mb: float
    first_time: pd.Timestamp | None
    last_time: pd.Timestamp | None
    inferred_dt_sec: float | None
    rows: int


@dataclass(frozen=True)
class XlsxGroupSummary:
    prefix: str
    files: tuple[XlsxFileSummary, ...]
    overlaps: tuple[tuple[str, str, float], ...]
    touches: tuple[tuple[str, str], ...]
    gaps: tuple[tuple[str, str, float], ...]

    @property
    def first_time(self) -> pd.Timestamp | None:
        values = [item.first_time for item in self.files if item.first_time is not None]
        return min(values) if values else None

    @property
    def last_time(self) -> pd.Timestamp | None:
        values = [item.last_time for item in self.files if item.last_time is not None]
        return max(values) if values else None

    @property
    def inferred_dt_sec(self) -> float | None:
        counts = self.dt_counts
        if not counts:
            return None
        return counts.most_common(1)[0][0]

    @property
    def dt_counts(self) -> Counter[float]:
        return Counter(
            item.inferred_dt_sec
            for item in self.files
            if item.inferred_dt_sec is not None and item.inferred_dt_sec > 0
        )

    @property
    def total_rows(self) -> int:
        return sum(item.rows for item in self.files)


def default_channel_configs(summary: XlsxWorkbookSummary) -> list[dict]:
    configs: list[dict] = []
    for sheet in summary.sheets:
        configs.append({
            "sheet": sheet.sheet_name,
            "source_name": sheet.channel_name,
            "name": sheet.channel_name,
            "enabled": bool(sheet.channel_name),
            "scale": 1.0,
        })
    return configs


def xlsx_filename_prefix(path: str | Path) -> str:
    stem = Path(path).stem
    if "-" not in stem:
        return stem
    return stem.split("-", 1)[0]


def is_real_xlsx_file(path: str | Path) -> bool:
    name = Path(path).name
    return Path(path).suffix.lower() == ".xlsx" and not name.startswith(".~") and not name.startswith("~$")


def read_xlsx_header(path: str) -> XlsxWorkbookSummary:
    path_obj = Path(path)
    with ZipFile(path_obj) as zf:
        shared_strings = _read_shared_strings(zf)
        sheets: list[XlsxChannelSummary] = []
        for sheet_name, sheet_path in _workbook_sheets(zf):
            header: list[str | None] | None = None
            first_time: pd.Timestamp | None = None
            last_time: pd.Timestamp | None = None
            previous_time: pd.Timestamp | None = None
            interval_counts: Counter[float] = Counter()
            rows = 0

            # 优先用 fast regex 路径（适用于 inlineStr sheet），
            # 在 sharedStrings / 数值型 sheet 上回退到 ElementTree iterparse。
            row_iter = _iter_sheet_rows_fast(zf, sheet_path)
            if row_iter is None:
                row_iter = _iter_sheet_rows(zf, sheet_path, shared_strings)

            for row_index, row in enumerate(row_iter):
                if row_index == 0:
                    header = row
                    continue
                if len(row) < 2:
                    continue
                timestamp = _parse_naive_timestamp(row[0])
                if timestamp is None:
                    continue
                if first_time is None:
                    first_time = timestamp
                last_time = timestamp
                if previous_time is not None:
                    delta = (timestamp - previous_time).total_seconds()
                    if delta > 0:
                        interval_counts[float(delta)] += 1
                previous_time = timestamp
                rows += 1

            channel_name = ""
            if header and len(header) > 1 and header[1]:
                channel_name = str(header[1]).strip()
            if not channel_name:
                channel_name = sheet_name.split("_", 1)[0].strip()

            top = tuple(interval_counts.most_common(8))
            inferred = top[0][0] if top else None
            sheets.append(XlsxChannelSummary(
                sheet_name=sheet_name,
                channel_name=channel_name,
                rows=rows,
                first_time=first_time,
                last_time=last_time,
                inferred_dt_sec=inferred,
                top_intervals=top,
            ))

    return XlsxWorkbookSummary(
        path=str(path_obj),
        file_size_mb=path_obj.stat().st_size / (1024 * 1024),
        sheets=tuple(sheets),
    )


def read_xlsx_structure(path: str) -> XlsxWorkbookSummary:
    """只读取 sheet 名和通道列头，不扫描全量时间序列。"""
    path_obj = Path(path)
    with ZipFile(path_obj) as zf:
        shared_strings = _read_shared_strings(zf)
        sheets: list[XlsxChannelSummary] = []
        for sheet_name, sheet_path in _workbook_sheets(zf):
            first_row = next(_iter_sheet_rows(zf, sheet_path, shared_strings), [])
            channel_name = ""
            if len(first_row) > 1 and first_row[1]:
                channel_name = str(first_row[1]).strip()
            if not channel_name:
                channel_name = sheet_name.split("_", 1)[0].strip()
            sheets.append(XlsxChannelSummary(
                sheet_name=sheet_name,
                channel_name=channel_name,
                rows=0,
                first_time=None,
                last_time=None,
                inferred_dt_sec=None,
                top_intervals=(),
            ))
    return XlsxWorkbookSummary(
        path=str(path_obj),
        file_size_mb=path_obj.stat().st_size / (1024 * 1024),
        sheets=tuple(sheets),
    )


def read_xlsx_file_summary(path: str) -> XlsxFileSummary:
    """读取单个 xlsx 的轻量摘要。

    为了让批量选择可接受，只扫描第一个 sheet 的时间列。当前数据集中
    8 个 sheet 的首尾时间基本一致；最终导入仍会逐 sheet 读取真实数据。
    """
    path_obj = Path(path)
    with ZipFile(path_obj) as zf:
        shared_strings = _read_shared_strings(zf)
        sheets = _workbook_sheets(zf)
        if not sheets:
            raise ValueError(f"xlsx 中没有工作表: {path}")
        _sheet_name, sheet_path = sheets[0]
        fast = _fast_inline_sheet_time_summary(zf, sheet_path)
        if fast is not None:
            first_time, last_time, rows, interval_counts = fast
        else:
            first_time = None
            last_time = None
            previous_time: pd.Timestamp | None = None
            interval_counts: Counter[float] = Counter()
            rows = 0
            for timestamp in _iter_first_sheet_times(zf, sheet_path, shared_strings):
                if timestamp is None:
                    continue
                if first_time is None:
                    first_time = timestamp
                last_time = timestamp
                if previous_time is not None:
                    delta = (timestamp - previous_time).total_seconds()
                    if delta > 0:
                        interval_counts[float(delta)] += 1
                previous_time = timestamp
                rows += 1

    inferred = interval_counts.most_common(1)[0][0] if interval_counts else None
    return XlsxFileSummary(
        path=str(path_obj),
        prefix=xlsx_filename_prefix(path_obj),
        file_size_mb=path_obj.stat().st_size / (1024 * 1024),
        first_time=first_time,
        last_time=last_time,
        inferred_dt_sec=inferred,
        rows=rows,
    )


def _iter_first_sheet_times(
    zf: ZipFile,
    sheet_path: str,
    shared_strings: list[str],
):
    yielded = False
    try:
        data = zf.read(sheet_path)
        for match in _INLINE_A_CELL_RE.finditer(data):
            text = unescape(match.group(1).decode("utf-8", errors="replace"))
            timestamp = _parse_naive_timestamp(text)
            if timestamp is not None:
                yielded = True
                yield timestamp
    except Exception:
        yielded = False

    if yielded:
        return

    for row_index, row in enumerate(_iter_sheet_rows(zf, sheet_path, shared_strings)):
        if row_index == 0 or len(row) < 1:
            continue
        timestamp = _parse_naive_timestamp(row[0])
        if timestamp is not None:
            yield timestamp


def _fast_inline_sheet_time_summary(
    zf: ZipFile,
    sheet_path: str,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int, Counter[float]] | None:
    try:
        data = zf.read(sheet_path)
    except Exception:
        return None

    first_samples: list[pd.Timestamp] = []
    for match in _INLINE_A_CELL_RE.finditer(data):
        text = unescape(match.group(1).decode("utf-8", errors="replace"))
        timestamp = _parse_naive_timestamp(text)
        if timestamp is None:
            continue
        first_samples.append(timestamp)
        if len(first_samples) >= 300:
            break
    if not first_samples:
        return None

    last_time = _find_last_inline_time(data) or first_samples[-1]
    interval_counts: Counter[float] = Counter()
    for previous, current in zip(first_samples, first_samples[1:]):
        delta = (current - previous).total_seconds()
        if delta > 0:
            interval_counts[float(delta)] += 1

    # Row count is used only for preview. Counting XML row tags in C is much
    # faster than parsing every row; subtract the header row when present.
    rows = max(0, data.count(b"<row") - 1)
    return first_samples[0], last_time, rows, interval_counts


def _find_last_inline_time(data: bytes) -> pd.Timestamp | None:
    end = len(data)
    while True:
        start = data.rfind(b"<c ", 0, end)
        if start < 0:
            return None
        close = data.find(b"</c>", start)
        if close < 0:
            end = start
            continue
        snippet = data[start:close + 4]
        end = start
        if b'r="A' not in snippet:
            continue
        match = re.search(rb"<t[^>]*>(.*?)</t>", snippet, re.DOTALL)
        if not match:
            continue
        text = unescape(match.group(1).decode("utf-8", errors="replace"))
        timestamp = _parse_naive_timestamp(text)
        if timestamp is not None:
            return timestamp


def summarize_xlsx_groups(paths: Iterable[str | Path]) -> list[XlsxGroupSummary]:
    by_prefix: dict[str, list[XlsxFileSummary]] = {}
    for path in sorted(Path(p) for p in paths):
        if not is_real_xlsx_file(path):
            continue
        summary = read_xlsx_file_summary(str(path))
        by_prefix.setdefault(summary.prefix, []).append(summary)

    groups: list[XlsxGroupSummary] = []
    for prefix, files in sorted(by_prefix.items()):
        ordered = sorted(files, key=lambda item: (
            item.first_time or pd.Timestamp.max,
            Path(item.path).name,
        ))
        overlaps: list[tuple[str, str, float]] = []
        touches: list[tuple[str, str]] = []
        gaps: list[tuple[str, str, float]] = []
        for prev, current in zip(ordered, ordered[1:]):
            if prev.last_time is None or current.first_time is None:
                continue
            delta = (current.first_time - prev.last_time).total_seconds()
            prev_name = Path(prev.path).name
            current_name = Path(current.path).name
            if delta < 0:
                overlaps.append((prev_name, current_name, float(-delta)))
            elif delta == 0:
                touches.append((prev_name, current_name))
            else:
                gaps.append((prev_name, current_name, float(delta)))
        groups.append(XlsxGroupSummary(
            prefix=prefix,
            files=tuple(ordered),
            overlaps=tuple(overlaps),
            touches=tuple(touches),
            gaps=tuple(gaps),
        ))
    return groups


def _read_files_parallel(
    path_list: list[str],
    tz: str,
    channel_configs: list[dict] | None,
    workers: int,
    progress_callback: Callable[[int, int, str], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> tuple[list[dict[str, pd.Series]], list[dict[str, str]]]:
    """按文件读取多个 xlsx；workers > 1 时在全局共享 io 池上并行。

    - 结果按**原始文件顺序**回填，保证后续合并的确定性（对
      duplicate_policy 为 first/last 时的边界取舍语义很重要）；
    - 单文件失败不中断整体，记入 failed_files 并跳过；
    - cancel_check 返回 True 时抛 XlsxImportCancelled（在途文件让
      其跑完，仅取消尚未开始的任务）；
    - 进度粒度：已知通道配置时按"每读完一个 sheet"打一次进度，
      否则按文件打进度。回调可能在池线程中触发，上层如需更新
      UI 请自行经 Qt 信号排队回主线程。

    Returns:
        (raw_by_file, failed_files)
    """
    failed_files: list[dict[str, str]] = []
    progress_lock = threading.Lock()
    progress_done = [0]
    if channel_configs is not None:
        enabled_count = sum(
            1
            for config in channel_configs
            if config.get("enabled", True)
            and str(config.get("sheet", "")).strip()
            and str(config.get("name", "")).strip()
        )
        total_units = len(path_list) * max(enabled_count, 1)
    else:
        total_units = len(path_list)

    def _report(label: str) -> None:
        if progress_callback is None:
            return
        with progress_lock:
            progress_done[0] += 1
            current = progress_done[0]
        progress_callback(current, total_units, label)

    def _check_cancel() -> None:
        if cancel_check is not None and cancel_check():
            raise XlsxImportCancelled("用户取消了导入")

    def _read_one(path: str) -> dict[str, pd.Series]:
        if channel_configs is not None:
            return read_xlsx_raw_series(
                path,
                tz=tz,
                channel_configs=channel_configs,
                progress_callback=lambda: _report(Path(path).name),
            )
        result = read_xlsx_raw_series(path, tz=tz, channel_configs=None)
        _report(Path(path).name)
        return result

    _check_cancel()
    if workers <= 1 or len(path_list) <= 1:
        raw_by_file: list[dict[str, pd.Series]] = []
        for path in path_list:
            _check_cancel()
            try:
                raw_by_file.append(_read_one(path))
            except XlsxImportCancelled:
                raise
            except Exception as exc:
                logger.warning("读取 xlsx 失败，已跳过: %s (%s)", path, exc)
                failed_files.append({"path": path, "error": str(exc)})
        return raw_by_file, failed_files

    max_workers = max(1, min(int(workers), len(path_list)))
    # 复用全局共享 io 池；workers 参数作为本次操作的并发上限
    # （信号量），池自身 32 线程惰性创建，被其他导入任务共享。
    semaphore = threading.BoundedSemaphore(max_workers)

    def _guarded_read(path: str) -> dict[str, pd.Series]:
        with semaphore:
            return _read_one(path)

    results: list[dict[str, pd.Series] | None] = [None] * len(path_list)
    future_to_index = {
        io_pool.submit(_guarded_read, path): index
        for index, path in enumerate(path_list)
    }
    try:
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                logger.warning(
                    "读取 xlsx 失败，已跳过: %s (%s)", path_list[index], exc,
                )
                failed_files.append(
                    {"path": path_list[index], "error": str(exc)}
                )
            _check_cancel()
    except XlsxImportCancelled:
        for pending in future_to_index:
            pending.cancel()
        raise
    raw_by_file = [result for result in results if result is not None]
    return raw_by_file, failed_files


def read_xlsx_raw_series(
    path: str,
    tz: str,
    channel_configs: Iterable[dict] | None = None,
    progress_callback: Callable[[], None] | None = None,
) -> dict[str, pd.Series]:
    with ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf)
        sheet_paths = dict(_workbook_sheets(zf))
        configs = list(channel_configs or _configs_from_workbook(zf, shared_strings))

        series_by_name: dict[str, pd.Series] = {}
        for config in configs:
            if not config.get("enabled", True):
                continue
            sheet_name = str(config.get("sheet", "")).strip()
            output_name = str(config.get("name", "")).strip()
            if not sheet_name or not output_name:
                continue
            sheet_path = sheet_paths.get(sheet_name)
            if sheet_path is None:
                raise ValueError(f"xlsx 中找不到 sheet: {sheet_name}")

            timestamp_values: list[str] = []
            data_values: list[str | None] = []
            row_iter = _iter_sheet_rows_fast(zf, sheet_path)
            if row_iter is None:
                row_iter = _iter_sheet_rows(zf, sheet_path, shared_strings)
            for row_index, row in enumerate(row_iter):
                if row_index == 0:
                    continue
                if len(row) < 2:
                    continue
                if row[0] in (None, ""):
                    continue
                timestamp_values.append(str(row[0]))
                data_values.append(row[1])

            if not timestamp_values:
                continue

            scale = float(config.get("scale", 1.0))
            index, values = _parse_raw_columns(timestamp_values, data_values, tz)
            values = values * scale
            raw = pd.Series(values, index=index, name=output_name)
            raw = _deduplicate_series(raw)
            series_by_name[output_name] = raw
            # 每读完一个 sheet 回报一次进度（供上层按 文件×通道 计总刻度）
            if progress_callback is not None:
                progress_callback()

    if not series_by_name:
        raise ValueError("xlsx 中没有可导入的有效通道")
    return series_by_name


def merge_raw_series_by_channel(
    raw_by_file: Iterable[dict[str, pd.Series]],
    duplicate_policy: str = "error",
) -> tuple[dict[str, pd.Series], dict[str, dict]]:
    if duplicate_policy not in {"error", "first", "last", "mean"}:
        raise ValueError(f"未知重复时间处理策略: {duplicate_policy}")

    buckets: dict[str, list[pd.Series]] = {}
    for raw in raw_by_file:
        for channel_name, series in raw.items():
            buckets.setdefault(channel_name, []).append(series)

    merged: dict[str, pd.Series] = {}
    diagnostics: dict[str, dict] = {}
    for channel_name, items in buckets.items():
        combined = pd.concat(items).sort_index()
        resolved, diag = _resolve_duplicate_timestamps(
            combined,
            duplicate_policy=duplicate_policy,
            channel_name=channel_name,
        )
        merged[channel_name] = resolved
        diagnostics[channel_name] = diag
    return merged, diagnostics


def resample_series_to_grid(
    series: pd.Series,
    target_index: pd.DatetimeIndex,
    dt: pd.Timedelta,
    interpolation: str = INTERP_LINEAR,
    max_side_gap_factor: float = 1.0,
) -> pd.Series:
    if target_index.empty:
        return pd.Series(dtype=np.float64, index=target_index, name=series.name)
    if interpolation not in INTERPOLATION_METHODS:
        raise ValueError(f"未知插值方式: {interpolation}")
    if max_side_gap_factor <= 0:
        raise ValueError("max_side_gap_factor 必须大于 0")

    clean = _deduplicate_series(series.dropna()).sort_index()
    if clean.empty:
        return pd.Series(np.nan, index=target_index, name=series.name, dtype=np.float64)

    src_ns = _datetime_index_ns(clean.index)
    src_values = clean.to_numpy(dtype=np.float64, copy=False)
    target_ns = _datetime_index_ns(target_index)
    max_side_ns = int(dt.value * max_side_gap_factor)

    result = np.full(len(target_index), np.nan, dtype=np.float64)
    positions = np.searchsorted(src_ns, target_ns)

    exact_mask = np.zeros(len(target_index), dtype=bool)
    exact_candidates = positions < len(src_ns)
    exact_mask[exact_candidates] = (
        src_ns[positions[exact_candidates]] == target_ns[exact_candidates]
    )
    result[exact_mask] = src_values[positions[exact_mask]]

    if interpolation == INTERP_NONE:
        return pd.Series(result, index=target_index, name=series.name)

    unresolved = ~exact_mask
    if interpolation == INTERP_NEAREST:
        left = positions - 1
        right = positions
        left_ok = unresolved & (left >= 0)
        right_ok = unresolved & (right < len(src_ns))

        left_dist = np.full(len(target_index), np.iinfo(np.int64).max, dtype=np.int64)
        right_dist = np.full(len(target_index), np.iinfo(np.int64).max, dtype=np.int64)
        left_dist[left_ok] = target_ns[left_ok] - src_ns[left[left_ok]]
        right_dist[right_ok] = src_ns[right[right_ok]] - target_ns[right_ok]

        choose_left = left_dist <= right_dist
        nearest_ok = unresolved & (
            (choose_left & (left_dist <= max_side_ns))
            | (~choose_left & (right_dist <= max_side_ns))
        )
        result[nearest_ok & choose_left] = src_values[left[nearest_ok & choose_left]]
        result[nearest_ok & ~choose_left] = src_values[right[nearest_ok & ~choose_left]]
        return pd.Series(result, index=target_index, name=series.name)

    left = positions - 1
    right = positions
    bounds_ok = unresolved & (left >= 0) & (right < len(src_ns))
    if bounds_ok.any():
        left_dist = target_ns[bounds_ok] - src_ns[left[bounds_ok]]
        right_dist = src_ns[right[bounds_ok]] - target_ns[bounds_ok]
        side_ok = (left_dist <= max_side_ns) & (right_dist <= max_side_ns)
        indices = np.flatnonzero(bounds_ok)[side_ok]
        if len(indices):
            left_i = left[indices]
            right_i = right[indices]
            span = (src_ns[right_i] - src_ns[left_i]).astype(np.float64)
            weight = (target_ns[indices] - src_ns[left_i]).astype(np.float64) / span
            result[indices] = src_values[left_i] + weight * (src_values[right_i] - src_values[left_i])

    return pd.Series(result, index=target_index, name=series.name)


class XlsxLoader(EMDataLoader):
    @staticmethod
    def load(
        path: str,
        tz: str = "Asia/Shanghai",
        channel_configs: list[dict] | None = None,
        name: str | None = None,
        dt_override: float | None = None,
        interpolation: str = INTERP_LINEAR,
        max_side_gap_factor: float = 1.0,
    ) -> XlsxEMData:
        if dt_override is None or dt_override <= 0:
            summary = read_xlsx_header(path)
            inferred = summary.inferred_dt_sec
            if inferred is None or inferred <= 0:
                raise ValueError("无法自动检测采样周期，请手动指定 dt")
            dt = pd.Timedelta(seconds=float(inferred))
        else:
            dt = pd.Timedelta(seconds=float(dt_override))

        raw_series = read_xlsx_raw_series(path, tz=tz, channel_configs=channel_configs)
        start = min(series.index[0] for series in raw_series.values())
        end = max(series.index[-1] for series in raw_series.values())
        target_index = pd.date_range(start=start, end=end, freq=dt)
        if target_index.empty:
            raise ValueError("目标时间轴为空，请检查 dt 和时间范围")

        resampled: dict[str, pd.Series] = {}
        diagnostics: dict[str, dict] = {}
        for channel_name, raw in raw_series.items():
            aligned = resample_series_to_grid(
                raw,
                target_index,
                dt=dt,
                interpolation=interpolation,
                max_side_gap_factor=max_side_gap_factor,
            )
            resampled[channel_name] = aligned
            diagnostics[channel_name] = {
                "raw_points": int(raw.size),
                "imported_points": int(aligned.size),
                "nan_points": int(aligned.isna().sum()),
            }

        diagnostics["xlsx_import"] = {
            "interpolation": interpolation,
            "max_side_gap_factor": float(max_side_gap_factor),
            "dt_seconds": float(dt.total_seconds()),
        }
        return XlsxEMData(
            channel_series=resampled,
            timezone=tz,
            dt=dt,
            path=path,
            name=name or Path(path).stem,
            diagnostics=diagnostics,
        )

    @staticmethod
    def load_group(
        paths: Iterable[str],
        tz: str = "Asia/Shanghai",
        channel_configs: list[dict] | None = None,
        name: str | None = None,
        dt_override: float | None = None,
        interpolation: str = INTERP_LINEAR,
        max_side_gap_factor: float = 1.0,
        duplicate_policy: str = "error",
        workers: int = 1,
        progress_callback: Callable[[int, int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> XlsxEMData:
        path_list = [str(Path(path)) for path in paths]
        if not path_list:
            raise ValueError("没有选择 xlsx 文件")

        # 预筛打不开的文件（zip 中心目录级轻检查），贯彻"跳过并继续"
        # 策略；后续 sheet 级失败由线程池内逐文件 try/except 兜住。
        unreadable: list[dict[str, str]] = []
        readable: list[str] = []
        for path in path_list:
            try:
                with ZipFile(path):
                    pass
            except Exception as exc:
                logger.warning("无法打开 xlsx，已跳过: %s (%s)", path, exc)
                unreadable.append({"path": path, "error": str(exc)})
            else:
                readable.append(path)
        if not readable:
            details = "; ".join(
                f"{item['path']}: {item['error']}" for item in unreadable
            )
            raise ValueError(f"所有 xlsx 文件都读取失败: {details}")

        summaries = summarize_xlsx_groups(readable)
        if len(summaries) > 1:
            prefixes = ", ".join(group.prefix for group in summaries)
            raise ValueError(f"load_group 只能导入同一前缀文件，当前包含: {prefixes}")
        group_summary = summaries[0]

        if dt_override is None or dt_override <= 0:
            inferred = group_summary.inferred_dt_sec
            if inferred is None or inferred <= 0:
                raise ValueError("无法自动检测采样周期，请手动指定 dt")
            dt = pd.Timedelta(seconds=float(inferred))
        else:
            dt = pd.Timedelta(seconds=float(dt_override))

        raw_by_file, pool_failed = _read_files_parallel(
            readable,
            tz=tz,
            channel_configs=channel_configs,
            workers=workers,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        failed_files = unreadable + pool_failed
        if not raw_by_file:
            details = "; ".join(
                f"{item['path']}: {item['error']}" for item in failed_files
            )
            raise ValueError(f"所有 xlsx 文件都读取失败: {details}")
        raw_series, duplicate_diagnostics = merge_raw_series_by_channel(
            raw_by_file,
            duplicate_policy=duplicate_policy,
        )
        start = min(series.index[0] for series in raw_series.values())
        end = max(series.index[-1] for series in raw_series.values())
        target_index = pd.date_range(start=start, end=end, freq=dt)
        if target_index.empty:
            raise ValueError("目标时间轴为空，请检查 dt 和时间范围")

        resampled: dict[str, pd.Series] = {}
        diagnostics: dict[str, dict] = {}
        for channel_name, raw in raw_series.items():
            aligned = resample_series_to_grid(
                raw,
                target_index,
                dt=dt,
                interpolation=interpolation,
                max_side_gap_factor=max_side_gap_factor,
            )
            resampled[channel_name] = aligned
            duplicate_diag = duplicate_diagnostics.get(channel_name, {})
            diagnostics[channel_name] = {
                "raw_points": int(raw.size),
                "imported_points": int(aligned.size),
                "nan_points": int(aligned.isna().sum()),
                **duplicate_diag,
            }

        diagnostics["xlsx_import"] = {
            "files": path_list,
            "prefix": group_summary.prefix,
            "interpolation": interpolation,
            "max_side_gap_factor": float(max_side_gap_factor),
            "dt_seconds": float(dt.total_seconds()),
            "duplicate_policy": duplicate_policy,
            "overlap_count": len(group_summary.overlaps),
            "touch_count": len(group_summary.touches),
            "gap_count": len(group_summary.gaps),
            "failed_files": failed_files,
        }
        return XlsxEMData(
            channel_series=resampled,
            timezone=tz,
            dt=dt,
            path=path_list[0],
            name=name or group_summary.prefix,
            diagnostics=diagnostics,
        )


def _configs_from_workbook(zf: ZipFile, shared_strings: list[str]) -> list[dict]:
    configs: list[dict] = []
    for sheet_name, sheet_path in _workbook_sheets(zf):
        first_row = next(_iter_sheet_rows(zf, sheet_path, shared_strings), [])
        channel_name = ""
        if len(first_row) > 1 and first_row[1]:
            channel_name = str(first_row[1]).strip()
        if not channel_name:
            channel_name = sheet_name.split("_", 1)[0].strip()
        configs.append({
            "sheet": sheet_name,
            "source_name": channel_name,
            "name": channel_name,
            "enabled": bool(channel_name),
            "scale": 1.0,
        })
    return configs


def _workbook_sheets(zf: ZipFile) -> list[tuple[str, str]]:
    workbook = fromstring(zf.read("xl/workbook.xml"))
    rels = fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rid_to_target = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall(f"{OFFICE_REL_NS}Relationship")
    }
    sheets: list[tuple[str, str]] = []
    sheet_root = workbook.find(f"{XLSX_MAIN_NS}sheets")
    if sheet_root is None:
        return sheets
    for sheet in sheet_root:
        name = sheet.attrib["name"]
        rid = sheet.attrib[f"{XLSX_REL_NS}id"]
        target = rid_to_target[rid]
        if target.startswith("/"):
            normalized = target.lstrip("/")
        else:
            normalized = posixpath.normpath(posixpath.join("xl", target))
        sheets.append((name, normalized))
    return sheets


def _read_shared_strings(zf: ZipFile) -> list[str]:
    try:
        root = fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    strings: list[str] = []
    for item in root.findall(f"{XLSX_MAIN_NS}si"):
        strings.append("".join(text.text or "" for text in item.iter(f"{XLSX_MAIN_NS}t")))
    return strings


def _iter_sheet_rows(zf: ZipFile, sheet_path: str, shared_strings: list[str]):
    with zf.open(sheet_path) as sheet_file:
        for _event, elem in iterparse(sheet_file, events=("end",)):
            if elem.tag != f"{XLSX_MAIN_NS}row":
                continue
            cells: dict[int, str | None] = {}
            max_column = -1
            for cell in elem.findall(f"{XLSX_MAIN_NS}c"):
                ref = cell.attrib.get("r", "")
                column = _column_index(ref)
                if column is None:
                    column = max_column + 1
                max_column = max(max_column, column)
                cells[column] = _cell_value(cell, shared_strings)
            yield [cells.get(i) for i in range(max_column + 1)]
            elem.clear()


def _iter_sheet_rows_fast(zf: ZipFile, sheet_path: str):
    """try Rust first; fall back to Python regex fast path; else None.

    Rust 路径调 ``read_xlsx_sheet_inline_str_rs``，单趟 StAX 解析 inlineStr
    sheet，比 Python regex 快约 5x。
    """
    # Rust 路径要求 zip 已打开且能 re-open 二次 read；改用 ID-From-Archive
    # 不直接走 zf 传入，因为 zip 4.x 在 archive 被 freeze 时不能并发 read。
    # 这里我们用 archive 持有 path 来重新从 zip 文件读到 sheet bytes。
    archive_path = getattr(zf, 'filename', None)
    if archive_path is None:
        archive_path = getattr(zf, 'fp', None) and getattr(zf.fp, 'name', None)
    if _HAS_RUST_XLSX and archive_path:
        try:
            data = zf.read(sheet_path)
        except Exception:
            data = None
        else:
            # 仅当 sheet 是 inlineStr 格式才走 Rust
            if data is not None and b'inlineStr' in data and b'<c ' in data and b'<row' in data:
                def _rust_generator():
                    col_a, col_b = rust_lib.read_xlsx_sheet_inline_str_rs(
                        str(archive_path), sheet_path,
                    )
                    for a, b in zip(col_a, col_b):
                        yield [a, b]
                return _rust_generator()

    # Python regex fast path (fallback)
    try:
        data = zf.read(sheet_path)
    except Exception:
        return None
    if b"<row" not in data or b"<c " not in data or b"inlineStr" not in data:
        return None

    def _generator():
        for row_match in _INLINE_ROW_RE.finditer(data):
            row_body = row_match.group(1)
            cells: list[str | None] = []
            for cell_match in _INLINE_CELL_RE.finditer(row_body):
                cells.append(_extract_inline_cell_text(cell_match.group(2)))
            yield cells

    return _generator()


def _extract_inline_cell_text(cell_body: bytes) -> str | None:
    match = re.search(rb"<t[^>]*>(.*?)</t>", cell_body, re.DOTALL)
    if not match:
        return ""
    return unescape(match.group(1).decode("utf-8", errors="replace"))


def _cell_value(cell, shared_strings: list[str]) -> str | None:
    typ = cell.attrib.get("t")
    if typ == "inlineStr":
        inline = cell.find(f"{XLSX_MAIN_NS}is")
        if inline is None:
            return ""
        return "".join(text.text or "" for text in inline.iter(f"{XLSX_MAIN_NS}t"))

    value = cell.find(f"{XLSX_MAIN_NS}v")
    if value is None or value.text is None:
        return None
    raw = value.text
    if typ == "s":
        try:
            return shared_strings[int(raw)]
        except (IndexError, ValueError):
            return raw
    return raw


def _column_index(cell_ref: str) -> int | None:
    letters = ""
    for char in cell_ref:
        if char.isalpha():
            letters += char.upper()
        else:
            break
    if not letters:
        return None
    column = 0
    for char in letters:
        column = column * 26 + (ord(char) - ord("A") + 1)
    return column - 1


def _parse_naive_timestamp(value) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp


def _parse_local_timestamp(value, tz: str) -> pd.Timestamp | None:
    timestamp = _parse_naive_timestamp(value)
    if timestamp is None:
        return None
    timezone = pytz.timezone(tz)
    return timestamp.tz_localize(timezone).tz_convert("UTC")


def _parse_float(value) -> float:
    if value in (None, ""):
        return np.nan
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return np.nan


def _parse_raw_columns(
    timestamp_values: list[str],
    data_values: list[str | None],
    tz: str,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    parsed_times = pd.to_datetime(
        timestamp_values,
        errors="coerce",
        format="%Y-%m-%d %H:%M:%S",
    )
    valid = ~pd.isna(parsed_times)
    index = pd.DatetimeIndex(parsed_times[valid])
    timezone = pytz.timezone(tz)
    index = index.tz_localize(timezone).tz_convert("UTC")

    numeric_values = pd.to_numeric(pd.Series(data_values), errors="coerce").to_numpy(
        dtype=np.float64,
        copy=False,
    )
    return index, numeric_values[valid]


def _deduplicate_series(series: pd.Series) -> pd.Series:
    if not series.index.has_duplicates:
        return series.sort_index()
    return series.groupby(level=0).mean().sort_index()


def _datetime_index_ns(index: pd.DatetimeIndex) -> np.ndarray:
    return index.to_numpy(dtype="datetime64[ns]").astype(np.int64)


def _resolve_duplicate_timestamps(
    series: pd.Series,
    duplicate_policy: str,
    channel_name: str,
) -> tuple[pd.Series, dict]:
    if not series.index.has_duplicates:
        return series.sort_index(), {
            "duplicate_points": 0,
            "duplicate_timestamps": 0,
            "conflict_timestamps": 0,
        }

    grouped = series.groupby(level=0, sort=True)
    counts = grouped.size()
    duplicate_timestamps = int((counts > 1).sum())
    duplicate_points = int((counts - 1).clip(lower=0).sum())
    mins = grouped.min()
    maxs = grouped.max()
    conflict_mask = ((maxs - mins).abs() > 0) & ~(mins.isna() & maxs.isna())
    conflict_timestamps = int(conflict_mask.sum())
    if conflict_timestamps and duplicate_policy == "error":
        first_conflict = conflict_mask[conflict_mask].index[0]
        raise ValueError(
            f"通道 {channel_name} 存在重复时间但数值不一致: "
            f"{first_conflict} 等 {conflict_timestamps} 个时间点"
        )

    if duplicate_policy == "last":
        resolved = series[~series.index.duplicated(keep="last")]
    elif duplicate_policy == "mean":
        resolved = grouped.mean()
    else:
        resolved = series[~series.index.duplicated(keep="first")]

    return resolved.sort_index(), {
        "duplicate_points": duplicate_points,
        "duplicate_timestamps": duplicate_timestamps,
        "conflict_timestamps": conflict_timestamps,
    }
