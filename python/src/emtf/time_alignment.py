"""Time-index alignment helpers for multi-station EMTF processing."""

from __future__ import annotations

import pandas as pd


def matching_remote_slice(
    local_index: pd.DatetimeIndex,
    remote_index: pd.DatetimeIndex,
    start: int,
    end: int,
) -> tuple[int, int]:
    """Return remote bounds that exactly match ``local_index[start:end]``.

    The array-based backends pair FFT windows by position.  Match their physical
    time ranges without requiring identical timezone representations; truncating
    independently selected arrays could otherwise pair different time windows.
    """
    local_times = pd.DatetimeIndex(local_index[start:end])
    remote_times = pd.DatetimeIndex(remote_index)
    if local_times.empty:
        raise ValueError("本地站选定时间范围为空")
    if not local_times.is_monotonic_increasing or not remote_times.is_monotonic_increasing:
        raise ValueError("本地站和远参考站的时间索引必须按升序排列")

    remote_start = int(remote_times.searchsorted(local_times[0], side="left"))
    remote_end = remote_start + len(local_times)
    if remote_end > len(remote_times) or not (remote_times[remote_start:remote_end] == local_times).all():
        raise ValueError(
            "远参考站在选定范围内没有与本地站逐样点对齐的数据；"
            "请选取两个台站共有的时间范围。"
        )
    return remote_start, remote_end


__all__ = ["matching_remote_slice"]
