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

    Remote-reference regression requires synchronous samples.  Truncating two
    independently selected ranges can silently pair different timestamps, so
    reject a range unless every local timestamp has its counterpart remotely.
    """
    local_times = pd.DatetimeIndex(local_index[start:end])
    remote_times = pd.DatetimeIndex(remote_index)
    if local_times.empty:
        raise ValueError("本地站选定时间范围为空")
    if not local_times.is_monotonic_increasing or not remote_times.is_monotonic_increasing:
        raise ValueError("本地站和远参考站的时间索引必须按升序排列")

    remote_start = int(remote_times.searchsorted(local_times[0], side="left"))
    remote_end = remote_start + len(local_times)
    if remote_end > len(remote_times) or not remote_times[remote_start:remote_end].equals(local_times):
        raise ValueError(
            "远参考站在选定范围内没有与本地站逐样点对齐的数据；"
            "请选取两个台站共有的时间范围。"
        )
    return remote_start, remote_end


__all__ = ["matching_remote_slice"]
