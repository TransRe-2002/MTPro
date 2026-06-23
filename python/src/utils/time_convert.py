from __future__ import annotations

from pandas import Timestamp, DatetimeIndex
from PySide6.QtCore import QDateTime, QTimeZone


def pts_to_qdt(pts: Timestamp, tz: str | None = None) -> QDateTime:
    """将 pandas.Timestamp 转换为 QDateTime。

    Args:
        pts: 时区感知的 Timestamp。
        tz: 目标时区名称。None（默认）时使用 Timestamp 自身的时区；
            传入如 ``'Asia/Shanghai'`` 则先 tz_convert 再创建对应时区的 QDateTime。

    Returns:
        QDateTime，其时区与 pts 的时区一致（或 tz 指定的目标时区）。

    Example:
        >>> ts = pd.Timestamp('2024-05-01 00:00:00', tz='Asia/Shanghai')
        >>> qdt = pts_to_qdt(ts)                 # QDateTime 显示为 Asia/Shanghai 时间
        >>> qdt = pts_to_qdt(ts, tz='UTC')       # QDateTime 显示为 UTC 时间
    """
    if tz is not None:
        pts = pts.tz_convert(tz)

    if pts.tz is None:
        # naive Timestamp → 当 UTC 处理
        qtz = QTimeZone.UTC
    else:
        qtz = QTimeZone(str(pts.tz).encode())

    return QDateTime.fromSecsSinceEpoch(int(pts.timestamp()), qtz)


def qdt_to_pts(qdt: QDateTime, tz: str = 'UTC') -> Timestamp:
    """将 QDateTime 转换为 pandas.Timestamp。

    Args:
        qdt: QDateTime 对象。
        tz: 目标时区名称，默认 ``'UTC'``。

    Returns:
        带指定时区的 Timestamp。

    Example:
        >>> qdt = QDateTime.currentDateTime()
        >>> ts_utc = qdt_to_pts(qdt)             # → Timestamp in UTC
        >>> ts_cst = qdt_to_pts(qdt, tz='Asia/Shanghai')  # → Timestamp in CST
    """
    return Timestamp(qdt.toSecsSinceEpoch(), unit='s', tz=tz)
