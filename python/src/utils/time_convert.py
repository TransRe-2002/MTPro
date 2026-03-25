from __future__ import annotations

from pandas import Timestamp, DatetimeIndex
from PySide6.QtCore import QDateTime

def pts_to_qdt(pts: Timestamp|DatetimeIndex) -> QDateTime:
    """将 pandas.Timestamp 转换为 QDateTime"""
    return QDateTime.fromSecsSinceEpoch(int(pts.timestamp()))

def qdt_to_pts(qdt: QDateTime) -> Timestamp:
    """将 QDateTime 转换为 pandas.Timestamp"""
    return Timestamp(qdt.toSecsSinceEpoch(), unit='s', tz='UTC+08:00')

def test_time_convert():
    # 创建一个当前的 pandas.Timestamp
    now_pts = Timestamp.now(tz='UTC+08:00')
    now_qdt = QDateTime.currentDateTime()
    print(f"原始 pandas.Timestamp: {now_pts}")
    print(f"原始 QDateTime: {now_qdt.toString('yyyy-MM-dd HH:mm:ss')}")

    # 转换为 QDateTime
    qdt = pts_to_qdt(now_pts)  # 请替换为您选择的函数名
    print(f"转换后的 QDateTime: {qdt.toString('yyyy-MM-dd HH:mm:ss')}")

    # 再转换回来
    new_pts = qdt_to_pts(qdt)  # 请替换为您选择的函数名
    print(f"转换回来的 pandas.Timestamp: {new_pts}")

    # 检查是否一致（在秒级精度下）
    assert int(now_pts.timestamp()) == int(new_pts.timestamp())