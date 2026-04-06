from gfz_client import GFZClient
import pandas as pd
import numpy as np
import os
from pandas import HDFStore
from requests.exceptions import ConnectionError
from typing import Optional


def align_to_3hour_boundary(timestamp: pd.Timestamp, direction: str = 'floor') -> pd.Timestamp:
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize('UTC')
    else:
        timestamp = timestamp.tz_convert('UTC')

    hour = timestamp.hour

    if direction == 'floor':
        aligned_hour = (hour // 3) * 3
    elif direction == 'ceil':
        if hour % 3 == 0 and timestamp.minute == 0 and timestamp.second == 0:
            aligned_hour = hour
        else:
            aligned_hour = ((hour // 3) + 1) * 3
            if aligned_hour >= 24:
                timestamp = timestamp + pd.Timedelta(days=1)
                aligned_hour = 0
    else:
        raise ValueError("direction must be 'floor' or 'ceil'")

    return timestamp.replace(hour=aligned_hour, minute=0, second=0, microsecond=0)


def fetch_kp(start: pd.Timestamp, end: pd.Timestamp, hdf5_path: str = 'kp.h5') -> Optional[pd.DataFrame]:
    """
    获取Kp指数数据，返回含 ['Kp_datetime', 'Kp'] 两列的DataFrame。
    优先从本地h5读取，缺失部分从网络补充并写回h5；网络不可用时缺失点填NaN。

    Returns:
        pd.DataFrame，列为 ['Kp_datetime', 'Kp']，与原始fetch_kp返回格式一致
    """
    # ── 1. 时区统一 & 对齐到3小时边界 ─────────────────────────────
    if start.tz is None:
        start = start.tz_localize('UTC')
    else:
        start = start.tz_convert('UTC')

    if end.tz is None:
        end = end.tz_localize('UTC')
    else:
        end = end.tz_convert('UTC')

    aligned_start = align_to_3hour_boundary(start, direction='ceil')
    aligned_end   = align_to_3hour_boundary(end,   direction='floor')

    if aligned_start != start or aligned_end != end:
        print(f"时间已对齐到3小时边界：")
        print(f"  原始范围: {start} 到 {end}")
        print(f"  调整后:   {aligned_start} 到 {aligned_end}")

    if aligned_start > aligned_end:
        print("对齐后起始时间晚于结束时间，无有效时间范围")
        return None

    # ── 2. 生成期望的完整3小时整点时间序列 ────────────────────────
    expected_times = pd.date_range(
        start=aligned_start,
        end=aligned_end,
        freq='3h',
        tz='UTC'
    )
    print(f"期望时间点共 {len(expected_times)} 个：{expected_times[0]} 到 {expected_times[-1]}")

    # ── 3. 从本地h5加载已有数据（内部用index存储） ─────────────────
    local_df = _load_kp_internal(hdf5_path)  # index=Kp_datetime, col=Kp

    if local_df is not None and not local_df.empty:
        existing_times = set(local_df.index)
        missing_times  = [t for t in expected_times if t not in existing_times]
    else:
        local_df      = None
        missing_times = list(expected_times)

    print(f"本地已有 {len(expected_times) - len(missing_times)} 个时间点，"
          f"缺失 {len(missing_times)} 个时间点")

    # ── 4. 无缺失 → 直接从本地返回 ────────────────────────────────
    if not missing_times:
        result = local_df.loc[local_df.index.isin(expected_times)].sort_index()
        print(f"本地数据完整，直接返回 {len(result)} 条记录")
        return _to_output_df(result)

    # ── 5. 有缺失 → 从第一个缺失点开始请求网络 ────────────────────
    fetch_start = missing_times[0]
    fetch_end   = expected_times[-1]
    print(f"尝试从网络获取缺失数据：{fetch_start} 到 {fetch_end}")

    net_df = _fetch_kp_from_network(fetch_start, fetch_end)  # index=Kp_datetime, col=Kp

    # ── 6a. 网络成功 → 合并写回h5，返回结果 ───────────────────────
    if net_df is not None and not net_df.empty:
        frames = [df for df in [local_df, net_df] if df is not None and not df.empty]
        merged_df = pd.concat(frames)
        merged_df = merged_df[~merged_df.index.duplicated(keep='last')].sort_index()
        merged_df.index.name = 'Kp_datetime'

        _save_kp_internal(merged_df, hdf5_path)
        print(f"合并数据：本地 {len(local_df) if local_df is not None else 0} 条 "
              f"+ 网络 {len(net_df)} 条 = 合并后 {len(merged_df)} 条")

        result = merged_df.loc[merged_df.index.isin(expected_times)].sort_index()
        return _to_output_df(result)

    # ── 6b. 网络失败 → 缺失点填NaN ────────────────────────────────
    print(f"网络获取失败，将 {len(missing_times)} 个缺失时间点填充为 NaN")
    nan_df = pd.DataFrame(
        {'Kp': np.nan},
        index=pd.DatetimeIndex(missing_times, tz='UTC', name='Kp_datetime')
    )
    frames = [df for df in [local_df, nan_df] if df is not None and not df.empty]
    result_df = pd.concat(frames)
    result_df = result_df[~result_df.index.duplicated(keep='last')].sort_index()
    result_df.index.name = 'Kp_datetime'

    result = result_df.loc[result_df.index.isin(expected_times)].sort_index()
    return _to_output_df(result)


def _to_output_df(df: pd.DataFrame) -> pd.DataFrame:
    """将内部以index存储的DataFrame转为对外输出格式：['Kp_datetime', 'Kp'] 两列"""
    out = df.reset_index()
    out['Kp_datetime'] = out['Kp_datetime'].dt.tz_convert('UTC+00:00')
    return out[['Kp_datetime', 'Kp']]


def _fetch_kp_from_network(fetch_start: pd.Timestamp, fetch_end: pd.Timestamp) -> Optional[pd.DataFrame]:
    """从GFZ网络获取Kp数据，返回以Kp_datetime为index的DataFrame，失败返回None"""
    try:
        client = GFZClient()
        data = client.get_nowcast(
            start_time=fetch_start.strftime('%Y-%m-%dT%H:%M:%SZ'),
            end_time=fetch_end.strftime('%Y-%m-%dT%H:%M:%SZ'),
            index="Kp"
        )
        net_df = pd.DataFrame({
            'Kp_datetime': pd.to_datetime(data['datetime']).tz_convert('UTC'),
            'Kp': data['Kp']
        }).set_index('Kp_datetime').sort_index()
        return net_df
    except (ConnectionError, Exception) as e:
        print(f"网络获取Kp数据失败: {e}")
        return None


def _save_kp_internal(df: pd.DataFrame, hdf5_path: str):
    """保存以Kp_datetime为index的DataFrame到h5"""
    if df is None or df.empty:
        return
    try:
        with HDFStore(hdf5_path, mode='w') as store:
            store.put('kp_data', df, format='fixed')
        print(f"数据已保存到 {hdf5_path}，共 {len(df)} 条记录")
    except Exception as e:
        print(f"保存数据时出错: {e}")


def _load_kp_internal(hdf5_path: str) -> Optional[pd.DataFrame]:
    """从h5读取数据，返回以Kp_datetime为index的DataFrame"""
    if not os.path.exists(hdf5_path):
        print(f"文件 {hdf5_path} 不存在")
        return None
    try:
        with HDFStore(hdf5_path, mode='r') as store:
            if 'kp_data' not in store:
                return None
            df = store['kp_data']
            print(f"从 {hdf5_path} 加载了 {len(df)} 条记录")
            return df
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None


import unittest


class Test(unittest.TestCase):
    def test_fetch_kp_then_plot(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from math import inf

        end   = pd.Timestamp.now(tz='UTC+08:00')
        start = end - pd.Timedelta(days=2000)

        print("正在从GFZ获取Kp指数数据...")
        kp_df = fetch_kp(start, end)
        if kp_df is None:
            self.fail("Failed to fetch Kp index data")

        print(kp_df)

        bounds     = [0, 4, 6, inf]
        colors_rgb = [
            (52/255,  152/255, 219/255),
            (243/255, 156/255,  18/255),
            (231/255,  76/255,  60/255)
        ]
        cmap = ListedColormap(colors_rgb)
        norm = BoundaryNorm(bounds, cmap.N)
        bar_colors = cmap(norm(kp_df['Kp']))

        plt.figure(figsize=(12, 6))
        plt.bar(kp_df['Kp_datetime'], kp_df['Kp'],
                width=0.12, edgecolor="white", linewidth=0.7, color=bar_colors)
        plt.xlim(kp_df['Kp_datetime'][0], kp_df['Kp_datetime'][35])
        plt.title('Kp Index')
        plt.xlabel('Datetime')
        plt.ylabel('Kp Index')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        kp_df.plot()