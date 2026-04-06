from gfz_client import GFZClient
import pandas as pd
import pytz as pytz
from typing import Optional
from pandas import HDFStore
import os
import numpy as np
from sympy.physics.units import years


def align_to_3hour_boundary(timestamp: pd.Timestamp, direction: str = 'floor') -> pd.Timestamp:
    """
    将时间戳对齐到3小时边界（0, 3, 6, 9, 12, 15, 18, 21点）

    Args:
        timestamp: 要调整的时间戳
        direction: 'floor' 向下取整到上一个3小时点，'ceil' 向上取整到下一个3小时点

    Returns:
        调整后的时间戳
    """
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

    aligned = timestamp.replace(hour=aligned_hour, minute=0, second=0, microsecond=0)
    return aligned


def fetch_kp(start: pd.Timestamp, end: pd.Timestamp, hdf5_path: str = 'kp.h5') -> Optional[pd.DataFrame]:
    """
    获取Kp指数数据：优先从本地h5读取，缺失部分从网络补充并写回h5；
    网络不可用时缺失时间点填NaN。

    流程：
        1. 对齐时间到3小时边界，生成完整期望时间序列
        2. 从本地h5读取已有数据
        3. 找到第一个缺失时间点，作为网络请求的起点
        4. 尝试从网络获取 [第一个缺失点, end] 的数据并合并写回h5
        5. 网络失败则缺失时间点填NaN
        6. 返回 [start, end] 范围内的完整DataFrame

    Args:
        start: 开始时间（会自动向上对齐到3小时边界）
        end:   结束时间（会自动向下对齐到3小时边界）
        hdf5_path: HDF5文件路径

    Returns:
        完整时间范围的DataFrame，缺失值为NaN；若时间范围内无任何有效数据则返回None
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

    # ── 3. 从本地h5加载已有数据 ────────────────────────────────────
    local_df = load_kp(hdf5_path)

    if local_df is not None and not local_df.empty:
        existing_times = set(local_df.index)
        missing_times  = [t for t in expected_times if t not in existing_times]
    else:
        local_df      = pd.DataFrame(columns=['Kp'])
        missing_times = list(expected_times)

    print(f"本地已有 {len(expected_times) - len(missing_times)} 个时间点，"
          f"缺失 {len(missing_times)} 个时间点")

    # ── 4. 无缺失 → 直接返回本地数据 ─────────────────────────────
    if not missing_times:
        result = local_df.loc[local_df.index.isin(expected_times)].sort_index()
        print(f"本地数据完整，直接返回 {len(result)} 条记录")
        return result

    # ── 5. 有缺失 → 从第一个缺失时间点开始向网络请求 ──────────────
    fetch_start = missing_times[0]   # 第一个缺失点
    fetch_end   = expected_times[-1] # 最后期望时间点

    print(f"尝试从网络获取缺失数据：{fetch_start} 到 {fetch_end}")
    net_df = _fetch_kp_from_network(fetch_start, fetch_end)

    # ── 6a. 网络获取成功 → 合并写回h5 ─────────────────────────────
    if net_df is not None and not net_df.empty:
        merged_df = pd.concat([local_df, net_df])
        merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
        merged_df = merged_df.sort_index()

        save_kp(merged_df, hdf5_path)
        print(f"合并数据：本地 {len(local_df)} 条 + 网络 {len(net_df)} 条 "
              f"= 合并后 {len(merged_df)} 条")

        result = merged_df.loc[merged_df.index.isin(expected_times)].sort_index()
        return result

    # ── 6b. 网络获取失败 → 缺失时间点填NaN ───────────────────────
    print(f"网络获取失败，将 {len(missing_times)} 个缺失时间点填充为 NaN")
    nan_df = pd.DataFrame(
        {'Kp': np.nan},
        index=pd.DatetimeIndex(missing_times, tz='UTC', name='Kp_datetime')
    )

    result_df = pd.concat([local_df, nan_df])
    result_df = result_df[~result_df.index.duplicated(keep='last')]
    result_df = result_df.sort_index()

    result = result_df.loc[result_df.index.isin(expected_times)].sort_index()
    return result


def _fetch_kp_from_network(fetch_start: pd.Timestamp, fetch_end: pd.Timestamp) -> Optional[pd.DataFrame]:
    """
    从GFZ网络获取指定时间范围的Kp数据（内部函数）

    Args:
        fetch_start: 网络请求起始时间（已对齐到3小时边界）
        fetch_end:   网络请求结束时间（已对齐到3小时边界）

    Returns:
        成功返回DataFrame（index为UTC时间戳），失败返回None
    """
    try:
        client = GFZClient()
        data = client.get_nowcast(
            start_time=fetch_start.strftime('%Y-%m-%dT%H:%M:%SZ'),
            end_time=fetch_end.strftime('%Y-%m-%dT%H:%M:%SZ'),
            index="Kp"
        )

        net_df = pd.DataFrame({
            'Kp_datetime': pd.to_datetime(data['datetime']),
            'Kp': data['Kp']
        })
        net_df['Kp_datetime'] = net_df['Kp_datetime'].dt.tz_convert('UTC')
        net_df = net_df.set_index('Kp_datetime').sort_index()
        return net_df

    except Exception as e:
        print(f"网络获取Kp数据失败: {e}")
        return None


def save_kp(df: pd.DataFrame, hdf5_path: str):
    """
    保存DataFrame到HDF5文件

    Args:
        df: 要保存的DataFrame（索引应为Kp_datetime）
        hdf5_path: HDF5文件路径
    """
    if df is None or df.empty:
        print("DataFrame为空，跳过保存")
        return

    try:
        with HDFStore(hdf5_path, mode='w') as store:
            store.put('kp_data', df, format='fixed')
        print(f"数据已保存到 {hdf5_path}，共 {len(df)} 条记录")
    except Exception as e:
        print(f"保存数据时出错: {e}")


def load_kp(hdf5_path: str, start: Optional[pd.Timestamp] = None,
            end: Optional[pd.Timestamp] = None) -> Optional[pd.DataFrame]:
    """
    从HDF5文件加载Kp指数数据

    Args:
        hdf5_path: HDF5文件路径
        start: 可选的开始时间，用于筛选数据
        end: 可选的结束时间，用于筛选数据

    Returns:
        DataFrame或None
    """
    if not os.path.exists(hdf5_path):
        print(f"文件 {hdf5_path} 不存在")
        return None

    try:
        with HDFStore(hdf5_path, mode='r') as store:
            if 'kp_data' not in store:
                print(f"文件中没有找到 'kp_data' 数据集")
                return None

            df = store['kp_data']

            if start is not None or end is not None:
                if start is not None:
                    if start.tz is None:
                        start = start.tz_localize('UTC')
                    else:
                        start = start.tz_convert('UTC')
                    df = df[df.index >= start]

                if end is not None:
                    if end.tz is None:
                        end = end.tz_localize('UTC')
                    else:
                        end = end.tz_convert('UTC')
                    df = df[df.index <= end]

            print(f"从 {hdf5_path} 加载了 {len(df)} 条记录")
            return df

    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None


def get_kp_stats(hdf5_path: str = 'kp.h5') -> dict:
    """获取存储数据的统计信息"""
    df = load_kp(hdf5_path)
    if df is None or df.empty:
        return {"error": "无数据"}

    return {
        "总记录数": len(df),
        "时间范围": f"{df.index.min()} 到 {df.index.max()}",
        "Kp均值": df['Kp'].mean(),
        "Kp最大值": df['Kp'].max(),
        "Kp最小值": df['Kp'].min(),
        "数据缺失": df['Kp'].isna().sum()
    }


import unittest


class Test(unittest.TestCase):
    def test_time_alignment(self):
        """测试时间对齐到3小时边界的功能"""
        print("\n测试时间对齐功能：")
        print("=" * 60)

        test_cases = [
            ('2024-01-01 00:00:00', '2024-01-01 00:00:00', '2024-01-01 00:00:00'),
            ('2024-01-01 00:30:00', '2024-01-01 00:00:00', '2024-01-01 03:00:00'),
            ('2024-01-01 03:00:00', '2024-01-01 03:00:00', '2024-01-01 03:00:00'),
            ('2024-01-01 10:30:00', '2024-01-01 09:00:00', '2024-01-01 12:00:00'),
            ('2024-01-01 14:59:59', '2024-01-01 12:00:00', '2024-01-01 15:00:00'),
            ('2024-01-01 21:00:00', '2024-01-01 21:00:00', '2024-01-01 21:00:00'),
            ('2024-01-01 23:30:00', '2024-01-01 21:00:00', '2024-01-02 00:00:00'),
        ]

        for time_str, expected_floor, expected_ceil in test_cases:
            ts = pd.Timestamp(time_str, tz='UTC')
            floor_result = align_to_3hour_boundary(ts, 'floor')
            ceil_result  = align_to_3hour_boundary(ts, 'ceil')
            print(f"{time_str:25} -> floor: {floor_result.strftime('%Y-%m-%d %H:%M:%S'):20} "
                  f"| ceil: {ceil_result.strftime('%Y-%m-%d %H:%M:%S')}")
            self.assertEqual(floor_result, pd.Timestamp(expected_floor, tz='UTC'))
            self.assertEqual(ceil_result,  pd.Timestamp(expected_ceil,  tz='UTC'))

        print("✓ 所有时间对齐测试通过")

    def test_fetch_kp_then_plot(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from math import inf

        end   = pd.Timestamp.now(tz='UTC+08:00')
        start = end - pd.Timedelta(days=30)

        print("正在获取Kp指数数据（优先本地，缺失从网络补充）...")
        kp_df = fetch_kp(start, end, hdf5_path='kp.h5')
        if kp_df is None:
            self.fail("Failed to fetch Kp index data")

        kp_df_plot = kp_df.reset_index()
        print(kp_df_plot.head())

        bounds     = [0, 4, 6, inf]
        colors_rgb = [
            (52/255,  152/255, 219/255),
            (243/255, 156/255,  18/255),
            (231/255,  76/255,  60/255)
        ]
        cmap = ListedColormap(colors_rgb)
        norm = BoundaryNorm(bounds, cmap.N)
        bar_colors = cmap(norm(kp_df_plot['Kp']))

        plt.figure(figsize=(12, 6))
        plt.bar(kp_df_plot['Kp_datetime'], kp_df_plot['Kp'],
                width=0.12, edgecolor="white", linewidth=0.7, color=bar_colors)

        if len(kp_df_plot) > 35:
            plt.xlim(kp_df_plot['Kp_datetime'].iloc[0], kp_df_plot['Kp_datetime'].iloc[35])

        plt.title('Kp Index')
        plt.xlabel('Datetime')
        plt.ylabel('Kp Index')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def test_multiple_fetches(self):
        """测试多次获取数据的合并功能"""
        end1   = pd.Timestamp.now(tz='UTC')
        start1 = end1 - pd.Timedelta(days=10)
        df1    = fetch_kp(start1, end1, hdf5_path='kp_test.h5')

        end2   = pd.Timestamp.now(tz='UTC')
        start2 = end2 - pd.Timedelta(days=5)
        df2    = fetch_kp(start2, end2, hdf5_path='kp_test.h5')

        self.assertIsNotNone(df2)
        print(f"\n多次获取测试完成：")
        print(get_kp_stats('kp_test.h5'))


if __name__ == '__main__':
    print("=" * 50)
    print("示例1: 获取最近30天的数据")
    print("=" * 50)
    end   = pd.Timestamp.now(tz='UTC')
    start = end - pd.Timedelta(days=1000)
    df    = fetch_kp(start, end, hdf5_path='kp.h5')

    if df is not None:
        print("\n数据统计:")
        print(get_kp_stats('kp.h5'))
        print("\n最近5条记录:")
        print(df.tail().reset_index())

    print("\n" + "=" * 50)
    print("示例2: 再次获取最近7天（测试合并功能）")
    print("=" * 50)
    end   = pd.Timestamp.now(tz='UTC')
    start = end - pd.Timedelta(days=7)
    df    = fetch_kp(start, end, hdf5_path='kp.h5')

    if df is not None:
        print("\n合并后的数据统计:")
        print(get_kp_stats('kp.h5'))