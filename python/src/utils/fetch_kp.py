from gfz_client import GFZClient
import pandas as pd
import pytz as pytz
from requests.exceptions import ConnectionError
from typing import Optional

def fetch_kp(start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.DataFrame]:
    client = GFZClient()

    if start.tz != pytz.UTC and end.tz != pytz.UTC:
        start = start.tz_convert('UTC')
        end = end.tz_convert('UTC')

    try:
        data = client.get_nowcast(start_time=start.strftime('%Y-%m-%dT%H:%M:%SZ'), end_time=end.strftime('%Y-%m-%dT%H:%M:%SZ'), index="Kp")
    except ConnectionError:
        return None

    df = pd.DataFrame({
        'Kp_datetime': pd.to_datetime(data['datetime']).tz_convert('UTC+00:00'),
        'Kp': data['Kp']
    })
    return df


import unittest
class Test(unittest.TestCase):
    def test_fetch_kp_then_plot(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from math import inf
        # 设置时间范围
        end = pd.Timestamp.now(tz='UTC+08:00')
        start = end - pd.Timedelta(days=30)

        # 获取数据
        print("正在从GFZ获取Kp指数数据...")
        kp_df = fetch_kp(start, end)
        if kp_df is None:
            self.fail("Failed to fetch Kp index data")

        print(kp_df)

        # 绘制Kp指数图
        bounds = [0, 4, 6, inf] # 边界点
        # 为每个区间指定颜色：绿色、黄色、红色
        colors_rgb = [
            (52/255, 152/255, 219/255),   # <4: 专业蓝
            (243/255, 156/255, 18/255),   # 4-6: 琥珀黄
            (231/255, 76/255, 60/255)     # >6: 深空警报红
        ]
        cmap = ListedColormap(colors_rgb)
        norm = BoundaryNorm(bounds, cmap.N)

        # 3. 根据数值获取对应的颜色
        bar_colors = cmap(norm(kp_df['Kp']))
        plt.figure(figsize=(12, 6))
        plt.bar(kp_df['Kp_datetime'], kp_df['Kp'], width=0.12, edgecolor="white", linewidth=0.7, color=bar_colors)
        plt.xlim(kp_df['Kp_datetime'][0], kp_df['Kp_datetime'][35])
        plt.title('Kp Index')
        plt.xlabel('Datetime')
        plt.ylabel('Kp Index')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        kp_df.plot()