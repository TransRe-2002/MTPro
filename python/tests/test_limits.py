"""utils/limits.py 的单元测试（纯函数，无 Qt 依赖）。"""

import random

from utils.limits import (
    MAX_POINTS_PER_CHANNEL,
    format_points,
    suggested_decimate_factor,
)
from utils.signal_decimate import decimated_length


class TestSuggestedDecimateFactor:
    def test_none_when_within_limit(self):
        assert suggested_decimate_factor(100) is None
        assert suggested_decimate_factor(MAX_POINTS_PER_CHANNEL) is None

    def test_small_overshoot_needs_factor_two(self):
        assert suggested_decimate_factor(MAX_POINTS_PER_CHANNEL + 1) == 2

    def test_round_multiples(self):
        cap = MAX_POINTS_PER_CHANNEL
        assert suggested_decimate_factor(2 * cap) == 2
        assert suggested_decimate_factor(3 * cap) == 3

    def test_custom_limit_and_minimality(self):
        rng = random.Random(7)
        for _ in range(200):
            npts = rng.randint(2, 10_000)
            cap = rng.randint(2, 500)
            f = suggested_decimate_factor(npts, cap)
            if npts <= cap:
                assert f is None
                continue
            assert f is not None and f >= 2
            # 建议因子确实压到上限以内
            assert decimated_length(npts, f) <= cap
            # 且是最小的：小一档就压不进去
            if f > 2:
                assert decimated_length(npts, f - 1) > cap

    def test_degenerate_cap_clamped(self):
        # max_points < 2 时按 2 处理：10 点压到 ≤2 需因子 5
        assert suggested_decimate_factor(10, 1) == 5


class TestFormatPoints:
    def test_wan(self):
        assert format_points(15_768_000) == "1576.8万"
        assert format_points(MAX_POINTS_PER_CHANNEL) == "500.0万"

    def test_yi(self):
        assert format_points(200_000_000) == "2.00亿"

    def test_thousands_separator(self):
        assert format_points(9_999) == "9,999"


def test_xlsx_io_reexport_keeps_compat():
    """旧导入路径 `from io_utils.xlsx_io import ...` 必须继续可用。"""
    from io_utils.xlsx_io import MAX_POINTS_PER_CHANNEL as cap2
    from io_utils.xlsx_io import format_points as fp2

    assert cap2 == MAX_POINTS_PER_CHANNEL
    assert fp2 is format_points
