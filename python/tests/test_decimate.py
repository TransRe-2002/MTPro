from __future__ import annotations

import numpy as np

import utils.decimate as dec
from utils.decimate import LODPyramid, minmax_decimate, slice_visible


def _spiky_series(n: int = 200_000, seed: int = 7):
    """生成带稀疏窄尖峰的正弦序列，用于验证 min/max 包络不丢尖峰。"""
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=np.float64)
    y = np.sin(2 * np.pi * x / 1000.0) + rng.normal(0, 0.05, n)
    # 每隔一段埋一个远高于背景的尖峰
    spike_positions = np.arange(500, n, 5000)
    y[spike_positions] += 50.0
    return x, y


def test_minmax_decimate_caps_points_and_preserves_extrema():
    x, y = _spiky_series()
    dx, dy = minmax_decimate(x, y, target_bins=1000)

    assert len(dx) <= 2 * 1000
    assert np.nanmax(dy) == np.nanmax(y)
    assert np.nanmin(dy) == np.nanmin(y)


def test_minmax_decimate_passthrough_for_small_data():
    x = np.arange(100, dtype=np.float64)
    y = x * 2.0
    dx, dy = minmax_decimate(x, y, target_bins=1000)
    np.testing.assert_array_equal(dx, x)
    np.testing.assert_array_equal(dy, y)


def test_lod_pyramid_zoomed_in_returns_exact_raw_slice():
    x, y = _spiky_series()
    pyramid = LODPyramid(x, y)

    # 100 个点的小视口、800 像素宽：可见点数远小于 4*像素 → 应返回原始切片
    xs, ys = pyramid.get(10_000.0, 10_100.0, pixel_width=800)
    mask = (x >= 10_000.0) & (x <= 10_100.0)
    np.testing.assert_array_equal(xs, x[mask])
    np.testing.assert_array_equal(ys, y[mask])


def test_lod_pyramid_zoomed_out_preserves_visible_extrema():
    x, y = _spiky_series()
    pyramid = LODPyramid(x, y)
    assert pyramid.level_count > 0

    xs, ys = pyramid.get(0.0, 100_000.0, pixel_width=800)
    mask = (x >= 0.0) & (x <= 100_000.0)
    # 返回点数受像素宽度约束
    assert len(xs) <= 800 * 4 + 2
    # min/max 包络保证可见范围内的极值不丢
    assert np.nanmax(ys) == np.nanmax(y[mask])
    assert np.nanmin(ys) == np.nanmin(y[mask])


def test_lod_pyramid_deep_zoom_on_large_data_is_not_empty():
    """回归：旧版按"层总点数"选层，大数据深放大时视口内会没有点。"""
    x, y = _spiky_series(2_000_000)
    pyramid = LODPyramid(x, y)

    # 200 点的极小视口：无论数据多大都应拿到原始分辨率切片
    xs, ys = pyramid.get(1_000_000.0, 1_000_200.0, pixel_width=800)
    assert len(xs) == 201
    np.testing.assert_array_equal(xs, x[1_000_000:1_000_201])


def test_lod_pyramid_get_normalizes_reversed_range():
    x, y = _spiky_series(10_000)
    pyramid = LODPyramid(x, y)
    forward = pyramid.get(1_000.0, 2_000.0, pixel_width=400)
    backward = pyramid.get(2_000.0, 1_000.0, pixel_width=400)
    np.testing.assert_array_equal(forward[0], backward[0])
    np.testing.assert_array_equal(forward[1], backward[1])


def test_get_full_overview_small_data_without_levels():
    x = np.arange(100, dtype=np.float64)
    y = x * 3.0
    pyramid = LODPyramid(x, y)
    assert pyramid.level_count == 0
    ox, oy = pyramid.get_full_overview(pixel_width=800)
    np.testing.assert_array_equal(ox, x)
    np.testing.assert_array_equal(oy, y)


def test_get_full_overview_large_data_is_bounded():
    x, y = _spiky_series()
    pyramid = LODPyramid(x, y)
    ox, oy = pyramid.get_full_overview(pixel_width=600)
    assert len(ox) <= max(256, 600 * 2) or len(ox) < len(x)


def test_slice_visible_matches_bruteforce_mask():
    x, y = _spiky_series(50_000)
    xs, ys = slice_visible(x, y, 10_000.0, 12_000.0, margin_ratio=0.1)

    margin = 2_000.0 * 0.1
    mask = (x >= 10_000.0 - margin) & (x <= 12_000.0 + margin)
    np.testing.assert_array_equal(xs, x[mask])
    np.testing.assert_array_equal(ys, y[mask])


def test_slice_visible_empty_range():
    x, y = _spiky_series(1_000)
    xs, ys = slice_visible(x, y, 5_000.0, 6_000.0)
    assert len(xs) == 0 and len(ys) == 0


def test_python_fallback_matches_rust_path(monkeypatch):
    """强制关闭 Rust 加速，验证 Python 回退路径行为一致。"""
    x, y = _spiky_series(50_000)
    pyramid_rs = LODPyramid(x, y)

    monkeypatch.setattr(dec, "_HAS_RUST", False)
    pyramid_py = LODPyramid(x, y)

    assert pyramid_py.level_count == pyramid_rs.level_count
    for (rx, ry), (px, py_) in zip(
        pyramid_rs._levels, pyramid_py._levels
    ):
        np.testing.assert_array_equal(rx, px)
        np.testing.assert_array_equal(ry, py_)


def test_pyramid_truncation_drops_oversized_top_levels():
    from utils.decimate import build_lod_pyramid

    n = 5_000_000  # 顶层 ~200 万点 > cap，下一层 ~80 万点 < cap
    x = np.arange(n, dtype=np.float64)
    y = np.sin(x / 1e5)

    full = build_lod_pyramid(x, y)
    trunc = build_lod_pyramid(x, y, max_finest_points=1_500_000)

    assert len(full[0][0]) > 1_500_000          # 不截顶时顶层确实超限
    assert len(trunc) < len(full)               # 截掉了顶层
    assert all(len(l[0]) <= 1_500_000 for l in trunc)
    # 链式推导不受影响：保留的最粗层与完整版逐点一致
    np.testing.assert_array_equal(trunc[-1][0], full[-1][0])
    np.testing.assert_array_equal(trunc[-1][1], full[-1][1])


def test_pyramid_truncation_noop_for_small_data():
    from utils.decimate import PYRAMID_MAX_FINEST_POINTS, build_lod_pyramid

    x, y = _spiky_series(200_000)   # 顶层 8 万点 << 150 万，截顶不应触发
    full = build_lod_pyramid(x, y)
    trunc = build_lod_pyramid(x, y, max_finest_points=PYRAMID_MAX_FINEST_POINTS)
    assert len(full) == len(trunc)


def test_lod_pyramid_truncated_still_serves_all_viewports():
    from utils.decimate import LODPyramid

    n = 5_000_000
    x = np.arange(n, dtype=np.float64)
    y = np.sin(x / 1e5)
    pyramid = LODPyramid(x, y, max_finest_points=1_500_000)

    # 极小视口 → 原始切片
    xs, ys = pyramid.get(1_000_000.0, 1_002_000.0, pixel_width=800)
    assert len(xs) == 2001
    # 中等视口 → 由保留的层兜底，非空
    xs, ys = pyramid.get(0.0, 2_000_000.0, pixel_width=800)
    assert len(xs) > 0 and len(xs) <= 800 * 4 + 2
    # 全览
    xs, ys = pyramid.get(0.0, float(n), pixel_width=800)
    assert len(xs) > 0
