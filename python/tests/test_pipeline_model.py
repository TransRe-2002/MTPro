from __future__ import annotations

import numpy as np

from core.pipeline_model import (
    ChannelPipeline,
    PipelineBlock,
    PipelineLayer,
    SnapEngine,
)


def test_pipeline_block_defaults_are_unique():
    a = PipelineBlock()
    b = PipelineBlock()
    assert a.block_id != b.block_id
    assert a.status == "empty"
    assert a.params == {}
    assert a.input_snapshot is None
    assert a.output_snapshot is None


def test_pipeline_layer_sorted_blocks_orders_by_start_end():
    layer = PipelineLayer()
    layer.blocks = [
        PipelineBlock(start=100, end=200, block_id="b2"),
        PipelineBlock(start=0, end=50, block_id="b1"),
        PipelineBlock(start=0, end=80, block_id="b3"),
    ]
    ordered = layer.sorted_blocks()
    assert [b.block_id for b in ordered] == ["b1", "b3", "b2"]


def test_channel_pipeline_add_layer_increments_label():
    pipeline = ChannelPipeline(data_id=1, channel="Ex")
    first = pipeline.add_layer()
    second = pipeline.add_layer()
    assert first.label == "step 1"
    assert second.label == "step 2"
    assert pipeline.layers == [first, second]


def test_channel_pipeline_ref_property():
    pipeline = ChannelPipeline(data_id=7, channel="Hy")
    assert pipeline.ref == (7, "Hy")


def test_channel_pipeline_find_block_returns_layer_and_block():
    pipeline = ChannelPipeline(data_id=1, channel="Ex")
    layer = pipeline.add_layer()
    block = PipelineBlock(block_id="blk-1", start=0, end=100)
    layer.blocks.append(block)

    found = pipeline.find_block("blk-1")
    assert found is not None
    found_layer, found_block = found
    assert found_layer is layer
    assert found_block is block

    assert pipeline.find_block("missing") is None


def test_channel_pipeline_remove_block_returns_true_when_removed():
    pipeline = ChannelPipeline(data_id=1, channel="Ex")
    layer = pipeline.add_layer()
    block = PipelineBlock(block_id="blk-1")
    layer.blocks.append(block)

    assert pipeline.remove_block("blk-1") is True
    assert layer.blocks == []
    assert pipeline.remove_block("blk-1") is False


def test_snap_engine_update_edges_collects_block_boundaries():
    pipeline = ChannelPipeline(data_id=1, channel="Ex")
    layer = pipeline.add_layer()
    layer.blocks = [
        PipelineBlock(block_id="a", start=0, end=100),
        PipelineBlock(block_id="b", start=200, end=300),
    ]
    snap = SnapEngine()
    snap.update_edges([pipeline])
    assert snap._edges == [0, 100, 200, 300]


def test_snap_engine_update_edges_excludes_requested_block():
    pipeline = ChannelPipeline(data_id=1, channel="Ex")
    layer = pipeline.add_layer()
    layer.blocks = [
        PipelineBlock(block_id="a", start=0, end=100),
        PipelineBlock(block_id="b", start=200, end=300),
    ]
    snap = SnapEngine()
    snap.update_edges([pipeline], exclude_id="b")
    assert snap._edges == [0, 100]


def test_snap_engine_snap_returns_input_when_disabled():
    snap = SnapEngine()
    snap.enabled = False
    pts, guides = snap.snap(50, px_per_pt=1.0, data_len=100)
    assert pts == 50
    assert guides == []


def test_snap_engine_snap_clamps_to_data_len():
    snap = SnapEngine()
    snap.grid_pts = 1000
    pts, _ = snap.snap(5000, px_per_pt=1.0, data_len=200)
    assert 0 <= pts <= 200


def test_snap_engine_snap_snaps_to_edge_when_within_threshold():
    snap = SnapEngine()
    snap.grid_pts = 1000
    snap._edges = [100]
    pts, guides = snap.snap(105, px_per_pt=1.0, data_len=1000)
    assert pts == 100
    assert guides == [100]


def test_snap_engine_snap_beyond_threshold_still_picks_nearest_but_no_guides():
    snap = SnapEngine()
    snap.grid_pts = 1000
    snap._edges = [100]
    pts, guides = snap.snap(200, px_per_pt=1.0, data_len=1000)
    assert pts == 100
    assert guides == []


def test_snap_engine_snap_interval_preserves_span():
    snap = SnapEngine()
    snap.grid_pts = 1000
    start, _ = snap.snap_interval(150, 250, px_per_pt=1.0, data_len=1000)
    assert 0 <= start <= 1000
    assert start + 100 <= 1000
