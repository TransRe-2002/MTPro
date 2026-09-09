from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np


def uid() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class PipelineBlock:
    block_id: str = field(default_factory=uid)
    tool_id: str = ""
    start: int = 0
    end: int = 0
    params: dict = field(default_factory=dict)
    status: str = "empty"
    # block 只缓存自己区间 [start:end] 的输入/输出切片，而不是整条通道，
    # 避免每个 block 都复制一份全量数据（内存随 block 数线性翻倍）。
    input_slice: np.ndarray | None = field(default=None, repr=False)
    output_slice: np.ndarray | None = field(default=None, repr=False)
    last_summary: str = ""
    # Non-empty when this block shares its reusable settings with peer blocks
    # on other Pipeline channels. Execution caches always remain per block.
    settings_group_id: str = ""


@dataclass
class PipelineSettingsGroup:
    group_id: str = field(default_factory=uid)
    display_label: str = ""
    tool_id: str = ""
    shared_params: dict = field(default_factory=dict)
    reference_block_id: str = ""
    source_sample_interval_seconds: float = 0.0
    revision: int = 0


@dataclass
class PipelineLayer:
    layer_id: str = field(default_factory=uid)
    label: str = "step 1"
    blocks: list[PipelineBlock] = field(default_factory=list)

    def sorted_blocks(self) -> list[PipelineBlock]:
        # 同 (start, end) 的块保持插入顺序（上游在先）：sort 稳定，
        # 只按区间排序即可。此前第三键 block_id 是随机串，
        # 会让同区间块的顺序每次运行都可能颠倒。
        return sorted(self.blocks, key=lambda block: (block.start, block.end))


@dataclass
class ChannelPipeline:
    data_id: int
    channel: str
    expanded: bool = True
    layers: list[PipelineLayer] = field(default_factory=list)
    baseline_snapshot: np.ndarray | None = field(default=None, repr=False)

    @property
    def ref(self) -> tuple[int, str]:
        return self.data_id, self.channel

    def add_layer(self) -> PipelineLayer:
        layer = PipelineLayer(label=f"step {len(self.layers) + 1}")
        self.layers.append(layer)
        return layer

    def find_block(self, block_id: str) -> tuple[PipelineLayer, PipelineBlock] | None:
        for layer in self.layers:
            for block in layer.blocks:
                if block.block_id == block_id:
                    return layer, block
        return None

    def remove_block(self, block_id: str) -> bool:
        for layer in self.layers:
            before = len(layer.blocks)
            layer.blocks = [block for block in layer.blocks if block.block_id != block_id]
            if len(layer.blocks) != before:
                return True
        return False


class SnapEngine:
    THRESHOLD_PX = 8

    def __init__(self):
        self.grid_pts = 500
        self.enabled = True
        self._edges: list[int] = []

    def update_edges(self, pipelines: list[ChannelPipeline], exclude_id: str = ""):
        edges = set()
        for pipeline in pipelines:
            for layer in pipeline.layers:
                for block in layer.blocks:
                    if block.block_id == exclude_id:
                        continue
                    edges.add(block.start)
                    edges.add(block.end)
        self._edges = sorted(edges)

    def snap(self, pts: int, px_per_pt: float, data_len: int) -> tuple[int, list[int]]:
        pts = max(0, min(data_len, pts))
        if not self.enabled:
            return pts, []

        grid = round(pts / self.grid_pts) * self.grid_pts
        best = max(0, min(data_len, grid))
        best_dist = abs(best - pts) * px_per_pt
        guides: list[int] = []

        for edge in self._edges:
            dist = abs(edge - pts) * px_per_pt
            if dist < best_dist:
                best_dist = dist
                best = edge

        if best_dist < self.THRESHOLD_PX:
            guides = [best]
        return max(0, min(data_len, best)), guides

    def snap_interval(self, start: int, end: int, px_per_pt: float, data_len: int) -> tuple[int, list[int]]:
        span = max(0, end - start)
        snapped_start, start_guides = self.snap(start, px_per_pt, data_len)
        snapped_end, end_guides = self.snap(end, px_per_pt, data_len)
        candidates = [
            (snapped_start, start_guides),
            (max(0, min(data_len - span, snapped_end - span)), end_guides),
        ]
        best_start, best_guides = min(candidates, key=lambda item: abs(item[0] - start))
        return best_start, best_guides
