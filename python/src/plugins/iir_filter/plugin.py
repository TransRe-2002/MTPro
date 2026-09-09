from __future__ import annotations

from core.algorithms.filters import iir_filter_series
from plugin_api import ParamSpec, PluginInfo, TransformResult


class IIRFilterPlugin:
    pipeline_preview = True

    info = PluginInfo(
        plugin_id="iir_filter",
        display_name="零相位频率滤波",
        version="1.0.0",
        category="auto",
        description=(
            "Butterworth 低/高/带通/带阻或陷波滤波。截止频率以奈奎斯特频率的比例填写，"
            "因此同一配方可用于不同采样率通道。"
        ),
        params_schema=(
            ParamSpec(
                "filter_kind",
                "滤波类型",
                str,
                "bandstop",
                options=("lowpass", "highpass", "bandpass", "bandstop", "notch"),
            ),
            ParamSpec("low_fraction", "低截止 / Nyquist", float, 0.1, minimum=1.0e-9, maximum=0.999999),
            ParamSpec("high_fraction", "高截止 / Nyquist", float, 0.3, minimum=1.0e-9, maximum=0.999999),
            ParamSpec("notch_fraction", "陷波频率 / Nyquist", float, 0.5, minimum=1.0e-9, maximum=0.999999),
            ParamSpec("quality_factor", "陷波 Q", float, 30.0, minimum=0.001),
            ParamSpec("order", "Butterworth 阶数", int, 4, minimum=1, maximum=16),
            ParamSpec("min_segment_points", "最短有效段点数", int, 16, minimum=2),
        ),
    )

    def process(self, block, params, ctx):
        ctx.progress(0.1, "执行零相位频率滤波")
        result = iir_filter_series(
            block.series,
            filter_kind=str(params["filter_kind"]),
            low_fraction=float(params["low_fraction"]),
            high_fraction=float(params["high_fraction"]),
            notch_fraction=float(params["notch_fraction"]),
            quality_factor=float(params["quality_factor"]),
            order=int(params["order"]),
            min_segment_points=int(params["min_segment_points"]),
        )
        ctx.progress(1.0, "频率滤波完成")
        return TransformResult(
            result.cleaned,
            {
                "processed_segments": [list(item) for item in result.processed_segments],
                "skipped_segments": [list(item) for item in result.skipped_segments],
            },
        )
