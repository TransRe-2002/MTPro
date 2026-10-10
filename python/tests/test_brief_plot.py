"""BriefPlot 堆叠通道标签的布局测试。"""

import colorsys
import random

import pytest

from ui.widgets.brief_plot import BriefPlot, STACK_LABEL_BASELINE_OFFSET


def test_stacked_channel_labels_are_near_their_baselines(qtbot):
    plot = BriefPlot()
    qtbot.addWidget(plot)
    offsets = {"Ex": 2.0, "Hy": 0.0}

    plot.set_stacked_layout(["Ex", "Hy"], offsets, scale_height=1.0)

    assert plot._stack_labels["Ex"].pos().y() == pytest.approx(
        offsets["Ex"] + STACK_LABEL_BASELINE_OFFSET,
    )
    assert plot._stack_labels["Hy"].pos().y() == pytest.approx(
        offsets["Hy"] + STACK_LABEL_BASELINE_OFFSET,
    )


def test_random_curve_colors_keep_adjacent_curves_visually_distinct(qtbot):
    plot = BriefPlot()
    qtbot.addWidget(plot)
    plot._color_allocator.rng = random.Random(7)
    labels = ["Ex1", "Ex2", "Ey1", "Ey2", "Bx", "By", "Bz"]

    colors = [plot._random_distinct_curve_color(label) for label in labels]
    hues = [colorsys.rgb_to_hls(*(component / 255 for component in color))[0] for color in colors]

    assert len(set(colors)) == len(colors)
    for first, second in zip(hues, hues[1:]):
        assert min(abs(first - second), 1.0 - abs(first - second)) >= 0.1
    assert plot._random_distinct_curve_color("Ex1") == colors[0]
