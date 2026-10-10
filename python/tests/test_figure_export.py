"""Synthetic export regressions; no external data or interactive Qt loop needed."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.image import imread

from ui.dialogs.figure_export import save_matplotlib_figure


def _figure():
    figure = Figure(figsize=(12, 8))
    canvas = FigureCanvasAgg(figure)
    rho, phase = figure.subplots(2, 1)
    rho.errorbar([1, 10, 100], [1, 10, 20], yerr=[0.1, 1, 40],
                 fmt="o", color="red", capsize=3)
    rho.set(xscale="log", yscale="log", xlabel="Period (s)",
            ylabel="Apparent Resistivity", title="Resistivity")
    phase.errorbar([1, 10, 100], [40, 50, 60], yerr=[2, 5, 200],
                   fmt="s", color="blue", capsize=3)
    phase.set(xscale="log", ylim=(0, 90), xlabel="Period (s)",
              ylabel="Phase (degrees)", title="Phase")
    figure.tight_layout()
    figure.set_layout_engine(None)
    return figure, canvas


def _state(figure):
    return {
        "size": tuple(figure.get_size_inches()),
        "dpi": figure.dpi,
        "layout": figure.get_layout_engine(),
        "subplot": tuple(getattr(figure.subplotpars, key) for key in
                         ("left", "right", "bottom", "top", "wspace", "hspace")),
        "axes": [(tuple(axis.get_position().bounds),
                  tuple(axis.get_position(original=True).bounds),
                  axis.get_in_layout(), axis.get_xlim(), axis.get_ylim())
                 for axis in figure.axes],
    }


def test_export_relayouts_at_target_size_and_restores_view(tmp_path, monkeypatch):
    figure, canvas = _figure()
    before = _state(figure)
    bounds = []
    original_savefig = figure.savefig

    def capture(*args, **kwargs):
        original_savefig(*args, **kwargs)
        canvas.draw()
        bounds.extend(axis.get_tightbbox(canvas.get_renderer()).frozen()
                      for axis in figure.axes)

    monkeypatch.setattr(figure, "savefig", capture)
    path = tmp_path / "response.png"
    with rc_context({"savefig.bbox": "tight"}):
        save_matplotlib_figure(figure, canvas, path, width_px=1280,
                               height_px=720, dpi=200, fmt="png")

    assert imread(path).shape[:2] == (720, 1280)
    assert all(box.x0 >= 0 and box.y0 >= 0 and box.x1 <= 1280 and box.y1 <= 720
               for box in bounds)
    assert bounds[1].y1 < bounds[0].y0
    assert _state(figure) == before


@pytest.mark.parametrize("zoomed", [False, True])
def test_svg_bounds_errorbars_without_changing_data(tmp_path, zoomed):
    figure, canvas = _figure()
    figure.axes[0].errorbar([1, 100], [2, 20], xerr=[2, 200],
                            fmt="o", color="green", capsize=3)
    if zoomed:
        # Include fully offscreen points and inverted axes after user zoom/pan.
        for axis in figure.axes:
            axis.set_xlim(80, 5)
        figure.axes[0].set_ylim(30, 0.5)
    before = _state(figure)
    collections = [collection for axis in figure.axes for container in axis.containers
                   for collection in container.lines[2]]
    segments = [collection.get_segments() for collection in collections]
    cap_lines = [cap for axis in figure.axes for container in axis.containers
                 for cap in container.lines[1]]
    cap_data = [(np.array(cap.get_xdata()), np.array(cap.get_ydata())) for cap in cap_lines]
    path = tmp_path / "response.svg"
    save_matplotlib_figure(figure, canvas, path, width_px=1280,
                           height_px=720, dpi=200, fmt="svg")

    root = ET.parse(path).getroot()
    ns = {"s": "http://www.w3.org/2000/svg"}
    clips = {}
    for clip in root.findall(".//s:clipPath", ns):
        rect = clip.find("s:rect", ns)
        if rect is not None:
            x, y, width, height = (float(rect.get(key)) for key in ("x", "y", "width", "height"))
            clips[clip.get("id")] = (x, y, x + width, y + height)
    checked = 0
    for group in root.findall(".//s:g", ns):
        if not group.get("id", "").startswith("LineCollection_"):
            continue
        for path in group.findall("s:path", ns):
            clip = path.get("clip-path")
            if clip is None:
                continue
            x0, y0, x1, y1 = clips[clip[5:-1]]
            values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", path.get("d", ""))]
            for x, y in zip(values[::2], values[1::2]):
                assert x0 - 1e-5 <= x <= x1 + 1e-5
                assert y0 - 1e-5 <= y <= y1 + 1e-5
                checked += 1
    assert checked >= (4 if zoomed else 12)
    for collection, original in zip(collections, segments):
        for actual, expected in zip(collection.get_segments(), original):
            np.testing.assert_array_equal(actual, expected)
    for cap, (x, y) in zip(cap_lines, cap_data):
        np.testing.assert_array_equal(cap.get_xdata(), x)
        np.testing.assert_array_equal(cap.get_ydata(), y)
    assert _state(figure) == before


@pytest.mark.parametrize("fmt", ["png", "svg", "pdf"])
def test_export_restores_figure_when_save_fails(tmp_path, monkeypatch, fmt):
    figure, canvas = _figure()
    before = _state(figure)
    collection = figure.axes[0].containers[0].lines[2][0]
    segments = collection.get_segments()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(figure, "savefig", fail)
    with pytest.raises(OSError, match="disk full"):
        save_matplotlib_figure(figure, canvas, tmp_path / f"response.{fmt}",
                               width_px=1280, height_px=720, dpi=200, fmt=fmt)
    assert _state(figure) == before
    for actual, expected in zip(collection.get_segments(), segments):
        np.testing.assert_array_equal(actual, expected)
