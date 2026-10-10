from __future__ import annotations

import numpy as np

from core.impedance_stitch import stitch_banded_results
from core.impedance_result import ImpedanceResult
from emtf.io.edi import BandedResult
from ui.impedance_stitch import ImpedanceStitchWidget


def _bands(periods, offset: float, *, tipper: bool = True) -> BandedResult:
    periods = np.asarray(periods, dtype=float)
    values = offset + np.arange(len(periods), dtype=float) * (1.0 + 1.0j)
    bands = BandedResult(
        periods=periods,
        zxx=values,
        zxy=values + 10,
        zyx=values + 20,
        zyy=values + 30,
        zxx_se=np.full(len(periods), offset + 1),
        zxy_se=np.full(len(periods), offset + 2),
        zyx_se=np.full(len(periods), offset + 3),
        zyy_se=np.full(len(periods), offset + 4),
        n_observations=np.full(len(periods), int(offset), dtype=np.int32),
        station_name="STA",
    )
    if tipper:
        bands.tzx = values / 100
        bands.tzy = values / 50
        bands.tzx_se = np.full(len(periods), 0.01)
        bands.tzy_se = np.full(len(periods), 0.02)
    return bands


def _snapshot(name: str, periods, offset: float) -> ImpedanceResult:
    return ImpedanceResult.from_payload({
        "method": "robust estimate",
        "name": name,
        "station_name": "STA",
        "bands": _bands(periods, offset),
    })


def test_stitch_copies_each_side_and_assigns_exact_boundary_to_long_source():
    short = _bands([1, 10, 20], 100)
    long = _bands([10, 30, 100], 200)

    stitched = stitch_banded_results(short, long, 10)

    np.testing.assert_array_equal(stitched.periods, [1, 10, 30, 100])
    np.testing.assert_array_equal(stitched.zxy, [short.zxy[0], long.zxy[0], long.zxy[1], long.zxy[2]])
    np.testing.assert_array_equal(stitched.zxy_se, [short.zxy_se[0], long.zxy_se[0], long.zxy_se[1], long.zxy_se[2]])
    np.testing.assert_array_equal(stitched.n_observations, [100, 200, 200, 200])
    np.testing.assert_array_equal(stitched.tzx, [short.tzx[0], long.tzx[0], long.tzx[1], long.tzx[2]])


def test_stitch_rejects_boundary_without_points_from_both_sources():
    short = _bands([1, 2], 100)
    long = _bands([10, 20], 200)

    try:
        stitch_banded_results(short, long, 30)
    except ValueError as error:
        assert "两侧都必须有数据点" in str(error)
    else:
        raise AssertionError("expected a boundary validation error")


def test_stitch_widget_plots_only_the_selected_side_and_changes_view(qtbot):
    short = _snapshot("short", [1, 10, 20], 100)
    long = _snapshot("long", [10, 30, 100], 200)
    widget = ImpedanceStitchWidget([short, long])
    qtbot.addWidget(widget)

    widget.boundary_spin.setValue(10)
    first_axis = widget.figure.axes[0]
    assert len(widget.figure.axes) == 2
    assert first_axis.get_title() == "视电阻率"
    assert widget.figure.axes[1].get_title() == "相位"
    points = [container.lines[0] for container in first_axis.containers]
    np.testing.assert_array_equal(points[0].get_xdata(), [1])
    np.testing.assert_array_equal(points[1].get_xdata(), [1])
    np.testing.assert_array_equal(points[2].get_xdata(), [10, 30, 100])
    np.testing.assert_array_equal(points[3].get_xdata(), [10, 30, 100])
    assert all(line.get_linestyle() == "None" for line in points)
    assert [text.get_text() for text in first_axis.get_legend().get_texts()] == [
        r"$\rho_{xy}$", r"$\rho_{yx}$",
    ]
    assert [line.get_color() for line in points] == ["red", "blue", "red", "blue"]
    assert widget.create_button.isEnabled()

    widget.view_combo.setCurrentIndex(widget.view_combo.findData("tipper"))
    assert len(widget.figure.axes) == 2
    assert widget.figure.axes[0].get_title() == "倾子实部"
    assert widget.figure.axes[1].get_title() == "倾子虚部"

    widget.view_combo.setCurrentIndex(widget.view_combo.findData("c_response"))
    assert len(widget.figure.axes) == 4
    assert widget.figure.axes[0].get_title() == "交叉分量实部"
    assert widget.figure.axes[1].get_title() == "对角分量实部"
    assert widget.figure.axes[2].get_title() == "交叉分量虚部"
    assert widget.figure.axes[3].get_title() == "对角分量虚部"
    assert all(
        container.lines[0].get_linestyle() == "None"
        for axis in widget.figure.axes
        for container in axis.containers
    )
    assert all(
        [text.get_text() for text in axis.get_legend().get_texts()]
        == ([r"$C_{xy}$", r"$C_{yx}$"] if index in (0, 2) else [r"$C_{xx}$", r"$C_{yy}$"])
        for index, axis in enumerate(widget.figure.axes)
    )


def test_stitch_widget_emits_a_new_snapshot_payload(qtbot):
    short = _snapshot("short", [1, 10, 20], 100)
    long = _snapshot("long", [10, 30, 100], 200)
    widget = ImpedanceStitchWidget([short, long])
    qtbot.addWidget(widget)
    payloads = []
    widget.result_created.connect(payloads.append)

    widget.boundary_spin.setValue(10)
    widget._create_result()

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["method"] == "stitched impedance"
    assert payload["parameters"]["stitch_boundary_period_s"] == 10
    np.testing.assert_array_equal(payload["bands"].periods, [1, 10, 30, 100])
