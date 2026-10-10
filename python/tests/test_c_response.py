from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pytest

from emtf.io.edi import BandedResult
from output_tools.remote_reference import RemoteReferenceWidget
from output_tools.robust_estimate import RobustEstimateWidget
from output_tools.shared.c_response import c_response_components
from ui.impedance_plot import draw_c_response


def test_c_response_uses_emtf_units_and_standard_complex_factor():
    periods = np.array([2.0, 0.0])
    z = np.array([1.0 + 1.0j, 2.0 + 3.0j])
    components = c_response_components(periods, z, z * 2, z * 3, z * 4)

    np.testing.assert_allclose(components["cxx"][0], 1000.0 * (1.0 - 1.0j) / np.pi)
    assert np.isnan(components["cxx"][1].real)
    np.testing.assert_allclose(components["cxy"], components["cxx"] * 2)
    np.testing.assert_allclose(components["cyx"], components["cxx"] * 3)
    np.testing.assert_allclose(components["cyy"], components["cxx"] * 4)


def test_c_response_plot_groups_cross_and_diagonal_components(qtbot):
    class Result:
        periods = np.array([1.0, 10.0])
        zxx = np.array([1.0 + 1.0j, 2.0 + 2.0j])
        zxy = np.array([2.0 + 1.0j, 3.0 + 2.0j])
        zyx = np.array([4.0 + 2.0j, 6.0 + 4.0j])
        zyy = np.array([5.0 + 1.0j, 7.0 + 2.0j])

    figure = Figure()
    canvas = FigureCanvas(figure)
    qtbot.addWidget(canvas)
    draw_c_response(figure, canvas, Result(), mode="amp_phase")

    assert len(figure.axes) == 4
    assert [axis.get_title() for axis in figure.axes] == [
        "交叉分量幅值", "对角分量幅值", "交叉分量相位", "对角分量相位",
    ]
    assert [container.get_label() for container in figure.axes[0].containers] == [r"$C_{xy}$", r"$C_{yx}$"]
    assert [container.get_label() for container in figure.axes[1].containers] == [r"$C_{xx}$", r"$C_{yy}$"]
    assert [container.lines[0].get_color() for container in figure.axes[0].containers] == ["red", "blue"]
    assert all(container.lines[0].get_linestyle() == "None" for axis in figure.axes for container in axis.containers)

    draw_c_response(figure, canvas, Result(), mode="re_im")
    assert [axis.get_title() for axis in figure.axes] == [
        "交叉分量实部", "对角分量实部", "交叉分量虚部", "对角分量虚部",
    ]


@pytest.mark.parametrize("widget_class", [RobustEstimateWidget, RemoteReferenceWidget])
def test_estimator_windows_expose_and_fill_c_response_tabs(qtbot, widget_class):
    widget = widget_class()
    qtbot.addWidget(widget)
    assert widget.result_tabs.indexOf(widget.c_response_table) >= 0
    assert widget.result_tabs.indexOf(widget.c_response_canvas) >= 0

    z = np.array([1.0 + 2.0j, 3.0 + 4.0j])
    se = np.array([0.1, 0.2])
    widget._bands = BandedResult(
        periods=np.array([1.0, 10.0]),
        zxx=z,
        zxy=z * 2,
        zyx=z * 3,
        zyy=z * 4,
        zxx_se=se,
        zxy_se=se,
        zyx_se=se,
        zyy_se=se,
    )
    widget._display_c_response_result()
    assert widget.c_response_table.rowCount() == 2
    assert widget.c_response_table.item(0, 2).text().endswith("j")
    assert len(widget.c_response_fig.axes) == 4
