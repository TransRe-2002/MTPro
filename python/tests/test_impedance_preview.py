from __future__ import annotations

import csv
import io

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTabWidget

from core.impedance_result import propagate_complex_standard_error
from emtf.io.edi import BandedResult
from ui.impedance_preview import (
    ImpedancePreviewWidget,
    ImpedanceResultPool,
    banded_result_csv,
)


def _bands() -> BandedResult:
    periods = np.array([1.0, 10.0])
    z = np.array([1.0 + 2.0j, 3.0 + 4.0j])
    se = np.array([0.1, 0.2])
    return BandedResult(
        periods=periods,
        zxx=z,
        zxy=z * 2,
        zyx=z * 3,
        zyy=z * 4,
        zxx_se=se,
        zxy_se=se,
        zyx_se=se,
        zyy_se=se,
        n_observations=np.array([12, 8]),
        station_name="STA",
    )


def _bands_with_tipper() -> BandedResult:
    bands = _bands()
    bands.tzx = np.array([0.1 + 0.2j, 0.2 + 0.3j])
    bands.tzy = np.array([0.3 + 0.4j, 0.4 + 0.5j])
    bands.tzx_se = np.array([0.01, 0.02])
    bands.tzy_se = np.array([0.02, 0.03])
    return bands


class _UncopyableBands:
    """Minimal stand-in for a PyO3 result that does not implement pickle."""

    def __init__(self, bands: BandedResult):
        self._bands = bands

    def __getattr__(self, name: str):
        return getattr(self._bands, name)

    def __deepcopy__(self, memo):
        raise TypeError("cannot pickle native result")


def test_pool_adds_independent_manual_snapshot_and_tracks_export():
    bands = _bands()
    pool = ImpedanceResultPool()
    snapshot = pool.add_payload({
        "method": "remote reference",
        "bands": bands,
        "station_name": "LOCAL",
        "remote_station_name": "REMOTE",
        "range_start": "2026-01-02 03:04:05",
        "range_end": "2026-01-02 04:05:06",
    })

    bands.zxy[0] = 99 + 99j
    assert snapshot.bands.zxy[0] != bands.zxy[0]
    assert snapshot.name.startswith("LOCAL←REMOTE RR")
    assert "20260102_030405" in snapshot.name
    assert pool.unexported_count() == 1

    pool.mark_exported(snapshot.snapshot_id, "csv")
    assert snapshot.exported


def test_pool_keeps_same_name_snapshots_independent():
    pool = ImpedanceResultPool()
    payload = {
        "method": "robust estimate",
        "bands": _bands(),
        "name": "same station and range",
    }
    first = pool.add_payload(payload)
    second = pool.add_payload(payload)
    assert first.snapshot_id != second.snapshot_id
    assert [item.name for item in pool.items()] == [payload["name"], payload["name"]]
    assert pool.get(first.snapshot_id) is first
    assert pool.get(second.snapshot_id) is second


def test_pool_detaches_uncopyable_native_bands_into_python_result():
    bands = _bands()
    pool = ImpedanceResultPool()

    snapshot = pool.add_payload({
        "method": "robust estimate",
        "bands": _UncopyableBands(bands),
        "station_name": "STA",
    })

    bands.zxy[0] = 99 + 99j
    assert isinstance(snapshot.bands, BandedResult)
    assert snapshot.bands.zxy[0] != bands.zxy[0]
    assert 'DATAID="STA"' in snapshot.bands.format_edi_str()


def test_csv_contains_only_banded_rows_and_complex_components():
    rows = list(csv.DictReader(io.StringIO(banded_result_csv(_bands()))))

    assert len(rows) == 2
    assert float(rows[0]["frequency_hz"]) == 1.0
    assert float(rows[1]["period_s"]) == 10.0
    assert float(rows[0]["zxy_real"]) == 2.0
    assert float(rows[0]["zxy_imag"]) == 4.0
    assert float(rows[0]["n_observations"]) == 12.0


def test_preview_empty_state_and_single_result_layout(qtbot):
    pool = ImpedanceResultPool()
    widget = ImpedancePreviewWidget(pool)
    qtbot.addWidget(widget)

    assert widget.right_stack.currentIndex() == 0
    assert not widget.right_stack.currentWidget().isAncestorOf(widget.list)
    empty_text = {label.text() for label in widget.right_stack.currentWidget().findChildren(type(widget.range_label))}
    assert "暂未保存阻抗数据" in empty_text
    assert '请在“阻抗输出”页面计算并保存一个阻抗结果' in empty_text

    snapshot = pool.add_payload({
        "method": "robust estimate",
        "bands": _bands(),
        "station_name": "STA",
    })
    widget.select_snapshot(snapshot.snapshot_id)

    assert widget.right_stack.currentIndex() == 1
    assert widget.right_stack.currentWidget().isAncestorOf(widget.list)
    assert widget.tabs.count() == 4
    assert widget.tabs.cornerWidget(Qt.Corner.TopRightCorner) is not None
    assert widget.tabs.cornerWidget(Qt.Corner.TopRightCorner).layout().itemAt(0).widget() is widget.reset_view_btn
    assert widget.main_canvas._mtpro_interaction_controller.reset_button.isHidden()
    assert widget.c_response_canvas._mtpro_interaction_controller.reset_button.isHidden()
    assert widget.impedance_splitter.count() == 2
    table_tabs = widget.impedance_table_tabs
    assert isinstance(table_tabs, QTabWidget)
    assert widget.impedance_splitter.widget(1) is table_tabs
    assert [table_tabs.tabText(i) for i in range(table_tabs.count())] == ["阻抗张量", "视电阻率/相位"]
    assert table_tabs.currentWidget() is widget.impedance_table
    tensor_value = widget.impedance_table.item(0, 3).text()
    table_tabs.setCurrentIndex(1)
    assert table_tabs.currentWidget() is widget.rho_table
    assert widget.rho_table.rowCount() == len(snapshot.bands.periods)
    table_tabs.setCurrentIndex(0)
    assert widget.impedance_table.item(0, 3).text() == tensor_value
    assert widget.tipper_splitter.count() == 2
    assert not widget.tabs.isTabVisible(widget._tipper_tab_index)
    assert widget.tabs.indexOf(widget._c_response_page) >= 0
    assert widget.c_response_table.rowCount() == len(snapshot.bands.periods)
    assert widget.list.item(0).isSelected()

    rho_axis, phase_axis = widget.main_figure.axes
    assert widget.main_figure.get_layout_engine() is None
    assert rho_axis.get_title() == "视电阻率"
    assert rho_axis.get_ylabel() == "Apparent Resistivity (Ω·m)"
    assert rho_axis.get_xscale() == "log"
    assert rho_axis.get_yscale() == "log"
    assert phase_axis.get_title() == "相位"
    assert phase_axis.get_ylabel() == "Phase (°)"
    assert phase_axis.get_xscale() == "log"
    rho_xy_line = next(container for container in rho_axis.containers if container.get_label() == r"$\rho_{xy}$").lines[0]
    rho_yx_line = next(container for container in rho_axis.containers if container.get_label() == r"$\rho_{yx}$").lines[0]
    phase_xy_line = next(container for container in phase_axis.containers if container.get_label() == r"$\phi_{xy}$").lines[0]
    assert (rho_xy_line.get_color(), rho_xy_line.get_marker(), rho_xy_line.get_linestyle()) == ("red", "o", "None")
    assert (rho_yx_line.get_color(), rho_yx_line.get_marker(), rho_yx_line.get_linestyle()) == ("blue", "s", "None")
    expected_phase = np.degrees(np.arctan(snapshot.bands.zxy.imag / snapshot.bands.zxy.real))
    np.testing.assert_allclose(phase_xy_line.get_ydata(), expected_phase)
    assert -180.0 <= phase_axis.get_ylim()[0] < phase_axis.get_ylim()[1] <= 180.0


def test_preview_switches_c_response_plot_type(qtbot):
    pool = ImpedanceResultPool()
    widget = ImpedancePreviewWidget(pool)
    qtbot.addWidget(widget)
    snapshot = pool.add_payload({"method": "robust estimate", "bands": _bands()})
    widget.select_snapshot(snapshot.snapshot_id)
    widget.tabs.setCurrentIndex(widget._c_response_tab_index)

    response = 1000 * snapshot.bands.zxy / (1j * 2 * np.pi / snapshot.bands.periods)
    assert widget.c_response_mode.currentData() == "re_im"
    np.testing.assert_allclose(widget.c_response_figure.axes[0].containers[0].lines[0].get_ydata(), response.real)

    widget.c_response_mode.setCurrentIndex(widget.c_response_mode.findData("amp_phase"))
    assert [axis.get_title() for axis in widget.c_response_figure.axes] == [
        "交叉分量幅值", "对角分量幅值", "交叉分量相位", "对角分量相位",
    ]
    np.testing.assert_allclose(widget.c_response_figure.axes[0].containers[0].lines[0].get_ydata(), np.abs(response))
    np.testing.assert_allclose(
        widget.c_response_figure.axes[2].containers[0].lines[0].get_ydata(),
        np.degrees(np.arctan(response.imag / response.real)),
    )
    widget.c_response_mode.setCurrentIndex(widget.c_response_mode.findData("re_im"))
    np.testing.assert_allclose(widget.c_response_figure.axes[2].containers[0].lines[0].get_ydata(), response.imag)
    assert widget.c_response_table.rowCount() == len(response)
    assert widget.c_response_canvas._mtpro_interaction_controller.reset_button.isHidden()


def test_preview_shows_tipper_tab_and_keeps_saved_order(qtbot):
    pool = ImpedanceResultPool()
    widget = ImpedancePreviewWidget(pool)
    qtbot.addWidget(widget)
    first = pool.add_payload({"method": "robust estimate", "bands": _bands(), "name": "first"})
    bands = _bands()
    bands.tzx = np.array([0.1 + 0.2j, 0.2 + 0.3j])
    bands.tzy = np.array([0.3 + 0.4j, 0.4 + 0.5j])
    bands.tzx_se = np.array([0.01, 0.01])
    bands.tzy_se = np.array([0.02, 0.02])
    second = pool.add_payload({"method": "remote reference", "bands": bands, "name": "second"})

    widget.select_snapshot(second.snapshot_id)

    assert [widget.list.item(i).text() for i in range(widget.list.count())] == ["first", "second"]
    assert not widget.list.item(0).isSelected()
    assert widget.list.item(1).isSelected()
    assert widget.tabs.isTabVisible(widget._tipper_tab_index)
    assert pool.get(first.snapshot_id) is first

    real_axis, imaginary_axis = widget.tipper_figure.axes
    assert widget.tipper_figure.get_layout_engine() is None
    assert real_axis.get_title() == "倾子响应实部"
    assert real_axis.get_ylabel() == "Real"
    assert imaginary_axis.get_title() == "倾子响应虚部"
    assert imaginary_axis.get_ylabel() == "Imaginary"
    assert imaginary_axis.get_xlabel() == "Period (s)"
    real_tzx_line = next(container for container in real_axis.containers if container.get_label() == r"$T_{zx}$").lines[0]
    real_tzy_line = next(container for container in real_axis.containers if container.get_label() == r"$T_{zy}$").lines[0]
    assert (real_tzx_line.get_color(), real_tzx_line.get_marker(), real_tzx_line.get_linestyle()) == ("red", "o", "None")
    assert (real_tzy_line.get_color(), real_tzy_line.get_marker(), real_tzy_line.get_linestyle()) == ("blue", "s", "None")

    widget.tipper_mode.setCurrentIndex(widget.tipper_mode.findData("amp_phase"))
    amplitude_axis, tipper_phase_axis = widget.tipper_figure.axes
    assert amplitude_axis.get_title() == "倾子响应幅值"
    assert amplitude_axis.get_ylabel() == "Amplitude"
    assert tipper_phase_axis.get_title() == "倾子响应相位"
    assert tipper_phase_axis.get_ylabel() == "Phase (°)"
    phase_tzx_line = next(container for container in tipper_phase_axis.containers if container.get_label() == r"$T_{zx}$").lines[0]
    expected_phase = np.degrees(np.arctan(second.bands.tzx.imag / second.bands.tzx.real))
    np.testing.assert_allclose(phase_tzx_line.get_ydata(), expected_phase)


def test_preview_select_reveals_filtered_snapshot_and_inline_rename(qtbot):
    pool = ImpedanceResultPool()
    widget = ImpedancePreviewWidget(pool)
    qtbot.addWidget(widget)
    widget.method_filter.setCurrentIndex(widget.method_filter.findData("remote reference"))
    snapshot = pool.add_payload({
        "method": "robust estimate",
        "bands": _bands(),
        "name": "before",
    })

    widget.select_snapshot(snapshot.snapshot_id)
    assert widget.method_filter.currentData() == "all"
    assert widget.list.item(0).isSelected()

    widget.list.item(0).setText("after")
    assert snapshot.name == "after"


def test_comparison_window_updates_only_after_explicit_click(qtbot):
    pool = ImpedanceResultPool()
    widget = ImpedancePreviewWidget(pool)
    qtbot.addWidget(widget)
    results = [
        pool.add_payload({
            "method": "robust estimate",
            "bands": _bands(),
            "name": f"result {index}",
        })
        for index in range(3)
    ]

    assert widget.list.selectionMode() == widget.list.SelectionMode.SingleSelection
    widget.list.item(0).setCheckState(Qt.CheckState.Checked)
    widget.list.item(1).setCheckState(Qt.CheckState.Checked)
    assert widget.compare_btn.text() == "对比所选 (2)"
    assert widget.compare_btn.isEnabled()

    widget.compare_btn.click()
    window = widget._comparison_window
    assert window is not None
    assert window.isWindow()
    assert window.result_ids == (results[0].snapshot_id, results[1].snapshot_id)
    assert window.export_button.isEnabled()

    widget.list.item(2).setCheckState(Qt.CheckState.Checked)
    assert widget.compare_btn.text() == "对比所选 (3)"
    assert window.result_ids == (results[0].snapshot_id, results[1].snapshot_id)

    widget.compare_btn.click()
    assert window.result_ids == tuple(result.snapshot_id for result in results)


def test_comparison_uses_four_shared_axes_result_colors_and_layered_errors(qtbot):
    pool = ImpedanceResultPool()
    widget = ImpedancePreviewWidget(pool)
    qtbot.addWidget(widget)
    first = pool.add_payload({
        "method": "robust estimate",
        "bands": _bands(),
        "name": "first result",
    })
    second = pool.add_payload({
        "method": "remote reference",
        "bands": _bands(),
        "name": "second result",
    })
    widget.list.item(0).setCheckState(Qt.CheckState.Checked)
    widget.list.item(1).setCheckState(Qt.CheckState.Checked)
    widget.compare_btn.click()

    window = widget._comparison_window
    assert window is not None
    axes = window.figure.axes
    assert len(axes) == 4
    assert [axis.get_title() for axis in axes] == [
        r"$\rho_{xy}$",
        r"$\rho_{yx}$",
        r"$\phi_{xy}$",
        r"$\phi_{yx}$",
    ]
    assert axes[0].get_shared_x_axes().joined(axes[0], axes[3])
    assert axes[0].get_shared_y_axes().joined(axes[0], axes[1])
    assert axes[2].get_shared_y_axes().joined(axes[2], axes[3])

    error_collections = [
        collection for collection in axes[0].collections
        if collection.get_zorder() == 1
    ]
    point_collections = [
        collection for collection in axes[0].collections
        if collection.get_zorder() == 3
    ]
    assert len(error_collections) == 2
    assert len(point_collections) == 2
    assert all(collection.get_alpha() == 0.6 for collection in error_collections)
    assert all(collection.get_alpha() == 1.0 for collection in point_collections)

    for result_index in range(2):
        colors = [
            axis.collections[result_index * 2 + 1].get_facecolors()[0, :3]
            for axis in axes
        ]
        for color in colors[1:]:
            np.testing.assert_allclose(color, colors[0])
    first_color = axes[0].collections[1].get_facecolors()[0, :3]
    second_color = axes[0].collections[3].get_facecolors()[0, :3]
    assert not np.allclose(first_color, second_color)

    assert len(window.figure.legends) == 1
    legend = window.figure.legends[0]
    assert legend._ncols == 1
    assert window.figure.subplotpars.right == 0.82
    assert legend._loc == 2
    assert [text.get_text() for text in legend.get_texts()] == [
        first.name,
        second.name,
    ]


def test_comparison_disambiguates_duplicate_names_with_backend(qtbot):
    pool = ImpedanceResultPool()
    widget = ImpedancePreviewWidget(pool)
    qtbot.addWidget(widget)
    for backend in ("emtf_pywrap", "emtf_rs"):
        pool.add_payload({
            "method": "robust estimate",
            "bands": _bands(),
            "name": "same result",
            "parameters": {"backend": backend},
        })
    widget.list.item(0).setCheckState(Qt.CheckState.Checked)
    widget.list.item(1).setCheckState(Qt.CheckState.Checked)
    widget.compare_btn.click()

    legend = widget._comparison_window.figure.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == [
        "same result · EMTF pywrap",
        "same result · emtf-rs",
    ]


def test_comparison_wraps_long_legend_names_and_spaces_entries(qtbot):
    pool = ImpedanceResultPool()
    widget = ImpedancePreviewWidget(pool)
    qtbot.addWidget(widget)
    pool.add_payload({
        "method": "robust estimate",
        "bands": _bands(),
        "name": "这是一个非常非常长的阻抗对比结果名称",
    })
    pool.add_payload({
        "method": "remote reference",
        "bands": _bands(),
        "name": "second result",
    })
    for index in range(widget.list.count()):
        widget.list.item(index).setCheckState(Qt.CheckState.Checked)
    widget.compare_btn.click()

    legend = widget._comparison_window.impedance_figure.legends[0]
    assert "\n" in legend.get_texts()[0].get_text()
    assert legend.labelspacing == 1.2


def test_tipper_uncertainty_propagates_equal_component_errors():
    amplitude_error, phase_error = propagate_complex_standard_error(
        np.array([3.0 + 4.0j, 0.0 + 0.0j]),
        np.array([0.5, 0.5]),
    )

    np.testing.assert_allclose(amplitude_error[0], 0.5)
    np.testing.assert_allclose(phase_error[0], np.degrees(0.1))
    assert np.isnan(amplitude_error[1])
    assert np.isnan(phase_error[1])


def test_comparison_tipper_tab_ignores_results_without_tipper_and_inherits_mode(qtbot):
    pool = ImpedanceResultPool()
    widget = ImpedancePreviewWidget(pool)
    qtbot.addWidget(widget)
    without_tipper = pool.add_payload({
        "method": "robust estimate",
        "bands": _bands(),
        "name": "without tipper",
    })
    first = pool.add_payload({
        "method": "robust estimate",
        "bands": _bands_with_tipper(),
        "name": "first tipper",
    })
    second = pool.add_payload({
        "method": "remote reference",
        "bands": _bands_with_tipper(),
        "name": "second tipper",
    })
    widget.tipper_mode.setCurrentIndex(widget.tipper_mode.findData("amp_phase"))
    for index in range(widget.list.count()):
        widget.list.item(index).setCheckState(Qt.CheckState.Checked)
    widget.compare_btn.click()

    window = widget._comparison_window
    assert window is not None
    assert window.result_ids == (
        without_tipper.snapshot_id,
        first.snapshot_id,
        second.snapshot_id,
    )
    assert window.tipper_result_ids == (first.snapshot_id, second.snapshot_id)
    assert window.tabs.isTabVisible(window.tipper_tab_index)
    assert window.tipper_mode_combo.currentData() == "amp_phase"

    window.tabs.setCurrentIndex(window.tipper_tab_index)
    assert window.tipper_mode_combo.isVisibleTo(window)
    assert [axis.get_title() for axis in window.tipper_figure.axes] == [
        r"$|T_{zx}|$",
        r"$|T_{zy}|$",
        r"$\arg(T_{zx})$",
        r"$\arg(T_{zy})$",
    ]

    window.tipper_mode_combo.setCurrentIndex(
        window.tipper_mode_combo.findData("re_im")
    )
    axes = window.tipper_figure.axes
    assert [axis.get_title() for axis in axes] == [
        r"$\mathrm{Re}(T_{zx})$",
        r"$\mathrm{Re}(T_{zy})$",
        r"$\mathrm{Im}(T_{zx})$",
        r"$\mathrm{Im}(T_{zy})$",
    ]
    assert axes[0].get_shared_x_axes().joined(axes[0], axes[3])
    assert axes[0].get_shared_y_axes().joined(axes[0], axes[1])
    assert axes[2].get_shared_y_axes().joined(axes[2], axes[3])
    assert len(window.tipper_figure.legends) == 1

    for axis in axes:
        errors = [item for item in axis.collections if item.get_zorder() == 1]
        points = [item for item in axis.collections if item.get_zorder() == 3]
        assert len(errors) == 2
        assert len(points) == 2
        assert all(item.get_alpha() == 0.6 for item in errors)
        assert all(item.get_alpha() == 1.0 for item in points)
    for result_index in range(2):
        colors = [
            axis.collections[result_index * 2 + 1].get_facecolors()[0, :3]
            for axis in axes
        ]
        for color in colors[1:]:
            np.testing.assert_allclose(color, colors[0])


def test_comparison_hides_tipper_tab_when_fewer_than_two_results_have_tipper(qtbot):
    pool = ImpedanceResultPool()
    widget = ImpedancePreviewWidget(pool)
    qtbot.addWidget(widget)
    pool.add_payload({
        "method": "robust estimate",
        "bands": _bands_with_tipper(),
        "name": "only tipper",
    })
    pool.add_payload({
        "method": "remote reference",
        "bands": _bands(),
        "name": "no tipper",
    })
    for index in range(widget.list.count()):
        widget.list.item(index).setCheckState(Qt.CheckState.Checked)
    widget.compare_btn.click()

    window = widget._comparison_window
    assert window is not None
    assert len(window.tipper_result_ids) == 1
    assert not window.tabs.isTabVisible(window.tipper_tab_index)
    assert not window.tipper_mode_combo.isVisibleTo(window)
