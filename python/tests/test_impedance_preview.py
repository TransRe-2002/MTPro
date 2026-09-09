from __future__ import annotations

import csv
import io

import numpy as np
from PySide6.QtCore import Qt

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
    assert widget.tabs.count() == 3
    assert widget.tabs.cornerWidget(Qt.Corner.TopRightCorner) is not None
    assert widget.impedance_splitter.count() == 2
    assert widget.tipper_splitter.count() == 2
    assert not widget.tabs.isTabVisible(widget._tipper_tab_index)
    assert widget.list.item(0).isSelected()

    rho_axis, phase_axis = widget.main_figure.axes
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
    assert real_axis.get_title() == "倾子响应实部"
    assert real_axis.get_ylabel() == "Real"
    assert imaginary_axis.get_title() == "倾子响应虚部"
    assert imaginary_axis.get_ylabel() == "Imaginary"
    assert imaginary_axis.get_xlabel() == "Period (s)"
    real_tzx_line = next(container for container in real_axis.containers if container.get_label() == r"$\mathrm{Re}(T_{zx})$").lines[0]
    real_tzy_line = next(container for container in real_axis.containers if container.get_label() == r"$\mathrm{Re}(T_{zy})$").lines[0]
    assert (real_tzx_line.get_color(), real_tzx_line.get_marker(), real_tzx_line.get_linestyle()) == ("purple", "o", "None")
    assert (real_tzy_line.get_color(), real_tzy_line.get_marker(), real_tzy_line.get_linestyle()) == ("green", "s", "None")

    widget.tipper_mode.setCurrentIndex(widget.tipper_mode.findData("amp_phase"))
    amplitude_axis, tipper_phase_axis = widget.tipper_figure.axes
    assert amplitude_axis.get_title() == "倾子响应幅值"
    assert amplitude_axis.get_ylabel() == "Amplitude"
    assert tipper_phase_axis.get_title() == "倾子响应相位"
    assert tipper_phase_axis.get_ylabel() == "Phase (°)"
    phase_tzx_line = next(container for container in tipper_phase_axis.containers if container.get_label() == r"$\arg(T_{zx})$").lines[0]
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
    assert [text.get_text() for text in legend.get_texts()] == [
        first.name,
        second.name,
    ]

