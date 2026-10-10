"""Shared Matplotlib rendering for impedance and tipper results."""

from __future__ import annotations
import unicodedata

import numpy as np
from matplotlib.lines import Line2D

from core.impedance_result import propagate_complex_standard_error
from emtf.io.zfile import emtf_phase_deg
from output_tools.shared.emtf import phase_value_axis_limits
from output_tools.shared.c_response import c_response_from_bands
from ui.theme import configure_matplotlib_canvas


def _as_results(results: object | list[object] | tuple[object, ...] | None) -> list[object]:
    if results is None:
        return []
    if isinstance(results, (list, tuple)):
        return [result for result in results if result is not None]
    return [results]


def _clear_figure(figure) -> None:
    for axis in figure.axes:
        axis.set_xscale("linear")
        axis.set_yscale("linear")
    figure.clear()


def _draw_response_figure(figure, canvas, widget=None) -> None:
    """Lay out once, then disable per-frame automatic layout during interaction."""
    if figure.axes:
        figure.tight_layout()
    figure.set_layout_engine(None)
    configure_matplotlib_canvas(canvas, figure, widget)
    canvas.draw()


def _adaptive_legend(axis, canvas) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if not handles:
        return

    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)

    count = len(unique)
    width = max(int(canvas.width()), 320)
    max_columns = 1 if width < 720 else 2 if width < 1100 else 3
    columns = min(count, max_columns)
    fontsize = 7 if count <= 4 else 6
    axis.legend(
        unique.values(),
        unique.keys(),
        loc="best",
        ncols=columns,
        fontsize=fontsize,
        framealpha=0.4,
        edgecolor="none",
    )


def _refresh_adaptive_legends(figure, canvas) -> None:
    for axis in figure.axes:
        _adaptive_legend(axis, canvas)


def _install_legend_resize_handler(figure, canvas) -> None:
    if getattr(canvas, "_mtpro_legend_resize_cid", None) is not None:
        return

    def refresh_on_resize(_event) -> None:
        _refresh_adaptive_legends(figure, canvas)
        canvas.draw_idle()

    canvas._mtpro_legend_resize_cid = canvas.mpl_connect("resize_event", refresh_on_resize)


def draw_impedance_response(figure, canvas, results, widget=None) -> None:
    """Draw impedance results using the one canonical application style."""
    _clear_figure(figure)
    resolved_results = _as_results(results)
    if not resolved_results:
        _draw_response_figure(figure, canvas, widget)
        return
    rho_axis, phase_axis = figure.subplots(2, 1)
    colour_pairs = (("red", "blue"), ("darkorange", "teal"), ("purple", "green"))
    phase_values: list[np.ndarray] = []
    plotted = False

    for index, result in enumerate(resolved_results):
        periods = np.asarray(getattr(result, "periods", []), dtype=float)
        rho_xy = np.asarray(getattr(result, "rho_xy", []), dtype=float)
        rho_yx = np.asarray(getattr(result, "rho_yx", []), dtype=float)
        rho_xy_se = np.asarray(getattr(result, "rho_xy_se", []), dtype=float)
        rho_yx_se = np.asarray(getattr(result, "rho_yx_se", []), dtype=float)
        zxy = np.asarray(getattr(result, "zxy", []), dtype=np.complex128)
        zyx = np.asarray(getattr(result, "zyx", []), dtype=np.complex128)
        phi_xy_se = np.asarray(getattr(result, "phi_xy_se", []), dtype=float)
        phi_yx_se = np.asarray(getattr(result, "phi_yx_se", []), dtype=float)
        count = min(map(len, (
            periods, rho_xy, rho_yx, rho_xy_se, rho_yx_se,
            zxy, zyx, phi_xy_se, phi_yx_se,
        )))
        finite = np.isfinite(periods[:count]) & (periods[:count] > 0)
        if not finite.any():
            continue

        periods = periods[:count][finite]
        rho_xy = rho_xy[:count][finite]
        rho_yx = rho_yx[:count][finite]
        rho_xy_se = rho_xy_se[:count][finite]
        rho_yx_se = rho_yx_se[:count][finite]
        phi_xy = emtf_phase_deg(zxy[:count])[finite]
        phi_yx = emtf_phase_deg(zyx[:count])[finite]
        phi_xy_se = phi_xy_se[:count][finite]
        phi_yx_se = phi_yx_se[:count][finite]
        phase_values.extend((phi_xy, phi_yx))

        xy_colour, yx_colour = colour_pairs[index % len(colour_pairs)]
        rho_axis.errorbar(
            periods, rho_xy, yerr=rho_xy_se, fmt="o", color=xy_colour,
            capsize=3, markersize=5, label=r"$\rho_{xy}$",
        )
        rho_axis.errorbar(
            periods, rho_yx, yerr=rho_yx_se, fmt="s", color=yx_colour,
            capsize=3, markersize=5, label=r"$\rho_{yx}$",
        )
        phase_axis.errorbar(
            periods, phi_xy, yerr=phi_xy_se, fmt="o", color=xy_colour,
            capsize=3, markersize=5, label=r"$\phi_{xy}$",
        )
        phase_axis.errorbar(
            periods, phi_yx, yerr=phi_yx_se, fmt="s", color=yx_colour,
            capsize=3, markersize=5, label=r"$\phi_{yx}$",
        )
        plotted = True

    if not plotted:
        _clear_figure(figure)
        _draw_response_figure(figure, canvas, widget)
        return

    rho_axis.set_yscale("log")
    rho_axis.set_xscale("log")
    rho_axis.set_ylabel("Apparent Resistivity (Ω·m)")
    rho_axis.set_xlabel("Period (s)")
    rho_axis.set_title("视电阻率")
    rho_axis.grid(True, alpha=0.3)

    phase_axis.set_xscale("log")
    phase_axis.set_xlabel("Period (s)")
    phase_axis.set_ylabel("Phase (°)")
    phase_axis.set_title("相位")
    phase_axis.grid(True, alpha=0.3)
    phase_limits = phase_value_axis_limits(*phase_values)
    if phase_limits is not None:
        phase_axis.set_ylim(*phase_limits)

    _install_legend_resize_handler(figure, canvas)
    _refresh_adaptive_legends(figure, canvas)
    _draw_response_figure(figure, canvas, widget)


def draw_tipper_response(
    figure,
    canvas,
    results,
    mode: str = "re_im",
    widget=None,
) -> None:
    """Draw tipper results using the one canonical application style."""
    _clear_figure(figure)
    resolved_results = _as_results(results)
    if not resolved_results:
        _draw_response_figure(figure, canvas, widget)
        return
    real_or_amplitude_axis, imaginary_or_phase_axis = figure.subplots(2, 1)
    amplitude_phase = mode == "amp_phase"
    colour_pairs = (("red", "blue"), ("darkorange", "teal"), ("purple", "green"))
    plotted = False

    for index, result in enumerate(resolved_results):
        tzx = getattr(result, "tzx", None)
        tzy = getattr(result, "tzy", None)
        tzx_se = getattr(result, "tzx_se", None)
        tzy_se = getattr(result, "tzy_se", None)
        if tzx is None or tzy is None or tzx_se is None or tzy_se is None:
            continue

        periods = np.asarray(getattr(result, "periods", []), dtype=float)
        tzx = np.asarray(tzx, dtype=np.complex128)
        tzy = np.asarray(tzy, dtype=np.complex128)
        tzx_se = np.asarray(tzx_se, dtype=float)
        tzy_se = np.asarray(tzy_se, dtype=float)
        count = min(len(periods), len(tzx), len(tzy), len(tzx_se), len(tzy_se))
        finite = (
            np.isfinite(periods[:count]) & (periods[:count] > 0)
            & np.isfinite(tzx[:count].real) & np.isfinite(tzx[:count].imag)
            & np.isfinite(tzy[:count].real) & np.isfinite(tzy[:count].imag)
        )
        if not finite.any():
            continue

        periods = periods[:count][finite]
        tzx = tzx[:count][finite]
        tzy = tzy[:count][finite]
        tzx_se = tzx_se[:count][finite]
        tzy_se = tzy_se[:count][finite]
        tzx_colour, tzy_colour = colour_pairs[index % len(colour_pairs)]

        if amplitude_phase:
            tzx_amplitude_se, tzx_phase_se = propagate_complex_standard_error(tzx, tzx_se)
            tzy_amplitude_se, tzy_phase_se = propagate_complex_standard_error(tzy, tzy_se)
            real_or_amplitude_axis.errorbar(
                periods, np.abs(tzx), yerr=tzx_amplitude_se, fmt="o", color=tzx_colour,
                capsize=3, markersize=5, label=r"$T_{zx}$",
            )
            real_or_amplitude_axis.errorbar(
                periods, np.abs(tzy), yerr=tzy_amplitude_se, fmt="s", color=tzy_colour,
                capsize=3, markersize=5, label=r"$T_{zy}$",
            )
            imaginary_or_phase_axis.errorbar(
                periods,
                emtf_phase_deg(tzx),
                yerr=tzx_phase_se,
                fmt="o",
                color=tzx_colour,
                capsize=3,
                markersize=5,
                label=r"$T_{zx}$",
            )
            imaginary_or_phase_axis.errorbar(
                periods,
                emtf_phase_deg(tzy),
                yerr=tzy_phase_se,
                fmt="s",
                color=tzy_colour,
                capsize=3,
                markersize=5,
                label=r"$T_{zy}$",
            )
        else:
            real_or_amplitude_axis.errorbar(
                periods, tzx.real, yerr=tzx_se, fmt="o", color=tzx_colour,
                capsize=3, markersize=5, label=r"$T_{zx}$",
            )
            real_or_amplitude_axis.errorbar(
                periods, tzy.real, yerr=tzy_se, fmt="s", color=tzy_colour,
                capsize=3, markersize=5, label=r"$T_{zy}$",
            )
            imaginary_or_phase_axis.errorbar(
                periods, tzx.imag, yerr=tzx_se, fmt="o", color=tzx_colour,
                capsize=3, markersize=5, label=r"$T_{zx}$",
            )
            imaginary_or_phase_axis.errorbar(
                periods, tzy.imag, yerr=tzy_se, fmt="s", color=tzy_colour,
                capsize=3, markersize=5, label=r"$T_{zy}$",
            )
        plotted = True

    if not plotted:
        _clear_figure(figure)
        _draw_response_figure(figure, canvas, widget)
        return

    real_or_amplitude_axis.set_xscale("log")
    real_or_amplitude_axis.set_ylabel("Amplitude" if amplitude_phase else "Real")
    real_or_amplitude_axis.set_title("倾子响应幅值" if amplitude_phase else "倾子响应实部")
    real_or_amplitude_axis.grid(True, alpha=0.3)

    imaginary_or_phase_axis.set_xscale("log")
    imaginary_or_phase_axis.set_xlabel("Period (s)")
    imaginary_or_phase_axis.set_ylabel("Phase (°)" if amplitude_phase else "Imaginary")
    imaginary_or_phase_axis.set_title("倾子响应相位" if amplitude_phase else "倾子响应虚部")
    imaginary_or_phase_axis.grid(True, alpha=0.3)

    _install_legend_resize_handler(figure, canvas)
    _refresh_adaptive_legends(figure, canvas)
    _draw_response_figure(figure, canvas, widget)


def draw_c_response(figure, canvas, results, mode: str = "re_im", widget=None) -> None:
    """Draw C-response components with propagated error bars and no connecting lines."""
    _clear_figure(figure)
    resolved_results = _as_results(results)
    if not resolved_results:
        _draw_response_figure(figure, canvas, widget)
        return

    axes = figure.subplots(2, 2, sharex=True)
    amplitude_phase = mode == "amp_phase"
    plotted = False
    series = (
        (axes[0, 0], "cxy", "zxy", "red", "o"),
        (axes[0, 0], "cyx", "zyx", "blue", "s"),
        (axes[0, 1], "cxx", "zxx", "red", "o"),
        (axes[0, 1], "cyy", "zyy", "blue", "s"),
        (axes[1, 0], "cxy", "zxy", "red", "o"),
        (axes[1, 0], "cyx", "zyx", "blue", "s"),
        (axes[1, 1], "cxx", "zxx", "red", "o"),
        (axes[1, 1], "cyy", "zyy", "blue", "s"),
    )

    for result in resolved_results:
        periods = np.asarray(getattr(result, "periods", []), dtype=np.float64)
        components = c_response_from_bands(result)
        if not components:
            continue
        count = min(len(periods), *(len(value) for value in components.values()))
        finite_periods = np.isfinite(periods[:count]) & (periods[:count] > 0)
        if not finite_periods.any():
            continue
        periods = periods[:count][finite_periods]
        components = {name: values[:count][finite_periods] for name, values in components.items()}
        omega = 2.0 * np.pi / periods
        for axis, name, z_name, colour, marker in series:
            component = components[name]
            z_error = getattr(result, f"{z_name}_se", None)
            aligned_error = np.full(count, np.nan, dtype=np.float64)
            if z_error is not None:
                raw_error = np.asarray(z_error, dtype=np.float64)
                copied = min(count, len(raw_error))
                aligned_error[:copied] = raw_error[:copied]
            z_error = aligned_error[finite_periods]
            c_error = np.abs(1000.0 / omega) * z_error
            if amplitude_phase:
                amplitude_error, phase_error = propagate_complex_standard_error(component, c_error)
                first_row = axis is axes[0, 0] or axis is axes[0, 1]
                values = np.abs(component) if first_row else emtf_phase_deg(component)
                errors = amplitude_error if first_row else phase_error
            else:
                values = component.real if axis in (axes[0, 0], axes[0, 1]) else component.imag
                errors = c_error
            valid = (
                np.isfinite(periods) & (periods > 0)
                & np.isfinite(component.real) & np.isfinite(component.imag)
                & np.isfinite(values)
            )
            if valid.any():
                axis.errorbar(
                    periods[valid], values[valid], yerr=errors[valid],
                    fmt=marker, linestyle="none", color=colour,
                    capsize=3, markersize=5, label=rf"$C_{{{name[1:]}}}$",
                )
        plotted = True

    if not plotted:
        _clear_figure(figure)
        _draw_response_figure(figure, canvas, widget)
        return

    titles = (
        ("交叉分量幅值" if amplitude_phase else "交叉分量实部",
         "对角分量幅值" if amplitude_phase else "对角分量实部"),
        ("交叉分量相位" if amplitude_phase else "交叉分量虚部",
         "对角分量相位" if amplitude_phase else "对角分量虚部"),
    )
    ylabels = (
        ("Amplitude (m)" if amplitude_phase else "Real (m)",) * 2,
        ("Phase (°)" if amplitude_phase else "Imaginary (m)",) * 2,
    )
    for row in range(2):
        for column in range(2):
            axis = axes[row, column]
            axis.set_xscale("log")
            axis.set_title(titles[row][column])
            axis.set_ylabel(ylabels[row][column])
            axis.grid(True, alpha=0.3)
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                unique = dict(zip(labels, handles))
                axis.legend(unique.values(), unique.keys(), loc="best", fontsize=8, framealpha=0.4)
    axes[1, 0].set_xlabel("Period (s)")
    axes[1, 1].set_xlabel("Period (s)")
    _draw_response_figure(figure, canvas, widget)


def _comparison_color(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(channel / 255.0 for channel in rgb)


def _legend_text_width(value: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        for character in value
    )


def _wrap_comparison_label(value: object, max_width: int = 26) -> str:
    """Wrap result names by rendered width, including CJK full-width text."""
    text = str(value).strip()
    if not text:
        return ""

    lines: list[str] = []
    current = ""
    for character in text:
        if character == "\n":
            lines.append(current.rstrip())
            current = ""
            continue
        if current and _legend_text_width(current + character) > max_width:
            split_at = max(current.rfind(" "), current.rfind("·"))
            if split_at > 0:
                lines.append(current[:split_at].rstrip())
                current = current[split_at:].lstrip()
            else:
                lines.append(current.rstrip())
                current = ""
        current += character
    if current or not lines:
        lines.append(current.rstrip())
    return "\n".join(lines)


def _add_comparison_legend(figure, handles) -> None:
    legend_fontsize = max(6.0, 9.0 - max(0, len(handles) - 5) * 0.3)
    legend = figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.835, 0.96),
        ncols=1,
        fontsize=legend_fontsize,
        framealpha=0.4,
        edgecolor="none",
        handletextpad=0.6,
        labelspacing=1.2,
        borderaxespad=0.0,
    )
    legend.get_frame().set_facecolor("white")


def _plot_comparison_points(axis, periods, values, errors, color) -> bool:
    count = min(len(periods), len(values), len(errors))
    periods = np.asarray(periods[:count], dtype=float)
    values = np.asarray(values[:count], dtype=float)
    errors = np.asarray(errors[:count], dtype=float)
    finite = np.isfinite(periods) & (periods > 0) & np.isfinite(values)
    if not finite.any():
        return False

    periods = periods[finite]
    values = values[finite]
    errors = errors[finite]
    axis.errorbar(
        periods,
        values,
        yerr=errors,
        fmt="none",
        ecolor=color,
        elinewidth=1.0,
        capsize=2,
        alpha=0.6,
        zorder=1,
    )
    axis.scatter(
        periods,
        values,
        s=22,
        marker="o",
        color=[color],
        alpha=1.0,
        zorder=3,
    )
    return True


def draw_impedance_comparison(figure, canvas, series, widget=None) -> None:
    """Draw multiple impedance results as four component-specific subplots."""
    _clear_figure(figure)
    if not series:
        configure_matplotlib_canvas(canvas, figure, widget)
        canvas.draw()
        return

    axes = np.asarray(figure.subplots(2, 2, sharex=True, sharey="row"))
    rho_xy_axis, rho_yx_axis = axes[0]
    phi_xy_axis, phi_yx_axis = axes[1]
    phase_values: list[np.ndarray] = []
    legend_handles = []
    plotted = False

    for name, result, rgb in series:
        periods = np.asarray(getattr(result, "periods", []), dtype=float)
        rho_xy = np.asarray(getattr(result, "rho_xy", []), dtype=float)
        rho_yx = np.asarray(getattr(result, "rho_yx", []), dtype=float)
        rho_xy_se = np.asarray(getattr(result, "rho_xy_se", []), dtype=float)
        rho_yx_se = np.asarray(getattr(result, "rho_yx_se", []), dtype=float)
        zxy = np.asarray(getattr(result, "zxy", []), dtype=np.complex128)
        zyx = np.asarray(getattr(result, "zyx", []), dtype=np.complex128)
        phi_xy_se = np.asarray(getattr(result, "phi_xy_se", []), dtype=float)
        phi_yx_se = np.asarray(getattr(result, "phi_yx_se", []), dtype=float)
        count = min(map(len, (
            periods,
            rho_xy,
            rho_yx,
            rho_xy_se,
            rho_yx_se,
            zxy,
            zyx,
            phi_xy_se,
            phi_yx_se,
        )))
        if count == 0:
            continue

        periods = periods[:count]
        phi_xy = emtf_phase_deg(zxy[:count])
        phi_yx = emtf_phase_deg(zyx[:count])
        phase_values.extend((phi_xy, phi_yx))
        color = _comparison_color(rgb)
        result_plotted = any((
            _plot_comparison_points(
                rho_xy_axis, periods, rho_xy[:count], rho_xy_se[:count], color,
            ),
            _plot_comparison_points(
                rho_yx_axis, periods, rho_yx[:count], rho_yx_se[:count], color,
            ),
            _plot_comparison_points(
                phi_xy_axis, periods, phi_xy, phi_xy_se[:count], color,
            ),
            _plot_comparison_points(
                phi_yx_axis, periods, phi_yx, phi_yx_se[:count], color,
            ),
        ))
        if not result_plotted:
            continue

        plotted = True
        legend_handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=6,
                color=color,
                label=_wrap_comparison_label(name),
            )
        )

    if not plotted:
        _clear_figure(figure)
        configure_matplotlib_canvas(canvas, figure, widget)
        canvas.draw()
        return

    for axis, title in (
        (rho_xy_axis, r"$\rho_{xy}$"),
        (rho_yx_axis, r"$\rho_{yx}$"),
        (phi_xy_axis, r"$\phi_{xy}$"),
        (phi_yx_axis, r"$\phi_{yx}$"),
    ):
        axis.set_title(title)
        axis.set_xscale("log")
        axis.set_axisbelow(True)
        axis.grid(True, alpha=0.2, zorder=0)

    rho_xy_axis.set_yscale("log")
    rho_yx_axis.set_yscale("log")
    rho_xy_axis.set_ylabel("Apparent Resistivity (Ω·m)")
    phi_xy_axis.set_ylabel("Phase (°)")
    phi_xy_axis.set_xlabel("Period (s)")
    phi_yx_axis.set_xlabel("Period (s)")

    phase_limits = phase_value_axis_limits(*phase_values)
    if phase_limits is not None:
        phi_xy_axis.set_ylim(*phase_limits)

    figure.subplots_adjust(
        left=0.08,
        right=0.82,
        bottom=0.09,
        top=0.94,
        wspace=0.12,
        hspace=0.22,
    )
    _add_comparison_legend(figure, legend_handles)
    configure_matplotlib_canvas(canvas, figure, widget)
    canvas.draw()


def draw_tipper_comparison(
    figure,
    canvas,
    series,
    mode: str = "re_im",
    widget=None,
) -> None:
    """Draw multiple tipper results as four component-specific subplots."""
    _clear_figure(figure)
    if not series:
        configure_matplotlib_canvas(canvas, figure, widget)
        canvas.draw()
        return

    axes = np.asarray(figure.subplots(2, 2, sharex=True, sharey="row"))
    top_left_axis, top_right_axis = axes[0]
    bottom_left_axis, bottom_right_axis = axes[1]
    amplitude_phase = mode == "amp_phase"
    phase_values: list[np.ndarray] = []
    legend_handles = []
    plotted = False

    for name, result, rgb in series:
        tzx = getattr(result, "tzx", None)
        tzy = getattr(result, "tzy", None)
        tzx_se = getattr(result, "tzx_se", None)
        tzy_se = getattr(result, "tzy_se", None)
        if tzx is None or tzy is None or tzx_se is None or tzy_se is None:
            continue

        periods = np.asarray(getattr(result, "periods", []), dtype=float)
        tzx = np.asarray(tzx, dtype=np.complex128)
        tzy = np.asarray(tzy, dtype=np.complex128)
        tzx_se = np.asarray(tzx_se, dtype=float)
        tzy_se = np.asarray(tzy_se, dtype=float)
        count = min(len(periods), len(tzx), len(tzy), len(tzx_se), len(tzy_se))
        if count == 0:
            continue

        periods = periods[:count]
        tzx = tzx[:count]
        tzy = tzy[:count]
        tzx_se = tzx_se[:count]
        tzy_se = tzy_se[:count]
        if amplitude_phase:
            tzx_amplitude_se, tzx_phase_se = propagate_complex_standard_error(tzx, tzx_se)
            tzy_amplitude_se, tzy_phase_se = propagate_complex_standard_error(tzy, tzy_se)
            tzx_phase = emtf_phase_deg(tzx)
            tzy_phase = emtf_phase_deg(tzy)
            phase_values.extend((tzx_phase, tzy_phase))
            values_and_errors = (
                (np.abs(tzx), tzx_amplitude_se),
                (np.abs(tzy), tzy_amplitude_se),
                (tzx_phase, tzx_phase_se),
                (tzy_phase, tzy_phase_se),
            )
        else:
            values_and_errors = (
                (tzx.real, tzx_se),
                (tzy.real, tzy_se),
                (tzx.imag, tzx_se),
                (tzy.imag, tzy_se),
            )

        color = _comparison_color(rgb)
        result_plotted = any([
            _plot_comparison_points(axis, periods, values, errors, color)
            for axis, (values, errors) in zip(axes.flat, values_and_errors)
        ])
        if not result_plotted:
            continue

        plotted = True
        legend_handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=6,
                color=color,
                label=_wrap_comparison_label(name),
            )
        )

    if not plotted:
        _clear_figure(figure)
        configure_matplotlib_canvas(canvas, figure, widget)
        canvas.draw()
        return

    if amplitude_phase:
        titles = (
            r"$|T_{zx}|$",
            r"$|T_{zy}|$",
            r"$\arg(T_{zx})$",
            r"$\arg(T_{zy})$",
        )
        top_left_axis.set_ylabel("Amplitude")
        bottom_left_axis.set_ylabel("Phase (°)")
    else:
        titles = (
            r"$\mathrm{Re}(T_{zx})$",
            r"$\mathrm{Re}(T_{zy})$",
            r"$\mathrm{Im}(T_{zx})$",
            r"$\mathrm{Im}(T_{zy})$",
        )
        top_left_axis.set_ylabel("Real")
        bottom_left_axis.set_ylabel("Imaginary")

    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xscale("log")
        axis.set_axisbelow(True)
        axis.grid(True, alpha=0.2, zorder=0)
    bottom_left_axis.set_xlabel("Period (s)")
    bottom_right_axis.set_xlabel("Period (s)")

    if amplitude_phase:
        phase_limits = phase_value_axis_limits(*phase_values)
        if phase_limits is not None:
            bottom_left_axis.set_ylim(*phase_limits)

    figure.subplots_adjust(
        left=0.08,
        right=0.82,
        bottom=0.09,
        top=0.94,
        wspace=0.12,
        hspace=0.22,
    )
    _add_comparison_legend(figure, legend_handles)
    configure_matplotlib_canvas(canvas, figure, widget)
    canvas.draw()
