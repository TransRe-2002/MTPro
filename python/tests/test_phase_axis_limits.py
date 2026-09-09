import numpy as np
from matplotlib.figure import Figure

from output_tools.shared.emtf import phase_value_axis_limits


def test_phase_axis_limits_use_phase_values_with_padding():
    limits = phase_value_axis_limits(
        np.array([-20.0, 10.0]),
        np.array([30.0, 40.0]),
    )

    assert limits == (-26.0, 46.0)


def test_phase_axis_limits_ignore_error_bars_and_stay_within_phase_bounds():
    axis = Figure().subplots()
    phase_xy = np.array([-20.0, 10.0])
    phase_yx = np.array([30.0, 40.0])
    axis.errorbar([1.0, 2.0], phase_xy, yerr=[100.0, 100.0])
    axis.errorbar([1.0, 2.0], phase_yx, yerr=[100.0, 100.0])

    axis.set_ylim(*phase_value_axis_limits(phase_xy, phase_yx))

    assert axis.get_ylim() == (-26.0, 46.0)


def test_phase_axis_limits_return_none_without_finite_phase_values():
    assert phase_value_axis_limits(np.array([np.nan, np.inf])) is None
    assert phase_value_axis_limits() is None
