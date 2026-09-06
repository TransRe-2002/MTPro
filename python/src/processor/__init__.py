"""Deprecated compatibility namespace.

Processing implementations moved to :mod:`plugins.common` and the feature
packages. This alias table keeps older scripts importable without making the
host depend on the old package layout.
"""

from __future__ import annotations

import importlib
import sys

_MODULES = {
    "_base_processor": "plugins.common.base_processor",
    "_demo_utils": "plugins.common.demo",
    "_display_utils": "plugins.common.display",
    "_modifier_constraint": "plugins.common.modifier_constraint",
    "_mt_common": "output_tools.shared.emtf",
    "_region_mixin": "plugins.common.region",
    "diurnal_sinusoid_subtraction": "plugins.common.diurnal",
    "step_algorithms": "plugins.common.step_algorithms",
    "step_plot_widget": "plugins.common.plotting",
    "remove_diurnal_sinusoid": "plugins.remove_diurnal_sinusoid.editor",
    "remove_region": "plugins.remove_region.editor",
    "remove_spike": "plugins.remove_spike.editor",
    "remove_step": "plugins.common.step_editor_alias",
    "remove_step_by_baseline": "plugins.remove_step_baseline.editor",
    "remove_step_by_diff": "plugins.remove_step_diff.editor",
    "remove_step_by_window": "plugins.remove_step_window.editor",
    "remove_wavelet_denoise": "plugins.remove_wavelet_denoise.editor",
}
for _name, _target in _MODULES.items():
    _module = importlib.import_module(_target)
    sys.modules.setdefault(f"{__name__}.{_name}", _module)
    setattr(sys.modules[__name__], _name, _module)

# Output tools are intentionally not plugins yet, but remain import-compatible.
for _name in ("robust_estimate", "remote_reference"):
    _module = importlib.import_module(f"output_tools.{_name}")
    sys.modules.setdefault(f"{__name__}.{_name}", _module)
    setattr(sys.modules[__name__], _name, _module)

__all__ = []
