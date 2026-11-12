from __future__ import annotations

import pytest
from PySide6.QtWidgets import QInputDialog

from output_tools.cross_station_impedance import CrossStationImpedanceWidget
from output_tools.remote_reference import RemoteReferenceWidget
from output_tools.robust_estimate import RobustEstimateWidget
from output_tools.shared.result_naming import prompt_result_name
from test_impedance_preview import _bands
from ui.impedance_preview import ImpedanceResultPool


@pytest.mark.parametrize('widget_type,method', [
    (RobustEstimateWidget, 'robust estimate'),
    (RemoteReferenceWidget, 'remote reference'),
    (CrossStationImpedanceWidget, 'cross-station impedance'),
])
def test_save_prompts_and_cancel_does_not_save(qtbot, monkeypatch, widget_type, method):
    widget = widget_type()
    qtbot.addWidget(widget)
    widget._bands = _bands()
    widget._calculation_context = {'method': method, 'station_name': 'STA', 'parameters': {}}
    widget._saved_current = False
    widget.save_preview_btn.setEnabled(True)
    monkeypatch.setattr(widget, '_current_calculation_signature', lambda: (1,))
    monkeypatch.setattr(widget, '_saved_calculation_signature', lambda: (1,))
    pool = ImpedanceResultPool()
    widget.snapshot_save_requested.connect(pool.add_payload)
    prompts = []

    def ask(*args):
        prompts.append(args)
        return ('ignored', False) if len(prompts) == 1 else ('  测站去噪结果  ', True)

    monkeypatch.setattr(QInputDialog, 'getText', ask)
    widget._save_to_preview()
    assert not pool.items() and not widget._saved_current
    assert widget.save_preview_btn.isEnabled()
    assert prompts[0][4].startswith('STA')
    widget._save_to_preview()
    assert len(pool.items()) == 1 and pool.items()[0].name == '测站去噪结果'
    assert widget._saved_current and not widget.save_preview_btn.isEnabled()
    assert not widget.view_preview_btn.isHidden()


def test_empty_name_uses_default(qtbot, monkeypatch):
    widget = RobustEstimateWidget()
    qtbot.addWidget(widget)
    monkeypatch.setattr(QInputDialog, 'getText', lambda *_args: ('  ', True))
    assert prompt_result_name(widget, {'method': 'robust estimate', 'station_name': 'STA'}) == 'STA Robust'
