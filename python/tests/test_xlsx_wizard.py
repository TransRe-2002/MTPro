import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
from PySide6.QtWidgets import QApplication

from io_utils.xlsx_io import (
    XlsxChannelSummary,
    XlsxFileSummary,
    XlsxGroupSummary,
    XlsxWorkbookSummary,
)
from ui.import_xlsx_wizard import ImportXlsxWizard


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _group(prefix: str, first: str, last: str, dt: float) -> XlsxGroupSummary:
    file = XlsxFileSummary(
        path=f"/tmp/{prefix}.xlsx",
        prefix=prefix,
        file_size_mb=1.0,
        first_time=pd.Timestamp(first),
        last_time=pd.Timestamp(last),
        inferred_dt_sec=dt,
        rows=100,
    )
    return XlsxGroupSummary(
        prefix=prefix, files=(file,), overlaps=(), touches=(), gaps=(),
    )


def _summary(path: str, channel: str) -> XlsxWorkbookSummary:
    return XlsxWorkbookSummary(
        path=path,
        file_size_mb=1.0,
        sheets=(XlsxChannelSummary(f"{channel}_1", channel, 10, None, None, None, ()),),
    )


def test_sampling_page_warns_and_recommends_dt_for_year_at_2s():
    _app()
    wizard = ImportXlsxWizard(None)
    # 365 天、2 s 采样 → 1576.8 万点/通道，超过 500 万上限
    wizard._group_summaries = [_group("031BE", "2025-01-01", "2026-01-01", 2.0)]
    page = wizard._page3
    page.initializePage()

    text = page._dt_estimate_label.text()
    assert "1576.8万" in text
    assert "超过建议上限" in text
    assert "dt ≥ 6.4 s" in text


def test_sampling_page_no_warning_for_small_range():
    _app()
    wizard = ImportXlsxWizard(None)
    wizard._group_summaries = [_group("031BE", "2025-01-01", "2025-01-08", 2.0)]
    page = wizard._page3
    page.initializePage()

    text = page._dt_estimate_label.text()
    assert "超过建议上限" not in text
    assert "30.2万" in text


def test_sampling_page_estimate_updates_with_manual_dt():
    _app()
    wizard = ImportXlsxWizard(None)
    wizard._group_summaries = [_group("031BE", "2025-01-01", "2026-01-01", 2.0)]
    page = wizard._page3
    page.initializePage()
    assert "超过建议上限" in page._dt_estimate_label.text()

    # 手动选 10 s → 315.4 万点，低于上限，预警消失
    page._dt_manual_radio.setChecked(True)
    page._dt_manual_spin.setValue(10.0)
    text = page._dt_estimate_label.text()
    assert "315.4万" in text
    assert "超过建议上限" not in text


def test_channel_page_keeps_edits_when_revisiting():
    _app()
    wizard = ImportXlsxWizard(None)
    wizard._summary = _summary("/tmp/demo.xlsx", "Bx")
    page = wizard._page2
    page.initializePage()
    page._name_edits[0].setText("Bx_renamed")
    page._scale_spins[0].setValue(2.5)

    # 模拟 第3页 → 返回第2页（initializePage 会再次被调用）
    page.initializePage()
    assert page._name_edits[0].text() == "Bx_renamed"
    assert page._scale_spins[0].value() == 2.5


def test_channel_page_rebuilds_when_summary_source_changes():
    _app()
    wizard = ImportXlsxWizard(None)
    wizard._summary = _summary("/tmp/a.xlsx", "Bx")
    page = wizard._page2
    page.initializePage()
    page._name_edits[0].setText("edited")

    wizard._summary = _summary("/tmp/b.xlsx", "Hy")
    page.initializePage()
    assert page._name_edits[0].text() == "Hy"


def test_wizard_end_to_end_imports_dataset(tmp_path):
    """完整走一遍 validatePage：后台线程导入 → DataManager 收到数据集。"""
    from test_xlsx_io import _write_minimal_xlsx_with_times
    from base.data_manager import DataManager
    from io_utils.xlsx_io import summarize_xlsx_groups

    _app()
    first = tmp_path / "031BE-a.xlsx"
    second = tmp_path / "031BE-b.xlsx"
    _write_minimal_xlsx_with_times(
        first, ["2024-01-01 00:00:00", "2024-01-01 00:00:02"], [1.0, 2.0],
    )
    _write_minimal_xlsx_with_times(
        second, ["2024-01-01 00:00:04", "2024-01-01 00:00:06"], [3.0, 4.0],
    )

    dm = DataManager()
    wizard = ImportXlsxWizard(dm)
    wizard._group_summaries = summarize_xlsx_groups([str(first), str(second)])
    # 第 2 页未构建 → channel_configs 为空 → loader 从工作簿自推导通道
    assert wizard._page4.validatePage() is True

    keys = wizard.imported_keys()
    assert len(keys) == 1
    data = dm.get(keys[0])
    assert data is not None
    assert data.npts == 4


def test_wizard_cancel_rolls_back(monkeypatch, tmp_path):
    from test_xlsx_io import _write_minimal_xlsx_with_times
    from base.data_manager import DataManager
    from io_utils.xlsx_io import (
        XlsxImportCancelled, XlsxLoader, summarize_xlsx_groups,
    )

    _app()
    first = tmp_path / "031BE-a.xlsx"
    _write_minimal_xlsx_with_times(
        first, ["2024-01-01 00:00:00", "2024-01-01 00:00:02"], [1.0, 2.0],
    )
    dm = DataManager()
    wizard = ImportXlsxWizard(dm)
    wizard._group_summaries = summarize_xlsx_groups([str(first)])

    def fake_load_group(*args, **kwargs):
        raise XlsxImportCancelled("cancelled")

    monkeypatch.setattr(XlsxLoader, "load_group", staticmethod(fake_load_group))
    assert wizard._page4.validatePage() is False
    assert wizard.imported_keys() == []
    assert list(dm.items()) == []


def test_wizard_import_error_shows_failure_and_rolls_back(monkeypatch, tmp_path):
    from test_xlsx_io import _write_minimal_xlsx_with_times
    from base.data_manager import DataManager
    from io_utils.xlsx_io import XlsxLoader, summarize_xlsx_groups

    _app()
    first = tmp_path / "031BE-a.xlsx"
    _write_minimal_xlsx_with_times(
        first, ["2024-01-01 00:00:00", "2024-01-01 00:00:02"], [1.0, 2.0],
    )
    dm = DataManager()
    wizard = ImportXlsxWizard(dm)
    wizard._group_summaries = summarize_xlsx_groups([str(first)])

    def fake_load_group(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(XlsxLoader, "load_group", staticmethod(fake_load_group))
    # 屏蔽模态错误弹窗，避免阻塞测试
    monkeypatch.setattr(
        "ui.import_xlsx_wizard.QMessageBox.critical", lambda *a, **k: None
    )
    assert wizard._page4.validatePage() is False
    assert wizard.imported_keys() == []
    assert list(dm.items()) == []
