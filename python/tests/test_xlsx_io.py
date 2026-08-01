from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile

import numpy as np
import pandas as pd

from io_utils.xlsx_io import (
    INTERP_LINEAR,
    INTERP_NONE,
    MAX_POINTS_PER_CHANNEL,
    XlsxLoader,
    estimate_channel_points,
    format_points,
    read_xlsx_header,
    recommend_min_dt,
    resample_series_to_grid,
    summarize_xlsx_groups,
)


def test_linear_interpolation_is_weighted_by_time():
    source_index = pd.DatetimeIndex([
        "2024-01-01 00:00:00+00:00",
        "2024-01-01 00:00:04+00:00",
    ])
    source = pd.Series([0.0, 8.0], index=source_index, name="Bx")
    target = pd.DatetimeIndex(["2024-01-01 00:00:01+00:00"])

    result = resample_series_to_grid(
        source,
        target,
        dt=pd.Timedelta(seconds=4),
        interpolation=INTERP_LINEAR,
        max_side_gap_factor=1.0,
    )

    assert result.iloc[0] == 2.0


def test_linear_interpolation_rejects_point_when_either_side_exceeds_dt():
    source_index = pd.DatetimeIndex([
        "2024-01-01 00:00:00+00:00",
        "2024-01-01 00:00:04+00:00",
    ])
    source = pd.Series([0.0, 8.0], index=source_index, name="Bx")
    target = pd.DatetimeIndex(["2024-01-01 00:00:01+00:00"])

    result = resample_series_to_grid(
        source,
        target,
        dt=pd.Timedelta(seconds=2),
        interpolation=INTERP_LINEAR,
        max_side_gap_factor=1.0,
    )

    assert np.isnan(result.iloc[0])


def test_none_interpolation_keeps_exact_hits_only():
    source_index = pd.DatetimeIndex([
        "2024-01-01 00:00:00+00:00",
        "2024-01-01 00:00:04+00:00",
    ])
    source = pd.Series([0.0, 8.0], index=source_index, name="Bx")
    target = pd.DatetimeIndex([
        "2024-01-01 00:00:00+00:00",
        "2024-01-01 00:00:02+00:00",
        "2024-01-01 00:00:04+00:00",
    ])

    result = resample_series_to_grid(
        source,
        target,
        dt=pd.Timedelta(seconds=2),
        interpolation=INTERP_NONE,
        max_side_gap_factor=1.0,
    )

    np.testing.assert_allclose(result.to_numpy()[[0, 2]], [0.0, 8.0])
    assert np.isnan(result.iloc[1])


def test_xlsx_header_and_loader_read_inline_string_workbook(tmp_path):
    path = tmp_path / "demo.xlsx"
    _write_minimal_xlsx(path)

    summary = read_xlsx_header(str(path))
    assert [sheet.sheet_name for sheet in summary.sheets] == ["Bx_1", "Ex1_5"]
    assert summary.sheets[0].channel_name == "Bx"
    assert summary.sheets[0].rows == 3
    assert summary.inferred_dt_sec == 2.0

    data = XlsxLoader.load(
        str(path),
        tz="Asia/Shanghai",
        channel_configs=[
            {"sheet": "Bx_1", "name": "Bx", "enabled": True, "scale": 1.0},
            {"sheet": "Ex1_5", "name": "Ex1", "enabled": True, "scale": 1.0},
        ],
        dt_override=2.0,
    )

    assert data.channel_names() == ["Bx", "Ex1"]
    assert data.npts == 3
    np.testing.assert_allclose(data.get_channel_series("Bx").to_numpy(), [1.0, 2.0, 3.0])
    assert str(data.datetime_index[0]) == "2023-12-31 16:00:00+00:00"


def test_load_group_merges_same_prefix_files_and_deduplicates_equal_timestamps(tmp_path):
    first = tmp_path / "031BE-20240101-20240101.xlsx"
    second = tmp_path / "031BE-20240102-20240102.xlsx"
    _write_minimal_xlsx_with_times(
        first,
        ["2024-01-01 00:00:00", "2024-01-01 00:00:02", "2024-01-01 00:00:04"],
        [1.0, 2.0, 3.0],
    )
    _write_minimal_xlsx_with_times(
        second,
        ["2024-01-01 00:00:04", "2024-01-01 00:00:06", "2024-01-01 00:00:08"],
        [3.0, 4.0, 5.0],
    )

    groups = summarize_xlsx_groups([first, second])
    assert len(groups) == 1
    assert groups[0].prefix == "031BE"
    assert len(groups[0].touches) == 1

    data = XlsxLoader.load_group(
        [str(first), str(second)],
        tz="Asia/Shanghai",
        channel_configs=[
            {"sheet": "Bx_1", "name": "Bx", "enabled": True, "scale": 1.0},
            {"sheet": "Ex1_5", "name": "Ex1", "enabled": False, "scale": 1.0},
        ],
        dt_override=2.0,
    )

    assert data.npts == 5
    np.testing.assert_allclose(data.get_channel_series("Bx").to_numpy(), [1, 2, 3, 4, 5])
    assert data.import_diagnostics["Bx"]["duplicate_timestamps"] == 1
    assert data.import_diagnostics["Bx"]["conflict_timestamps"] == 0


def _write_minimal_xlsx(path):
    _write_minimal_xlsx_with_times(
        path,
        ["2024-01-01 00:00:00", "2024-01-01 00:00:02", "2024-01-01 00:00:04"],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    )


def _write_minimal_xlsx_with_times(path, times, bx_values, ex_values=None):
    if ex_values is None:
        ex_values = bx_values
    with ZipFile(path, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>""",
        )
        zf.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
        )
        zf.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets>
<sheet name="Bx_1" sheetId="1" r:id="rId1"/>
<sheet name="Ex1_5" sheetId="2" r:id="rId2"/>
</sheets>
</workbook>""",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
</Relationships>""",
        )
        zf.writestr("xl/worksheets/sheet1.xml", _sheet_xml("Bx", times, bx_values))
        zf.writestr("xl/worksheets/sheet2.xml", _sheet_xml("Ex1", times, ex_values))


def _sheet_xml(channel, times, values):
    rows = [
        _row(1, ["\u66f4\u65b0\u65f6\u95f4", channel]),
    ]
    for index, (timestamp, value) in enumerate(zip(times, values), start=2):
        rows.append(_row(index, [timestamp, f"{value}"]))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        + "".join(rows)
        + "</sheetData></worksheet>"
    )


def _row(row_number, values):
    refs = ["A", "B"]
    cells = []
    for ref, value in zip(refs, values):
        cells.append(
            f'<c r="{ref}{row_number}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
        )
    return f'<row r="{row_number}">' + "".join(cells) + "</row>"


def test_estimate_channel_points_inclusive_of_both_ends():
    assert estimate_channel_points(3600.0, 2.0) == 1801
    assert estimate_channel_points(0.0, 2.0) == 0
    assert estimate_channel_points(3600.0, 0.0) == 0


def test_recommend_min_dt_one_year_range_stays_under_limit():
    one_year = 365 * 24 * 3600.0
    dt = recommend_min_dt(one_year)
    assert dt == 6.4  # 6.307... 向上取整到 2 位有效数字
    assert estimate_channel_points(one_year, dt) <= MAX_POINTS_PER_CHANNEL


def test_recommend_min_dt_one_day_range():
    dt = recommend_min_dt(86400.0)
    assert dt == 0.018  # 0.01728 → 0.018
    assert estimate_channel_points(86400.0, dt) <= MAX_POINTS_PER_CHANNEL


def test_format_points_chinese_compact():
    assert format_points(9999) == "9,999"
    assert format_points(15_768_001) == "1576.8万"
    assert format_points(200_000_000) == "2.00亿"


def _two_touching_files(tmp_path):
    first = tmp_path / "031BE-a.xlsx"
    second = tmp_path / "031BE-b.xlsx"
    _write_minimal_xlsx_with_times(
        first, ["2024-01-01 00:00:00", "2024-01-01 00:00:02"], [1.0, 2.0],
    )
    _write_minimal_xlsx_with_times(
        second, ["2024-01-01 00:00:04", "2024-01-01 00:00:06"], [3.0, 4.0],
    )
    return [str(first), str(second)]


_ENABLED_BX = [{"sheet": "Bx_1", "name": "Bx", "enabled": True, "scale": 1.0}]


def test_load_group_workers_result_matches_sequential(tmp_path):
    files = _two_touching_files(tmp_path)
    seq = XlsxLoader.load_group(
        files, tz="Asia/Shanghai", channel_configs=_ENABLED_BX,
        dt_override=2.0, workers=1,
    )
    par = XlsxLoader.load_group(
        files, tz="Asia/Shanghai", channel_configs=_ENABLED_BX,
        dt_override=2.0, workers=4,
    )
    np.testing.assert_array_equal(
        seq.get_channel_series("Bx").to_numpy(),
        par.get_channel_series("Bx").to_numpy(),
    )
    assert par.import_diagnostics["xlsx_import"]["failed_files"] == []


def test_load_group_skips_unreadable_file_and_reports(tmp_path):
    files = _two_touching_files(tmp_path)
    bad = tmp_path / "031BE-bad.xlsx"
    bad.write_bytes(b"not a zip")

    data = XlsxLoader.load_group(
        [files[0], str(bad), files[1]],
        tz="Asia/Shanghai", channel_configs=_ENABLED_BX,
        dt_override=2.0, workers=4,
    )
    failed = data.import_diagnostics["xlsx_import"]["failed_files"]
    assert len(failed) == 1
    assert Path(failed[0]["path"]).name == "031BE-bad.xlsx"
    assert data.npts == 4  # 两个好文件照常合并


def test_load_group_cancel_raises(tmp_path):
    import pytest
    from io_utils.xlsx_io import XlsxImportCancelled

    files = _two_touching_files(tmp_path)
    with pytest.raises(XlsxImportCancelled):
        XlsxLoader.load_group(
            files, tz="Asia/Shanghai", channel_configs=_ENABLED_BX,
            dt_override=2.0, workers=2, cancel_check=lambda: True,
        )


def test_load_group_progress_reports_sheet_units(tmp_path):
    files = _two_touching_files(tmp_path)
    events = []
    XlsxLoader.load_group(
        files, tz="Asia/Shanghai", channel_configs=_ENABLED_BX,
        dt_override=2.0, workers=2,
        progress_callback=lambda done, total, label: events.append((done, total)),
    )
    # 2 文件 × 1 个启用通道 = 2 个 sheet 刻度
    assert {total for _, total in events} == {2}
    assert max(done for done, _ in events) == 2
