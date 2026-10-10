from __future__ import annotations

import numpy as np
import pytest

import rust_lib
from io_utils.txt_io import TxtLoader


ROW_1 = (
    "2026 05 31 03 55 00 26181.621 -61.059 48518.992 "
    "43.36 201.69 -19.898 -29.367 -4.446 -123.712 "
    "12.98 1009.1 4023.669619 N 08120.624305 E 8 2 0\n"
)
ROW_2 = (
    "2026 05 31 03 55 01 26181.604 -61.084 48519.020 "
    "43.40 201.69 -19.903 -29.370 -4.452 -123.712 "
    "12.98 1009.1 4023.669619 N 08120.624305 E 8 2 0\n"
)


pytestmark = pytest.mark.skipif(
    not hasattr(rust_lib, "read_mt_txt_rs"),
    reason="rust_lib was not rebuilt with the TXT parser",
)


def test_txt_loader_reads_all_channels_gps_and_localizes_time(tmp_path):
    path = tmp_path / "station.TXT"
    path.write_text(ROW_1 + ROW_2, encoding="ascii", newline="")

    data = TxtLoader.load(str(path), tz="Asia/Shanghai")

    assert data.name == "station"
    assert data.channel_names() == ["Hx", "Hy", "Hz", "Ex1", "Ey1", "Ex2", "Ey2"]
    assert data.npts == 2
    assert data.dt.total_seconds() == 1.0
    assert str(data.start_time) == "2026-05-30 19:55:00+00:00"
    assert str(data.end_time) == "2026-05-30 19:55:01+00:00"
    assert data.latitude == pytest.approx(40.39449365)
    assert data.longitude == pytest.approx(81.3437384167)
    np.testing.assert_allclose(data.get_channel_series("Hx"), [26181.621, 26181.604])
    np.testing.assert_allclose(data.get_channel_series("Ex2"), [-4.446, -4.452])
    assert data.import_diagnostics["gps_samples"] == 2


def test_txt_loader_reports_irregular_sampling(tmp_path):
    path = tmp_path / "irregular.txt"
    row_3 = ROW_2.replace("03 55 01", "03 55 03", 1)
    path.write_text(ROW_1 + ROW_2 + row_3, encoding="ascii", newline="")

    with pytest.raises(ValueError, match="采样间隔为2秒"):
        TxtLoader.load(str(path), tz="UTC")
