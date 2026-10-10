from emtf.config import render_bands
from emtf.band_presets import (
    fine_bands,
    half_decade_bands,
    period_grid_bands,
    preset_bands,
    standard_bands,
)
from emtf.estimate import _decimation_from_window_size


def test_standard_band_preset_matches_existing_default_shape():
    bands = standard_bands()

    assert len(bands) == 27
    assert bands[0].decimation_level == 1
    assert (bands[0].fc_start, bands[0].fc_end) == (8, 9)
    assert (bands[-1].fc_start, bands[-1].fc_end) == (5, 5)


def test_named_presets_render_valid_bs_cfg():
    for name in ("standard", "fine", "half-decade"):
        text = render_bands(preset_bands(name))
        lines = text.splitlines()
        assert int(lines[0]) == len(lines) - 1


def test_fine_and_half_decade_presets_have_expected_band_widths():
    fine = fine_bands()
    half_decade = half_decade_bands()

    assert all(b.fc_start == b.fc_end for b in fine)
    assert any(b.fc_start != b.fc_end for b in half_decade)


def test_period_grid_bands_map_period_range_to_bs_cfg():
    levels = _decimation_from_window_size(256)
    bands = period_grid_bands(
        1.0,
        levels,
        min_period=2.0,
        max_period=256.0,
        n_edges=40,
    )
    text = render_bands(bands)

    assert bands
    assert all(1 <= band.decimation_level <= len(levels) for band in bands)
    assert all(5 <= band.fc_start <= band.fc_end for band in bands)
    assert int(text.splitlines()[0]) == len(bands)
