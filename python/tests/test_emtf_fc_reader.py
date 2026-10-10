"""FC record layout regressions; generated fixtures do not require Fortran."""
import struct

import numpy as np
import pytest

from emtf.io.fc import FCOutput


@pytest.mark.parametrize("endian,prefix", [("little", "<"), ("big", ">")])
def test_unpacked_fc_reads_scale_records_set_numbers_and_level_boundaries(tmp_path, endian, prefix):
    nch, nd, nf = 4, 2, 2
    words = 2 * nch + 1
    header = "nch 4 nd 2 nfmx 2 nwin 1 128 2 128 dr 5.0 20.0 "
    content = bytearray(header.encode().ljust(20 * 4 * words, b" "))
    scales = np.array([1000., 2000., 500., 100.])
    expected = {}
    for level in (1, 2):
        for frequency in (1, 2):
            nsets = 4 - frequency
            content += struct.pack(prefix + "9i", level, frequency, nsets, 0, 0, 0, 0, 0, 0)
            content += struct.pack(prefix + "9f", *scales, 0., 0., 0., 0., 0.)
            values = np.empty((nch, nsets), dtype=complex)
            for segment in range(nsets):
                values[:, segment] = np.arange(nch) + level + 1j * (frequency + segment)
                stored = values[:, segment] * 1000. / scales
                pairs = [component for value in stored for component in (value.real, value.imag)]
                content += struct.pack(prefix + "i8f", segment + 2, *pairs)
            expected[level, frequency] = values
    path = tmp_path / "example.f4"
    path.write_bytes(content)
    fc = FCOutput.read(path, lpack=False, endian=endian)
    for key, values in expected.items():
        np.testing.assert_allclose(fc.load_spectrum(*key), values, rtol=1e-6)
    np.testing.assert_array_equal(fc.i_sets[0], [2, 3, 4])
    np.testing.assert_array_equal(fc.i_sets[1], [2, 3, 4])
    with pytest.raises(ValueError, match="out of range"):
        fc.load_spectrum(0, 1)

    truncated = tmp_path / "truncated.f4"
    truncated.write_bytes(content[:-4])
    with pytest.raises(ValueError, match="Truncated"):
        FCOutput.read(truncated, lpack=False, endian=endian)
