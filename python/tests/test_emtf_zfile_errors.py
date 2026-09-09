import numpy as np
import pytest

from emtf.io.zfile import ZFileData


def test_zfile_error_propagation_matches_emtf_matlab_formulas():
    zdata = ZFileData(
        tf=np.array([[[1 + 0j], [3 + 4j]], [[3 + 4j], [1 + 0j]]], dtype=np.complex128),
        sig_s=np.array([[[8.0], [0.0]], [[0.0], [18.0]]], dtype=np.complex128),
        sig_e=np.array([[[2.0], [0.0]], [[0.0], [8.0]]], dtype=np.complex128),
        periods=np.array([10.0], dtype=np.float64),
        nch=4,
        nche=2,
        nbt=1,
        chid=["HX", "HY", "EX", "EY"],
    )

    zerr_xy = 18.0 * 2.0
    zerr_yx = 8.0 * 8.0
    rho = 0.2 * 10.0 * 25.0

    assert np.allclose(zdata.Zxy_err(), [np.sqrt(zerr_xy / 2.0)])
    assert np.allclose(zdata.Zyx_err(), [np.sqrt(zerr_yx / 2.0)])
    assert np.allclose(zdata.rho_xy_err(), [np.sqrt(zerr_xy * 10.0 * rho * 2.0 / 5.0)])
    assert np.allclose(zdata.rho_yx_err(), [np.sqrt(zerr_yx * 10.0 * rho * 2.0 / 5.0)])
    assert np.allclose(zdata.phi_xy_err_deg(), [np.degrees(np.sqrt(zerr_xy / 2.0) / 5.0)])
    assert np.allclose(zdata.phi_yx_err_deg(), [np.degrees(np.sqrt(zerr_yx / 2.0) / 5.0)])


def test_zfile_impedance_properties_use_emtf_matlab_phase_convention():
    zdata = ZFileData(
        tf=np.array([[[0 + 0j], [1 + 1j]], [[-1 - 1j], [0 + 0j]]], dtype=np.complex128),
        periods=np.array([1.0], dtype=np.float64),
        nch=4,
        nche=2,
        nbt=1,
        chid=["HX", "HY", "EX", "EY"],
    )

    assert np.allclose(zdata.Zxy, [-1.0 - 1.0j])
    assert np.allclose(zdata.Zyx, [1.0 + 1.0j])
    assert np.allclose(zdata.phase_xy_deg(), [45.0])
    assert np.allclose(zdata.phase_yx_deg(), [45.0])


def test_zfile_reader_reports_invalid_channel_counts_without_numpy_dimension_error(tmp_path):
    z_path = tmp_path / "invalid.zrr"
    z_path.write_text(
        "header 1\n"
        "header 2\n"
        "Robust Remote Reference\n"
        "station : TEST\n"
        "coordinate 0 0 declination 0\n"
        "number of channels 0 number of frequencies 1\n"
        "orientations and tilts of each channel\n"
    )

    with pytest.raises(ValueError, match="expected at least four channels, got 0"):
        ZFileData.read(z_path)
