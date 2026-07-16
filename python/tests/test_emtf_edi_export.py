import numpy as np

from emtf.io import BandedResult, format_edi


def _banded_result(with_tipper: bool = False) -> BandedResult:
    kwargs = {}
    if with_tipper:
        kwargs.update(
            tzx=np.array([0.1 + 0.2j, 0.2 + 0.3j], dtype=np.complex128),
            tzy=np.array([0.3 + 0.4j, 0.4 + 0.5j], dtype=np.complex128),
            tzx_se=np.array([0.01, 0.02], dtype=np.float64),
            tzy_se=np.array([0.03, 0.04], dtype=np.float64),
        )

    return BandedResult(
        periods=np.array([10.0, 100.0], dtype=np.float64),
        zxx=np.array([0.1 + 0.2j, 0.2 + 0.3j], dtype=np.complex128),
        zxy=np.array([1.0 + 2.0j, 2.0 + 3.0j], dtype=np.complex128),
        zyx=np.array([3.0 + 4.0j, 4.0 + 5.0j], dtype=np.complex128),
        zyy=np.array([0.3 + 0.4j, 0.4 + 0.5j], dtype=np.complex128),
        zxx_se=np.array([0.01, 0.02], dtype=np.float64),
        zxy_se=np.array([0.03, 0.04], dtype=np.float64),
        zyx_se=np.array([0.05, 0.06], dtype=np.float64),
        zyy_se=np.array([0.07, 0.08], dtype=np.float64),
        n_observations=np.array([12, 34], dtype=np.int32),
        station_name="TEST",
        mode="rr",
        lat=11.5,
        lon=22.5,
        elev=123.0,
        declination=1.25,
        **kwargs,
    )


def test_format_edi_uses_banded_metadata_and_core_blocks():
    text = format_edi(_banded_result())

    assert 'DATAID="TEST"' in text
    assert "Processed by atm_rpc EMTF in RR mode." in text
    assert "LAT=11.500000" in text
    assert "LONG=22.500000" in text
    assert "ELEV=123.00" in text
    assert "NFREQ=2" in text
    assert ">FREQ ORDER=DEC  //          2" in text
    assert ">ZXYR  //          2" in text
    assert ">ZXY.VAR  //          2" in text
    assert text.rstrip().endswith(">END")


def test_export_to_folder_writes_edi_and_z_with_tipper(tmp_path):
    result = _banded_result(with_tipper=True)

    edi_path, z_path = result.export_to_folder(tmp_path)

    assert edi_path.name == "TEST.edi"
    assert z_path.name == "Z.TEST.zrr"
    edi_text = edi_path.read_text(encoding="ascii")
    z_text = z_path.read_text(encoding="ascii")
    assert ">TXR.EXP  //          2" in edi_text
    assert ">TYVAR.EXP  //          2" in edi_text
    assert "number of channels   5" in z_text
    assert "Robust Remote Reference" in z_text


def test_phase_uses_emtf_matlab_folded_values():
    result = BandedResult(
        periods=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        zxx=np.zeros(3, dtype=np.complex128),
        zxy=np.array([-1.0 + 1.0j, -1.0 - 1.0j, 1.0 + 1.0j], dtype=np.complex128),
        zyx=np.array([-1.0 - 1.0j, -1.0 + 1.0j, 1.0 - 1.0j], dtype=np.complex128),
        zyy=np.zeros(3, dtype=np.complex128),
        zxx_se=np.zeros(3, dtype=np.float64),
        zxy_se=np.zeros(3, dtype=np.float64),
        zyx_se=np.zeros(3, dtype=np.float64),
        zyy_se=np.zeros(3, dtype=np.float64),
    )

    assert np.allclose(result.phi_xy, [-45.0, 45.0, 45.0])
    assert np.allclose(result.phi_yx, [45.0, -45.0, -45.0])
