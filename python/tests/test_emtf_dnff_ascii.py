from pathlib import Path


def test_runner_passes_command_args(monkeypatch, tmp_path):
    from emtf.runner import Workspace, run

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)

    ws = Workspace(cleanup=False, prefix="test_emtf_runner_")
    ws.path = tmp_path
    run(Path("/bin/dnff"), stdin="x\n", workspace=ws, args=["-A"])

    assert calls[0][0] == ["/bin/dnff", "-A"]
    assert calls[0][1]["cwd"] == str(tmp_path)
    assert calls[0][1]["input"] == "x\n"


def test_decset_and_pwset_match_fortran_layout():
    from emtf.config import render_decset, render_pwset
    from emtf.types import DecimationLevel

    levels = [
        DecimationLevel(nwin=128, window_length=32.0, decimation_factor=1, nfilt=1, fil_1=1.0),
        DecimationLevel(
            nwin=128,
            window_length=32.0,
            decimation_factor=4,
            nfilt=4,
            fil_1=0.2154,
            fil_2=0.1911,
            fil_3=0.1307,
            fil_4=0.0705,
        ),
    ]

    decset_lines = render_decset(levels).splitlines()
    assert decset_lines[0].startswith("2  0")
    assert decset_lines[1].split()[-1] == "1"
    assert decset_lines[2].split() == ["1"]
    assert decset_lines[3].split()[-1] == "4"
    assert decset_lines[4].split() == ["0.2154", "0.1911", "0.1307", "0.0705"]

    assert render_pwset(2, 5, ar_order=3).splitlines() == ["2", "3", "3"]


def test_decset_rejects_single_level_for_dnff():
    import pytest

    from emtf.config import render_decset
    from emtf.types import DecimationLevel

    with pytest.raises(ValueError, match="at least two"):
        render_decset([DecimationLevel()])


def test_single_tranmt_control_terminates_after_one_run():
    from emtf.config import render_tranmt_control

    text = render_tranmt_control([
        ("TEST", "CF/options_SS.cfg", [(["TEST.f4"], 4)], False),
    ])

    assert text.splitlines() == [
        "TEST",
        "CF/options_SS.cfg",
        "1",
        "1  4",
        "TEST.f4",
        "n",
    ]


def test_remote_tranmt_control_keeps_both_fc_groups_in_one_run():
    from emtf.config import render_tranmt_control

    text = render_tranmt_control([
        (
            "TEST_RR",
            "CF/options_RR.cfg",
            [(["TEST.f4"], 4), (["TEST_RR.f2"], 2)],
            False,
        ),
    ])

    assert text.splitlines() == [
        "TEST_RR",
        "CF/options_RR.cfg",
        "2",
        "1  4",
        "TEST.f4",
        "1  2",
        "TEST_RR.f2",
        "n",
    ]
