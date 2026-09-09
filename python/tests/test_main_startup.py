from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import main


def test_matplotlib_cache_uses_configured_cache_root(tmp_path, monkeypatch):
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    main._configure_matplotlib_cache()

    assert os.environ["MPLCONFIGDIR"] == str(tmp_path / "mtpro" / "matplotlib")
    assert (tmp_path / "mtpro" / "matplotlib").is_dir()
