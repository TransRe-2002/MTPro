import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")


def _configure_matplotlib_cache() -> None:
    """Give Matplotlib a stable writable cache before UI modules import it."""
    if os.environ.get("MPLCONFIGDIR"):
        return
    if os.name == "nt":
        cache_root = os.environ.get("LOCALAPPDATA")
        root = Path(cache_root) if cache_root else Path.home() / "AppData" / "Local"
    else:
        cache_root = os.environ.get("XDG_CACHE_HOME")
        root = Path(cache_root) if cache_root else Path.home() / ".cache"
    user_suffix = str(os.getuid()) if hasattr(os, "getuid") else "user"
    candidates = (
        root / "mtpro" / "matplotlib",
        Path(tempfile.gettempdir()) / f"mtpro-{user_suffix}" / "matplotlib",
    )
    for target in candidates:
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(target, os.W_OK):
            os.environ["MPLCONFIGDIR"] = str(target)
            return


_configure_matplotlib_cache()

if __name__ == "__main__" and "--verify-emtf-rs" in sys.argv[1:]:
    import emtf_rs
    import rust_lib
    from output_tools.shared.emtf import native_emtf_rs_available

    if not native_emtf_rs_available() or not callable(getattr(emtf_rs, "estimate_emtf_bands", None)):
        raise SystemExit("emtf-rs backend is unavailable")
    if not callable(getattr(rust_lib, "minmax_decimate_rs", None)):
        raise SystemExit("built-in Rust extension is unavailable")
    print("emtf-rs and built-in Rust extensions are available")
    raise SystemExit(0)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QObject

from ui.mainwindow import MainWindow
from base.logging_setup import install_standard_stream_logging
from base.data_manager import DataManager

class Main(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_manager = DataManager(self)
        self.window = MainWindow(self.data_manager)

    def main(self):
        self.window.showMaximized()
        app.exec()

if __name__ == "__main__":
    app = QApplication([])
    install_standard_stream_logging()
    main = Main(app)
    main.main()
    sys.exit()
