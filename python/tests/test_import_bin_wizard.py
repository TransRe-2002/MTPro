import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel

from ui.import_bin_wizard import ChannelConfigPage


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_channel_config_unsynced_direction_combos_are_enabled():
    _app()
    page = ChannelConfigPage()

    page._dir_sync_cb.setChecked(False)
    page._pair_combos[4].setCurrentIndex(1)
    page._dir_combos[4].setCurrentText("S (负)")
    page._pair_combos[5].setCurrentIndex(2)
    page._dir_combos[5].setCurrentText("W (负)")

    assert page._dir_combos[4].isEnabled()
    assert not page._dir_combos[4].isHidden()
    assert page._dir_combos[5].isEnabled()
    assert not page._dir_combos[5].isHidden()

    configs = page._collect_configs()
    assert configs[4]["direction"] == "S"
    assert configs[5]["direction"] == "W"
    page.close()


def test_channel_config_synced_direction_combos_show_global_values_disabled():
    _app()
    page = ChannelConfigPage()

    page._dir_sync_cb.setChecked(True)
    page._ex_dir_combo.setCurrentText("S")
    page._ey_dir_combo.setCurrentText("W")
    page._pair_combos[4].setCurrentIndex(1)
    page._pair_combos[5].setCurrentIndex(2)

    assert not page._dir_combos[4].isHidden()
    assert not page._dir_combos[4].isEnabled()
    assert page._dir_combos[4].currentText() == "S (负)"
    assert not page._dir_combos[5].isHidden()
    assert not page._dir_combos[5].isEnabled()
    assert page._dir_combos[5].currentText() == "W (负)"

    configs = page._collect_configs()
    assert configs[4]["direction"] == "S"
    assert configs[5]["direction"] == "W"
    page.close()


def test_channel_config_has_direction_header_without_description_header():
    _app()
    page = ChannelConfigPage()

    labels = {label.text() for label in page.findChildren(QLabel)}

    assert "电极方向" in labels
    assert "说明" not in labels
    page.close()
