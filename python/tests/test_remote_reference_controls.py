"""Parameter window lifecycle tests using an MDI host and synthetic UI state."""

from PySide6.QtWidgets import QApplication, QMdiArea

from output_tools.remote_reference import RemoteReferenceWidget


def test_parameters_open_in_foreground_after_mdi_host_is_shown(qtbot):
    host = QMdiArea()
    qtbot.addWidget(host)
    widget = RemoteReferenceWidget()
    window = host.addSubWindow(widget)
    dialog = widget._control_dialog
    assert not dialog.isVisible()

    host.resize(1000, 800)
    host.show()
    window.show()
    window.showMaximized()
    host.setActiveSubWindow(window)
    qtbot.waitUntil(lambda: dialog.isVisible() and QApplication.activeWindow() is dialog)

    # Closing the nonmodal controls is intentional; hiding/showing the result
    # window later must not reopen them or steal focus again.
    dialog.close()
    widget.hide()
    widget.show()
    QApplication.processEvents()
    assert not dialog.isVisible()


def test_parameter_button_restores_minimized_controls(qtbot):
    widget = RemoteReferenceWidget()
    qtbot.addWidget(widget)
    widget.show()
    dialog = widget._control_dialog
    qtbot.waitUntil(dialog.isVisible)
    dialog.showMinimized()
    assert dialog.isMinimized()

    widget.tune_btn.click()
    qtbot.waitUntil(lambda: dialog.isVisible() and not dialog.isMinimized())
    qtbot.waitUntil(lambda: QApplication.activeWindow() is dialog)
