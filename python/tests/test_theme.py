"""Regression tests for pyqtgraph's theme-independent plot styling."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget

from ui.theme import configure_pyqtgraph_theme, pg_background, pg_foreground, subtle_pen


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_plot_colours_do_not_follow_a_dark_widget_palette():
    _app()
    widget = QWidget()
    palette = widget.palette()
    palette.setColor(QPalette.ColorRole.Base, QColor("black"))
    palette.setColor(QPalette.ColorRole.Text, QColor("white"))
    palette.setColor(QPalette.ColorRole.Mid, QColor("white"))
    widget.setPalette(palette)

    assert pg_background(widget) == QColor("white")
    assert pg_foreground(widget) == QColor("black")
    assert subtle_pen(widget).color() == QColor(120, 120, 120, 180)


def test_new_plot_widget_uses_fixed_white_background():
    app = _app()
    original_palette = QPalette(app.palette())
    dark_palette = QPalette(original_palette)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor("black"))
    dark_palette.setColor(QPalette.ColorRole.Text, QColor("white"))
    app.setPalette(dark_palette)
    try:
        configure_pyqtgraph_theme()
        plot = pg.PlotWidget()

        assert plot.backgroundBrush().color() == QColor("white")
        assert plot.getAxis("left").pen().color() == QColor("black")
        plot.close()
    finally:
        app.setPalette(original_palette)
