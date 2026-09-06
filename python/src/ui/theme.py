from __future__ import annotations

import matplotlib
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget
import pyqtgraph as pg


# pyqtgraph canvases are intentionally kept in the application's original
# light style.  Unlike ordinary Qt widgets, signal plots need to remain easy
# to compare across machines and screenshots, regardless of the OS theme.
_PG_BACKGROUND = QColor("white")
_PG_FOREGROUND = QColor("black")
_PG_SUBTLE = QColor(120, 120, 120)


def configure_matplotlib_fonts() -> None:
    """配置 matplotlib 中文字体，优先使用方正黑体。"""
    matplotlib.rcParams["font.sans-serif"] = [
        "FZHei-B01", "Microsoft YaHei", "Noto Sans CJK JP", "Sarasa UI SC",
    ] + matplotlib.rcParams["font.sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def _palette(widget: QWidget | None = None) -> QPalette:
    if widget is not None:
        return widget.palette()
    app = QApplication.instance()
    if app is not None:
        return app.palette()
    return QPalette()


def is_dark_theme(widget: QWidget | None = None) -> bool:
    palette = _palette(widget)
    return palette.color(QPalette.ColorRole.Window).lightness() < 128


def palette_color(role: QPalette.ColorRole, widget: QWidget | None = None) -> QColor:
    return _palette(widget).color(role)


def pg_background(widget: QWidget | None = None) -> QColor:
    """Return the fixed background colour for pyqtgraph PlotWidgets.

    ``widget`` is kept for API compatibility with existing callers; the
    widget palette must not affect a plot canvas.
    """
    return QColor(_PG_BACKGROUND)


def pg_foreground(widget: QWidget | None = None) -> QColor:
    """Return the fixed foreground colour for pyqtgraph PlotWidgets."""
    return QColor(_PG_FOREGROUND)


def configure_pyqtgraph_theme(widget: QWidget | None = None) -> None:
    """Configure pyqtgraph to use its fixed, theme-independent light style."""
    pg.setConfigOptions(antialias=True)
    pg.setConfigOption("background", pg_background(widget))
    pg.setConfigOption("foreground", pg_foreground(widget))


def transparent_legend_brush(widget: QWidget | None = None, alpha: int = 230) -> pg.QtGui.QBrush:
    color = QColor(pg_background(widget))
    color.setAlpha(alpha)
    return pg.mkBrush(color)


def subtle_pen(widget: QWidget | None = None, alpha: int = 180, width: float = 1.0):
    """Return the fixed subtle grid/border pen used inside PlotWidgets."""
    color = QColor(_PG_SUBTLE)
    color.setAlpha(alpha)
    return pg.mkPen(color, width=width)


def _mpl_color(color: QColor) -> tuple[float, float, float, float]:
    return color.redF(), color.greenF(), color.blueF(), color.alphaF()


def configure_matplotlib_figure(figure, widget: QWidget | None = None) -> None:
    bg = QColor("white")
    fg = QColor("black")
    mid = QColor(120, 120, 120)
    figure.patch.set_facecolor(_mpl_color(bg))
    figure.patch.set_edgecolor(_mpl_color(bg))
    for axis in figure.axes:
        axis.set_facecolor(_mpl_color(bg))
        axis.tick_params(colors=_mpl_color(fg))
        axis.xaxis.label.set_color(_mpl_color(fg))
        axis.yaxis.label.set_color(_mpl_color(fg))
        axis.title.set_color(_mpl_color(fg))
        for spine in axis.spines.values():
            spine.set_color(_mpl_color(mid))
        legend = axis.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            frame.set_facecolor(_mpl_color(bg))
            frame.set_edgecolor(_mpl_color(mid))
            for text in legend.get_texts():
                text.set_color(_mpl_color(fg))


def configure_matplotlib_canvas(canvas, figure, widget: QWidget | None = None) -> None:
    configure_matplotlib_figure(figure, widget)
    bg = QColor("white").name()
    canvas.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    canvas.setAutoFillBackground(True)
    canvas.setStyleSheet(f"background-color: {bg};")
