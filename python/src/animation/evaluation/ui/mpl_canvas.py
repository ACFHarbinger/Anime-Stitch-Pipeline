"""Qt-embeddable matplotlib canvas with a navigation toolbar.

The old ``PlotCanvas`` had no toolbar at all, so a plot could not be panned,
zoomed or saved — on a 16-frame alignment chart that is tolerable, on a
per-seam strip with 30 bars it is not. The dark theme itself lives in
``logic/figure_theme.py`` so figure builders never import Qt.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..constants.user_interface import COL_BORDER, COL_SURFACE, COL_TEXT
from ..logic.figure_theme import style_axis, themed_figure  # re-exported for callers

__all__ = ["PlotCanvas", "PlotPane", "themed_figure", "style_axis"]


class PlotCanvas(FigureCanvasQTAgg):
    """Thin wrapper so callers never import matplotlib's Qt backend
    directly — every figure builder returns a ``Figure`` and hands it here."""

    def __init__(self, fig: Figure, parent=None):
        super().__init__(fig)
        if parent is not None:
            self.setParent(parent)


class PlotPane(QWidget):
    """A canvas plus its navigation toolbar, styled to match the surrounding
    panels instead of matplotlib's default light chrome."""

    def __init__(self, fig: Figure, parent=None):
        super().__init__(parent)
        self.canvas = PlotCanvas(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setStyleSheet(
            f"QToolBar {{ background: {COL_SURFACE}; border: none;"
            f" border-bottom: 1px solid {COL_BORDER}; }}"
            f"QToolButton {{ background: transparent; color: {COL_TEXT}; padding: 3px; }}"
            f"QToolButton:hover {{ background: #29294a; border-radius: 3px; }}"
            f"QLabel {{ color: {COL_TEXT}; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)
