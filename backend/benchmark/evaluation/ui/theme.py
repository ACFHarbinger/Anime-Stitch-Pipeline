"""The inspector's stylesheets (dark + light) and small styled-widget helpers.

One stylesheet applied to the window, rather than per-widget ``setStyleSheet``
calls scattered through the build code (the old dashboard used unstyled default
Qt chrome with one inline monospace style). Colours come from
``constants.user_interface``, which derives the dark palette from the same two
figure background colours ``bench_anime_stitch.py`` plots with, so embedded
matplotlib canvases sit flush in their panels instead of floating in a
differently-tinted box.

Built from a palette dict rather than as one frozen f-string so the Settings
dialog's dark/light toggle has something to switch between; only the widget
*chrome* changes with the toggle — annotation/overlay colours
(``COL_ACCENT``, ``COL_BBOX``, etc.) stay fixed regardless, since they're drawn
over photographic content, not over the chrome itself (see the note in
``constants/user_interface.py``).
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QFrame, QLabel, QWidget

from ..constants.user_interface import (
    DARK_PALETTE,
    LIGHT_PALETTE,
    SCORE_COLORS,
    THEME_DARK,
    THEME_LIGHT,
)


def _build_stylesheet(p: dict) -> str:
    # The QGraphicsView canvas background is deliberately NOT part of the
    # palette and stays dark in both themes: an image viewer's canvas
    # surround stays neutral/dark regardless of chrome theme in most photo
    # tools, so a panorama's own colours (and any letterboxed empty canvas)
    # aren't visually distorted by a bright white surround.
    canvas_bg = "#08080f"
    return f"""
QWidget {{
    background: {p['bg']};
    color: {p['text']};
    font-size: 12px;
}}
QMainWindow, QDialog {{ background: {p['bg']}; }}

QLabel {{ background: transparent; }}
QLabel[role="heading"] {{ font-size: 14px; font-weight: 600; color: {p['text']}; }}
QLabel[role="subtle"] {{ color: {p['text_dim']}; }}
QLabel[role="panelTitle"] {{
    font-weight: 600;
    padding: 3px 6px;
    background: {p['surface_hi']};
    border: 1px solid {p['border']};
    border-radius: 4px;
}}
QLabel[role="mono"] {{ font-family: monospace; color: {p['text']}; }}

QFrame[role="card"] {{
    background: {p['surface']};
    border: 1px solid {p['border']};
    border-top: 2px solid {p['accent_dim']};
    border-radius: 6px;
}}

QGroupBox {{
    background: {p['surface']};
    border: 1px solid {p['border']};
    border-top: 2px solid {p['accent_dim']};
    border-radius: 6px;
    margin-top: 10px;
    padding: 8px 6px 6px 6px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {p['accent']};
}}

QPushButton {{
    background: {p['surface_hi']};
    border: 1px solid {p['border']};
    border-radius: 5px;
    padding: 5px 11px;
}}
QPushButton:hover {{ border-color: {p['accent']}; color: {p['accent']}; }}
QPushButton:pressed {{ background: {p['accent_dim']}; color: {p['bg']}; }}
QPushButton:checked {{
    background: {p['accent_dim']};
    border-color: {p['accent']};
    color: {p['bg']};
    font-weight: 600;
}}
QPushButton:disabled {{ color: {p['text_dim']}; }}
QPushButton[role="primary"] {{
    background: {p['accent_dim']};
    border-color: {p['accent']};
    color: {p['bg']};
    font-weight: 600;
}}
QPushButton[role="primary"]:hover {{ background: {p['accent']}; }}

QComboBox, QSpinBox, QLineEdit, QPlainTextEdit, QTextEdit {{
    background: {p['surface_hi']};
    border: 1px solid {p['border']};
    border-radius: 5px;
    padding: 4px 6px;
    selection-background-color: {p['accent_dim']};
}}
QComboBox:hover {{ border-color: {p['accent']}; }}
QComboBox QAbstractItemView {{
    background: {p['surface_hi']};
    border: 1px solid {p['border']};
    selection-background-color: {p['accent_dim']};
    outline: none;
}}
QComboBox::drop-down {{ border: none; width: 18px; }}

QCheckBox, QRadioButton {{ spacing: 6px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {p['border']};
    background: {p['surface_hi']};
}}
QCheckBox::indicator {{ border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: 7px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {p['accent']};
    border-color: {p['accent']};
}}

QSlider::groove:horizontal {{
    height: 4px; background: {p['border']}; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {p['accent_dim']}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 13px; margin: -5px 0;
    background: {p['accent']}; border-radius: 7px;
}}

QTabWidget::pane {{
    border: 1px solid {p['border']};
    border-radius: 6px;
    background: {p['surface']};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {p['text_dim']};
    padding: 6px 13px;
    border: 1px solid transparent;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}}
QTabBar::tab:hover {{ color: {p['text']}; }}
QTabBar::tab:selected {{
    background: {p['surface']};
    color: {p['accent']};
    border-color: {p['border']};
    border-bottom-color: {p['surface']};
    font-weight: 600;
}}

QListWidget, QTableWidget, QTreeWidget {{
    background: {p['surface']};
    border: 1px solid {p['border']};
    border-radius: 5px;
    outline: none;
}}
QListWidget::item, QTreeWidget::item {{ padding: 3px 5px; border-radius: 3px; }}
QListWidget::item:selected, QTreeWidget::item:selected,
QTableWidget::item:selected {{ background: {p['accent_dim']}; color: {p['bg']}; }}
QListWidget::item:hover, QTreeWidget::item:hover {{ background: {p['surface_hi']}; }}
QHeaderView::section {{
    background: {p['surface_hi']};
    color: {p['text_dim']};
    border: none;
    border-right: 1px solid {p['border']};
    border-bottom: 1px solid {p['border']};
    padding: 4px 6px;
    font-weight: 600;
}}
QTableWidget {{ gridline-color: {p['border']}; }}
QTableCornerButton::section {{ background: {p['surface_hi']}; border: none; }}

QGraphicsView {{
    background: {canvas_bg};
    border: 1px solid {p['border']};
    border-radius: 5px;
}}

QSplitter::handle {{ background: {p['border']}; }}
QSplitter::handle:horizontal {{ width: 3px; }}
QSplitter::handle:vertical {{ height: 3px; }}
QSplitter::handle:hover {{ background: {p['accent_dim']}; }}

QScrollBar:vertical, QScrollBar:horizontal {{
    background: transparent; border: none;
}}
QScrollBar:vertical {{ width: 10px; }}
QScrollBar:horizontal {{ height: 10px; }}
QScrollBar::handle {{ background: {p['border']}; border-radius: 5px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {p['accent_dim']}; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
    background: none; border: none; height: 0; width: 0;
}}

QToolTip {{
    background: {p['surface_hi']};
    color: {p['text']};
    border: 1px solid {p['accent_dim']};
    padding: 4px;
}}

QStatusBar {{ background: {p['surface']}; border-top: 1px solid {p['border']}; }}
QStatusBar::item {{ border: none; }}
QProgressBar {{
    background: {p['surface_hi']};
    border: 1px solid {p['border']};
    border-radius: 5px;
    text-align: center;
    height: 14px;
}}
QProgressBar::chunk {{ background: {p['accent']}; border-radius: 4px; }}
"""


DARK_STYLESHEET = _build_stylesheet(DARK_PALETTE)
LIGHT_STYLESHEET = _build_stylesheet(LIGHT_PALETTE)
_STYLESHEETS = {THEME_DARK: DARK_STYLESHEET, THEME_LIGHT: LIGHT_STYLESHEET}
_PALETTES = {THEME_DARK: DARK_PALETTE, THEME_LIGHT: LIGHT_PALETTE}

# The handful of widgets that build their own inline stylesheet per-instance
# (score chips, focused-panel title chips) rather than relying on the global
# QSS cascade — tracked here so they can re-derive their colours after a
# theme change instead of being stuck with whichever palette existed when
# they were first built. A module-level "current theme" is a deliberate,
# small piece of mutable global state: this is a single-window desktop app
# with exactly one theme visible at a time, not a library, so there's no
# reentrancy concern it would need to guard against.
_current_theme = THEME_DARK


def apply_theme(widget: QWidget, theme: str = THEME_DARK) -> None:
    global _current_theme
    _current_theme = theme if theme in _STYLESHEETS else THEME_DARK
    widget.setStyleSheet(_STYLESHEETS[_current_theme])


def current_theme() -> str:
    return _current_theme


def current_palette() -> dict:
    """The active theme's palette, for the few widgets that need to re-derive
    an inline (non-QSS-cascaded) style after a theme change."""
    return _PALETTES[_current_theme]


def heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "heading")
    return label


def subtle(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "subtle")
    return label


def mono(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "mono")
    label.setFont(QFont("monospace"))
    return label


def panel_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "panelTitle")
    return label


def card() -> QFrame:
    frame = QFrame()
    frame.setProperty("role", "card")
    return frame


def score_color(score: int) -> QColor:
    return QColor(SCORE_COLORS.get(score, current_palette()["text_dim"]))


def score_chip_style(score: int) -> str:
    """Inline style for a 0-4 score button, so the colour ramp is visible on
    the buttons themselves — at a ~28 s/test pace the colour is read, not the
    digit.

    Built from ``current_palette()`` (not the bare dark constants) since this
    is one of the few widgets that sets its own stylesheet directly rather
    than relying on the global QSS cascade — callers should re-invoke this
    (see ``ScoreRow.refresh_theme()``) after a theme change, or it stays
    whatever it was drawn with.
    """
    palette = current_palette()
    color = SCORE_COLORS.get(score, palette["border"])
    return (
        f"QPushButton {{ border: 1px solid {palette['border']}; border-radius: 4px;"
        f" background: {palette['surface_hi']}; color: {palette['text_dim']}; font-weight: 600; }}"
        f"QPushButton:hover {{ border-color: {color}; color: {color}; }}"
        f"QPushButton:checked {{ background: {color}; color: {palette['bg']}; border-color: {color}; }}"
    )


def severity_chip_style(severity: int) -> str:
    """Inline style for 0-3 defect severity chips."""
    palette = current_palette()
    # 0: absent (muted/neutral), 1: trace (warn/yellow), 2: noticeable (orange), 3: severe (bad/red)
    colors = {
        0: palette["good"],
        1: palette["warn"],
        2: "#ff9f68",
        3: palette["bad"],
    }
    color = colors.get(severity, palette["border"])
    if severity == 0:
        return (
            f"QPushButton {{ border: 1px solid {palette['border']}; border-radius: 4px;"
            f" background: {palette['surface_hi']}; color: {palette['text_dim']}; font-size: 11px; padding: 2px 6px; }}"
            f"QPushButton:hover {{ border-color: {color}; color: {color}; }}"
            f"QPushButton:checked {{ background: {color}; color: {palette['bg']}; border-color: {color}; font-weight: 600; }}"
        )
    return (
        f"QPushButton {{ border: 1px solid {palette['border']}; border-radius: 4px;"
        f" background: {palette['surface_hi']}; color: {palette['text_dim']}; font-size: 11px; padding: 2px 6px; }}"
        f"QPushButton:hover {{ border-color: {color}; color: {color}; }}"
        f"QPushButton:checked {{ background: {color}; color: {palette['bg']}; border-color: {color}; font-weight: 600; }}"
    )

