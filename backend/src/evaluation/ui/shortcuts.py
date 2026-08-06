"""The inspector's keyboard map.

Split out of ``main_window.py`` for the 500-LoC budget (§5.17 / issues
#121-#122), and because the map itself is the thing worth reading in one place:
§0.1 budgets ~45 min for 97 tests (~28 s each), which a mouse-only form cannot
reach, so this table *is* the ergonomics of the rating pass.

Shortcuts are installed on the window rather than on the individual widgets so
they fire regardless of which panel has focus. The user-facing crib sheet is
``KEY_HINTS`` in ``constants/user_interface.py``, rendered into the window's
footer — keep the two in step.
"""

from __future__ import annotations

from PySide6.QtGui import QKeySequence, QShortcut

from ..constants.schema import (
    IMAGE_ASP,
    IMAGE_SIMPLE,
    PREF_ASP,
    PREF_SIMPLE,
    PREF_TIE,
    SCORE_MAX,
    SCORE_MIN,
)
from ..constants.user_interface import (
    DISPLAY_PIXEL,
    MODE_BBOX,
    MODE_NAVIGATE,
    MODE_POINT,
    MODE_PROBE,
)


def install(window) -> None:
    """Bind every shortcut on ``window`` (an ``InspectorWindow``)."""

    def bind(sequence: str, handler) -> None:
        QShortcut(QKeySequence(sequence), window).activated.connect(handler)

    # Scoring: digits score the focused panel's coherence — the fast path.
    for value in range(SCORE_MIN, SCORE_MAX + 1):
        bind(str(value), lambda v=value: window._score_focused(v))

    # Defect tags. Ctrl-prefixed so the bare digits stay free for scoring.
    # Ctrl+0 is torn_anatomy through Ctrl+9 geometry_warp — the ten numbered
    # tags in DEFECTS; "Other" has no shortcut (0-9 are all spoken for).
    for index in range(10):
        bind(f"Ctrl+{index}", lambda i=index: window._toggle_defect(i))

    # Focus
    bind("A", lambda: window.grid.set_focus(IMAGE_ASP))
    bind("S", lambda: window.grid.set_focus(IMAGE_SIMPLE))
    bind("Tab", lambda: window.grid.cycle_focus(1))
    bind("Shift+Tab", lambda: window.grid.cycle_focus(-1))

    # Pairwise preference
    bind("[", lambda: window.scoring_panel.set_preference(PREF_ASP))
    bind("]", lambda: window.scoring_panel.set_preference(PREF_SIMPLE))
    bind("=", lambda: window.scoring_panel.set_preference(PREF_TIE))

    # View
    bind("F", window._fit_all)
    bind("P", lambda: window.toolbar.set_display_pixel(
        window.grid.panels[IMAGE_ASP].display_mode() != DISPLAY_PIXEL
    ))

    # Tool modes
    bind("N", lambda: window.toolbar.set_mode(MODE_NAVIGATE))
    bind("B", lambda: window.toolbar.set_mode(MODE_BBOX))
    bind("L", lambda: window.toolbar.set_mode(MODE_POINT))
    bind("V", lambda: window.toolbar.set_mode(MODE_PROBE))

    # Navigation / persistence
    bind("Space", window._go_next)
    bind("Backspace", window._go_back)
    bind("Ctrl+S", window._save_now)

    # Multi-point links (Link mode's chain can hold 2 or more endpoints, so
    # there's no click count that means "done" — these end it explicitly).
    bind("Return", window._finish_link)
    bind("Enter", window._finish_link)  # numpad Enter is a distinct QKeySequence
    bind("Escape", window._cancel_link)
