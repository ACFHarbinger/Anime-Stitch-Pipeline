"""The inspector's top control bar: interaction tool, display mode, which
comparators are visible, layout, view locking, and zoom.

Split out of ``main_window.py`` so the window assembles named pieces rather than
building four layouts inline. The comparator visibility checkboxes are rebuilt
per test, because which comparators exist varies (all 97 have Overmix, only one
has a Hugin panorama, 55 have ground truth).
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QWidget,
)

from ..constants.schema import COMPARATOR_TITLES
from ..constants.user_interface import (
    DISPLAY_PIXEL,
    DISPLAY_RAW,
    LAYOUTS,
    TOOL_MODES,
)


class InspectorToolbar(QWidget):
    modeChanged = Signal(str)
    displayModeChanged = Signal(str)
    layoutChanged = Signal(str)
    visibilityChanged = Signal(list)
    lockToggled = Signal(bool)
    fitRequested = Signal()
    zoomRequested = Signal(float)
    queuePanelToggled = Signal(bool)
    sidePanelToggled = Signal(bool)
    saveRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checks: dict[str, QCheckBox] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._tool_combo = QComboBox()
        for key, label, hint in TOOL_MODES:
            self._tool_combo.addItem(label, key)
            self._tool_combo.setItemData(self._tool_combo.count() - 1, hint, 3)  # ToolTipRole
        self._tool_combo.currentIndexChanged.connect(
            lambda _i: self.modeChanged.emit(self._tool_combo.currentData())
        )
        layout.addWidget(QLabel("Tool"))
        layout.addWidget(self._tool_combo)

        self._display_check = QCheckBox("Pixel values")
        self._display_check.setToolTip(
            "Show a hover magnifier with numeric RGB values at any zoom; also overlays "
            "a per-pixel grid across the visible area once zoomed in far enough"
        )
        self._display_check.toggled.connect(
            lambda checked: self.displayModeChanged.emit(DISPLAY_PIXEL if checked else DISPLAY_RAW)
        )
        layout.addWidget(self._display_check)

        self._layout_combo = QComboBox()
        for key, label, hint in LAYOUTS:
            self._layout_combo.addItem(label, key)
            self._layout_combo.setItemData(self._layout_combo.count() - 1, hint, 3)
        self._layout_combo.currentIndexChanged.connect(
            lambda _i: self.layoutChanged.emit(self._layout_combo.currentData())
        )
        layout.addWidget(QLabel("Layout"))
        layout.addWidget(self._layout_combo)

        self._lock_check = QCheckBox("Lock zoom + pan")
        self._lock_check.setChecked(True)
        self._lock_check.setToolTip(
            "Mirror magnification and viewport centre across panels. Relative to each "
            "panel's fitted view, since comparators have different canvas sizes."
        )
        self._lock_check.toggled.connect(self.lockToggled.emit)
        layout.addWidget(self._lock_check)

        fit_btn = QPushButton("Fit")
        fit_btn.setToolTip("Fit all visible panels (F)")
        fit_btn.clicked.connect(self.fitRequested.emit)
        layout.addWidget(fit_btn)
        zoom_out = QPushButton("−")
        zoom_out.setFixedWidth(28)
        zoom_out.clicked.connect(lambda: self.zoomRequested.emit(1 / 1.6))
        zoom_in = QPushButton("+")
        zoom_in.setFixedWidth(28)
        zoom_in.clicked.connect(lambda: self.zoomRequested.emit(1.6))
        layout.addWidget(zoom_out)
        layout.addWidget(zoom_in)

        layout.addWidget(QLabel("Show"))
        self._checks_host = QWidget()
        self._checks_layout = QHBoxLayout(self._checks_host)
        self._checks_layout.setContentsMargins(0, 0, 0, 0)
        self._checks_layout.setSpacing(8)

        self._checks_scroll = QScrollArea()
        self._checks_scroll.setWidget(self._checks_host)
        self._checks_scroll.setWidgetResizable(True)
        self._checks_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._checks_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._checks_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._checks_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(self._checks_scroll, stretch=1)

        # Collapsible sidebars: free up horizontal space for the panel grid
        # itself. Qt's QSplitter remembers a hidden child's proportion and
        # restores it on show(), so plain setVisible() toggling is enough —
        # no manual size bookkeeping needed on either side of the toggle.
        self._queue_toggle = QPushButton("◧ Tests")
        self._queue_toggle.setCheckable(True)
        self._queue_toggle.setChecked(True)
        self._queue_toggle.setToolTip("Show/hide the test queue sidebar")
        self._queue_toggle.toggled.connect(self.queuePanelToggled.emit)
        layout.addWidget(self._queue_toggle)

        self._side_toggle = QPushButton("Scoring ◨")
        self._side_toggle.setCheckable(True)
        self._side_toggle.setChecked(True)
        self._side_toggle.setToolTip("Show/hide the scoring sidebar")
        self._side_toggle.toggled.connect(self.sidePanelToggled.emit)
        layout.addWidget(self._side_toggle)

        # Ratings only reach disk from here (or Ctrl+S). The label carries the
        # unsaved marker so the state is visible without reading the footer.
        self._save_btn = QPushButton("Save ratings")
        self._save_btn.setToolTip("Write ratings to the evaluations file (Ctrl+S)")
        self._save_btn.clicked.connect(self.saveRequested.emit)
        layout.addWidget(self._save_btn)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def minimumSizeHint(self) -> QSize:  # noqa: D102 - Qt override
        return QSize(400, 32)

    def set_dirty(self, dirty: bool) -> None:
        """Mark whether there are unsaved ratings."""
        self._save_btn.setText("Save ratings *" if dirty else "Save ratings")

    def set_comparators(self, available: list[str], visible: list[str]) -> None:
        for check in self._checks.values():
            check.setParent(None)
        self._checks = {}
        for key in available:
            check = QCheckBox(COMPARATOR_TITLES.get(key, key))
            check.setChecked(key in visible)
            check.toggled.connect(lambda _c: self._emit_visibility())
            self._checks_layout.addWidget(check)
            self._checks[key] = check

    def _emit_visibility(self) -> None:
        selected = [key for key, check in self._checks.items() if check.isChecked()]
        if not selected:
            # Refuse to hide everything — an empty grid gives the user no way
            # back except reaching for a checkbox they can no longer see next
            # to any content.
            return
        self.visibilityChanged.emit(selected)

    def set_display_pixel(self, enabled: bool) -> None:
        self._display_check.setChecked(enabled)

    def set_mode(self, key: str) -> None:
        index = self._tool_combo.findData(key)
        if index >= 0:
            self._tool_combo.setCurrentIndex(index)

    def mode(self) -> str:
        return self._tool_combo.currentData()
