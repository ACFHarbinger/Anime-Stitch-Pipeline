"""The scoring form: per-dimension 0-4 sub-scores per comparator, a pairwise
ASP-vs-Simple preference with confidence, defect-taxonomy tags, and free-text
notes.

Designed for the pace §0.1 budgets — ~45 min for 97 tests, ~28 s each — which a
mouse-only form cannot hit. Every control is reachable from the keyboard
(handled in ``main_window.py``, which owns the shortcuts), the 0-4 buttons carry
the score colour ramp so a row is read by colour rather than by digit, and only
the coherence row is required: the other four dimensions are optional
diagnostics, so a fast pass stays fast.

The coherence row of ASP and Simple *is* the top-level ``asp``/``simple`` score
``bench_anime_stitch.py`` reads — ``RatingEntry.set_score`` keeps the mirror in
sync, so there's no separate "overall" control to forget to fill in.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..constants.schema import (
    COMPARATOR_TITLES,
    CONFIDENCE_LABELS,
    DEFECTS,
    DIM_COHERENCE,
    DIMENSIONS,
    IMAGE_ASP,
    PREFERENCES,
    SCORE_LABELS,
    SCORE_MAX,
    SCORE_MIN,
    SEVERITY_LABELS,
)
from ..other.schema import RatingEntry
from .theme import current_palette, score_chip_style, severity_chip_style, subtle


class ScoreRow(QWidget):
    """A 0-4 button row for one (image, dimension) pair."""

    scoreChanged = Signal(str, str, object)  # image key, dimension key, score or None

    def __init__(self, image_key: str, dimension: str, label: str, hint: str = "", parent=None):
        super().__init__(parent)
        self.image_key = image_key
        self.dimension = dimension
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        name = QLabel(label)
        name.setMinimumWidth(94)
        if hint:
            name.setToolTip(hint)
        self._name_label = name
        self._dim_is_optional = dimension != DIM_COHERENCE
        if self._dim_is_optional:
            name.setStyleSheet(f"color: {current_palette()['text_dim']};")
        layout.addWidget(name)
        self._group = QButtonGroup(self)
        self._group.setExclusive(False)
        self._buttons: list[QPushButton] = []
        for value in range(SCORE_MIN, SCORE_MAX + 1):
            btn = QPushButton(str(value))
            btn.setCheckable(True)
            btn.setFixedSize(30, 24)
            btn.setStyleSheet(score_chip_style(value))
            btn.setToolTip(SCORE_LABELS.get(value, ""))
            btn.clicked.connect(lambda _checked, v=value: self._on_clicked(v))
            self._group.addButton(btn, value)
            layout.addWidget(btn)
            self._buttons.append(btn)
        layout.addStretch(1)
        self._score: int | None = None

    def _on_clicked(self, value: int) -> None:
        # Clicking the active score clears it, so a mis-hit is one click to
        # undo rather than a value you can't remove.
        self.set_score(None if self._score == value else value)
        self.scoreChanged.emit(self.image_key, self.dimension, self._score)

    def set_score(self, score: int | None) -> None:
        self._score = score
        for value, btn in enumerate(self._buttons):
            btn.setChecked(value == score)

    def score(self) -> int | None:
        return self._score

    def refresh_theme(self) -> None:
        """Re-derive this row's inline styles (the score chips' colour ramp
        and the optional-dimension label dimming) from the now-current
        palette — both are built by ``score_chip_style()``/direct palette
        lookups rather than cascaded QSS."""
        if self._dim_is_optional:
            self._name_label.setStyleSheet(f"color: {current_palette()['text_dim']};")
        for value, btn in enumerate(self._buttons):
            btn.setStyleSheet(score_chip_style(value))


class ImageScoreBlock(QGroupBox):
    """All dimension rows for one comparator."""

    scoreChanged = Signal(str, str, object)

    def __init__(self, image_key: str, parent=None):
        super().__init__(COMPARATOR_TITLES.get(image_key, image_key), parent)
        self.image_key = image_key
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setSpacing(2)
        self.rows: dict[str, ScoreRow] = {}
        for dim_key, dim_label, dim_hint in DIMENSIONS:
            row = ScoreRow(image_key, dim_key, dim_label, dim_hint)
            row.scoreChanged.connect(self.scoreChanged.emit)
            layout.addWidget(row)
            self.rows[dim_key] = row

    def load(self, entry: RatingEntry) -> None:
        for dim_key, row in self.rows.items():
            row.set_score(entry.score(self.image_key, dim_key))

    def set_score(self, dimension: str, score: int | None) -> None:
        row = self.rows.get(dimension)
        if row is not None:
            row.set_score(score)

    def set_detailed(self, detailed: bool) -> None:
        """Show or hide the four optional dimension rows.

        Collapsed is the default: only ``coherence`` is required, and four
        comparators x five dimensions is 20 rows that don't fit beside the
        images and aren't needed for the fast pass. A row holding a score stays
        visible regardless, so collapsing can never hide recorded data.
        """
        for dim_key, row in self.rows.items():
            if dim_key == DIM_COHERENCE:
                continue
            row.setVisible(detailed or row.score() is not None)

    def refresh_theme(self) -> None:
        for row in self.rows.values():
            row.refresh_theme()


class DefectSeverityRow(QWidget):
    """One ordinal 0--3 severity row for the selected comparator output, styled with crisp color-coded score chips."""

    severityChanged = Signal(str, int)

    def __init__(self, defect: str, title: str, hint: str, parent=None):
        super().__init__(parent)
        self.defect = defect
        self._severity: int | None = 0
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        label = QLabel(title)
        label.setMinimumWidth(125)
        label.setToolTip(hint)
        self._title_label = label
        layout.addWidget(label)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[int, QPushButton] = {}
        for value, text in ((0, "0 absent"), *sorted(SEVERITY_LABELS.items())):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setToolTip("No defect (0)" if value == 0 else f"Severity {value}: {text}")
            btn.setStyleSheet(severity_chip_style(value))
            btn.clicked.connect(lambda _checked, v=value: self._on_clicked(v))
            self._group.addButton(btn, value)
            layout.addWidget(btn)
            self._buttons[value] = btn

        self._legacy_label = subtle("legacy: ungraded")
        self._legacy_label.hide()
        layout.addWidget(self._legacy_label)
        layout.addStretch(1)

    def _on_clicked(self, severity: int) -> None:
        self.set_severity(severity)
        self.severityChanged.emit(self.defect, severity)

    def set_severity(self, severity: int | None) -> None:
        self._severity = severity
        for value, button in self._buttons.items():
            button.setChecked(value == severity)
        self._legacy_label.setVisible(severity is None)

    def refresh_theme(self) -> None:
        for value, button in self._buttons.items():
            button.setStyleSheet(severity_chip_style(value))


class ScoringPanel(QWidget):
    """The whole feedback form for one test."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entry = RatingEntry()
        self._loading = False
        self.blocks: dict[str, ImageScoreBlock] = {}
        self._severity_image = IMAGE_ASP

        self._blocks_host = QWidget()
        self._blocks_layout = QVBoxLayout(self._blocks_host)
        self._blocks_layout.setContentsMargins(0, 0, 0, 0)
        self._blocks_layout.setSpacing(6)

        self._detail_toggle = QPushButton("Show all dimensions")
        self._detail_toggle.setCheckable(True)
        self._detail_toggle.setToolTip(
            "Only coherence is required. The sharpness/framing/seams/colour rows are "
            "optional diagnostics — expand them when a score needs explaining."
        )
        self._detail_toggle.toggled.connect(self._on_detail_toggled)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(8)
        inner_layout.addWidget(self._detail_toggle)
        inner_layout.addWidget(self._blocks_host)
        inner_layout.addWidget(self._build_preference_box())
        inner_layout.addWidget(self._build_defects_box())
        inner_layout.addWidget(self._build_notes_box())
        inner_layout.addStretch(1)
        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # -- construction --------------------------------------------------------

    def _build_preference_box(self) -> QGroupBox:
        box = QGroupBox("Which is better? (ASP vs Simple)")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setSpacing(4)

        pref_row = QHBoxLayout()
        pref_row.setSpacing(4)
        self._pref_group = QButtonGroup(self)
        self._pref_group.setExclusive(False)
        self._pref_buttons: dict[str, QPushButton] = {}
        for key, label in PREFERENCES:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _c, k=key: self.set_preference(k, toggle=True))
            self._pref_group.addButton(btn)
            pref_row.addWidget(btn)
            self._pref_buttons[key] = btn
        layout.addLayout(pref_row)

        conf_row = QHBoxLayout()
        conf_row.setSpacing(4)
        conf_row.addWidget(subtle("Confidence"))
        self._conf_group = QButtonGroup(self)
        self._conf_group.setExclusive(False)
        self._conf_buttons: dict[int, QPushButton] = {}
        for value, label in sorted(CONFIDENCE_LABELS.items()):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _c, v=value: self.set_confidence(v, toggle=True))
            self._conf_group.addButton(btn)
            conf_row.addWidget(btn)
            self._conf_buttons[value] = btn
        conf_row.addStretch(1)
        layout.addLayout(conf_row)
        return box

    def _build_defects_box(self) -> QGroupBox:
        box = QGroupBox("Defects by output (0=absent, 1=trace, 2=noticeable, 3=severe)")
        outer = QVBoxLayout(box)
        outer.setContentsMargins(8, 4, 8, 6)
        outer.setSpacing(6)
        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        self._defect_buttons: dict[str, QPushButton] = {}
        # The last entry ("Other") is deliberately unnumbered — it has no
        # keyboard shortcut (0-9 are all spoken for) and is described in the
        # notes, so it gets the full row width instead of sharing a slot.
        numbered = DEFECTS[:-1]
        other_key, other_title, other_hint = DEFECTS[-1]
        for index, (key, title, hint) in enumerate(numbered):
            btn = QPushButton(f"{index}. {title}")
            btn.setCheckable(True)
            btn.setToolTip(hint)
            btn.clicked.connect(lambda _c, k=key: self.toggle_defect(k))
            grid.addWidget(btn, index // 2, index % 2)
            self._defect_buttons[key] = btn
        other_row = -(-len(numbered) // 2)  # ceiling division: the row after the last pair
        other_btn = QPushButton(other_title)
        other_btn.setCheckable(True)
        other_btn.setToolTip(other_hint)
        other_btn.clicked.connect(lambda _c, k=other_key: self.toggle_defect(k))
        grid.addWidget(other_btn, other_row, 0, 1, 2)  # span both columns
        self._defect_buttons[other_key] = other_btn
        outer.addWidget(grid_host)

        self._severity_selector = QHBoxLayout()
        self._severity_selector.setSpacing(4)
        self._severity_selector.addWidget(subtle("Grade output:"))
        self._severity_selector.addStretch(1)
        outer.addLayout(self._severity_selector)
        self._severity_image_buttons: dict[str, QPushButton] = {}

        self._severity_rows: dict[str, DefectSeverityRow] = {}
        severity_host = QWidget()
        severity_layout = QVBoxLayout(severity_host)
        severity_layout.setContentsMargins(0, 0, 0, 0)
        severity_layout.setSpacing(2)
        for key, title, hint in DEFECTS:
            row = DefectSeverityRow(key, title, hint)
            row.severityChanged.connect(self._on_severity_changed)
            severity_layout.addWidget(row)
            self._severity_rows[key] = row
        outer.addWidget(severity_host)
        return box

    def _build_notes_box(self) -> QGroupBox:
        box = QGroupBox("Notes")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 4, 8, 6)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText(
            "What differs between the outputs, and why the scores above."
        )
        self.notes_edit.setFixedHeight(72)
        self.notes_edit.textChanged.connect(self._on_notes_changed)
        layout.addWidget(self.notes_edit)
        return box

    # -- content -------------------------------------------------------------

    def set_comparators(self, keys: list[str]) -> None:
        """Rebuild the score blocks for the comparators this test actually has,
        so a test without an Overmix output shows no dead Overmix rows."""
        for block in self.blocks.values():
            block.setParent(None)
        self.blocks = {}
        for key in keys:
            block = ImageScoreBlock(key)
            block.scoreChanged.connect(self._on_score_changed)
            self._blocks_layout.addWidget(block)
            block.set_detailed(self._detail_toggle.isChecked())
            self.blocks[key] = block
        self._set_severity_comparators(keys)

    def load_entry(self, entry: RatingEntry) -> None:
        self._loading = True
        try:
            self._entry = entry
            for block in self.blocks.values():
                block.load(entry)
                block.set_detailed(self._detail_toggle.isChecked())
            self._sync_preference_buttons()
            self._sync_defect_buttons()
            self._sync_severity_rows()
            self.notes_edit.setPlainText(entry.notes)
        finally:
            self._loading = False

    def _on_detail_toggled(self, detailed: bool) -> None:
        self._detail_toggle.setText("Hide optional dimensions" if detailed else "Show all dimensions")
        for block in self.blocks.values():
            block.set_detailed(detailed)

    def entry(self) -> RatingEntry:
        return self._entry

    # -- edits ---------------------------------------------------------------

    def _emit_changed(self) -> None:
        if not self._loading:
            self.changed.emit()

    def _on_score_changed(self, image_key: str, dimension: str, score: int | None) -> None:
        self._entry.set_score(image_key, dimension, score)
        self._emit_changed()

    def score_focused(
        self, image_key: str, score: int | None, dimension: str = DIM_COHERENCE
    ) -> bool:
        """Set a score from a keypress. Returns False when the target block
        isn't present, so the caller can report why nothing happened."""
        block = self.blocks.get(image_key)
        if block is None:
            return False
        block.set_score(dimension, score)
        self._entry.set_score(image_key, dimension, score)
        self._emit_changed()
        return True

    def set_preference(self, key: str | None, toggle: bool = False) -> None:
        if toggle and self._entry.preference == key:
            key = None
        self._entry.preference = key
        self._sync_preference_buttons()
        self._emit_changed()

    def set_confidence(self, value: int | None, toggle: bool = False) -> None:
        if toggle and self._entry.confidence == value:
            value = None
        self._entry.confidence = value
        self._sync_preference_buttons()
        self._emit_changed()

    def toggle_defect(self, key: str) -> None:
        severity = self._entry.severity(self._severity_image, key)
        self.set_defect_severity(key, 0 if severity else 1)

    def set_defect_severity(self, key: str, severity: int) -> None:
        self._entry.set_severity(self._severity_image, key, severity)
        self._sync_defect_buttons()
        self._sync_severity_rows()
        self._emit_changed()

    def toggle_defect_index(self, index: int) -> str | None:
        if not (0 <= index < len(DEFECTS)):
            return None
        key = DEFECTS[index][0]
        self.toggle_defect(key)
        return key

    def _on_notes_changed(self) -> None:
        self._entry.notes = self.notes_edit.toPlainText()
        self._emit_changed()

    def _set_severity_comparators(self, keys: list[str]) -> None:
        eligible = [key for key in keys if key in self.blocks]
        for button in self._severity_image_buttons.values():
            self._severity_selector.removeWidget(button)
            button.deleteLater()
        self._severity_image_buttons = {}
        if not eligible:
            return
        if self._severity_image not in eligible:
            self._severity_image = eligible[0]
        for key in eligible:
            button = QPushButton(COMPARATOR_TITLES.get(key, key))
            button.setCheckable(True)
            button.clicked.connect(lambda _checked, k=key: self._set_severity_image(k))
            self._severity_selector.insertWidget(self._severity_selector.count() - 1, button)
            self._severity_image_buttons[key] = button
        self._sync_severity_rows()

    def _set_severity_image(self, image: str) -> None:
        self._severity_image = image
        self._sync_severity_rows()

    def _on_severity_changed(self, defect: str, severity: int) -> None:
        self.set_defect_severity(defect, severity)

    def _sync_preference_buttons(self) -> None:
        for key, btn in self._pref_buttons.items():
            btn.setChecked(self._entry.preference == key)
        for value, btn in self._conf_buttons.items():
            btn.setChecked(self._entry.confidence == value)

    def _sync_defect_buttons(self) -> None:
        active = set(self._entry.defects)
        for key, btn in self._defect_buttons.items():
            btn.setChecked(key in active)

    def _sync_severity_rows(self) -> None:
        for key, button in self._severity_image_buttons.items():
            button.setChecked(key == self._severity_image)
        for key, row in self._severity_rows.items():
            severity = self._entry.severity(self._severity_image, key)
            row.set_severity(
                None if severity == 0 and key in self._entry.defects else severity
            )

    # -- status --------------------------------------------------------------

    def refresh_theme(self) -> None:
        """Re-derive every score block's inline chip styles after a theme
        change (see ``ScoreRow.refresh_theme``)."""
        for block in self.blocks.values():
            block.refresh_theme()
        for row in self._severity_rows.values():
            row.refresh_theme()

    def missing_required(self) -> list[str]:
        """Which required scores are still blank. Only the ASP and Simple
        coherence rows are required — those two are what the benchmark's veto
        logic reads, and demanding all 20 sub-scores would make the pass
        unaffordable."""
        missing = []
        if self._entry.asp is None:
            missing.append("ASP coherence")
        if self._entry.simple is None:
            missing.append("Simple coherence")
        return missing
