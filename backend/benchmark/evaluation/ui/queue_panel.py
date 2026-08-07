"""The dataset queue sidebar: every test in the corpus, its rating state, and
free navigation to any of them.

The old dashboard offered only Back / Skip / Next over a filtered ``todo`` list,
so there was no way to jump to a specific test, no view of overall progress
beyond a text line, and — because Skip didn't record history — no way back to
something you'd passed. A visible, clickable list of all 97 with their state is
also what makes a partially-finished pass resumable in practice: you can see at
a glance what's left.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from ..constants.user_interface import COL_GOOD, COL_TEXT_DIM, COL_WARN, SCORE_COLORS
from ..other.session import EvaluationSession

FILTER_ALL = "all"
FILTER_TODO = "todo"
FILTER_RATED = "rated"
FILTER_SKIPPED = "skipped"

_FILTERS = (
    (FILTER_ALL, "All tests"),
    (FILTER_TODO, "Not yet rated"),
    (FILTER_RATED, "Rated"),
    (FILTER_SKIPPED, "Skipped"),
)


class QueuePanel(QWidget):
    datasetSelected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session: EvaluationSession | None = None
        self._suppress = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._filter = QComboBox()
        for key, label in _FILTERS:
            self._filter.addItem(label, key)
        self._filter.currentIndexChanged.connect(lambda _i: self.refresh())
        layout.addWidget(self._filter)

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_current_changed)
        layout.addWidget(self._list, stretch=1)

        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        layout.addWidget(self._progress)
        self._summary = QLabel("")
        self._summary.setProperty("role", "subtle")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

    def set_session(self, session: EvaluationSession) -> None:
        self._session = session
        self.refresh()

    def refresh(self) -> None:
        if self._session is None:
            return
        mode = self._filter.currentData()
        self._suppress = True
        try:
            self._list.clear()
            for name in self._session.order:
                rated = self._session.is_rated(name)
                skipped = self._session.is_skipped(name) and not rated
                if mode == FILTER_TODO and rated:
                    continue
                if mode == FILTER_RATED and not rated:
                    continue
                if mode == FILTER_SKIPPED and not skipped:
                    continue
                self._list.addItem(self._make_item(name, rated, skipped))
            self._select_current()
        finally:
            self._suppress = False
        self._update_progress()

    def _make_item(self, name: str, rated: bool, skipped: bool) -> QListWidgetItem:
        entry = self._session.evaluations.get(name)
        if rated and entry is not None:
            marker = "●"
            detail = f"  ASP {entry.asp} / Sim {entry.simple}"
            color = SCORE_COLORS.get(entry.asp, COL_GOOD)
        elif skipped:
            marker, detail, color = "○", "  skipped", COL_WARN
        else:
            marker, detail, color = "·", "", COL_TEXT_DIM
        item = QListWidgetItem(f"{marker} {name}{detail}")
        item.setData(Qt.ItemDataRole.UserRole, name)
        item.setForeground(QColor(color))
        if entry is not None and entry.defects:
            item.setToolTip("defects: " + ", ".join(entry.defects))
        return item

    def _select_current(self) -> None:
        if self._session is None or self._session.current is None:
            return
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == self._session.current:
                self._list.setCurrentRow(row)
                self._list.scrollToItem(item)
                return

    def _on_current_changed(self, current: QListWidgetItem | None, _previous) -> None:
        if self._suppress or current is None:
            return
        name = current.data(Qt.ItemDataRole.UserRole)
        if name and self._session is not None and name != self._session.current:
            self.datasetSelected.emit(name)

    def _update_progress(self) -> None:
        progress = self._session.progress()
        self._progress.setMaximum(max(1, progress.total))
        self._progress.setValue(progress.rated)
        self._progress.setFormat(f"{progress.rated}/{progress.total} rated (%p%)")
        remaining = progress.total - progress.rated
        parts = [f"{remaining} left"]
        if progress.skipped:
            parts.append(f"{progress.skipped} skipped")
        if progress.position:
            parts.append(f"at #{progress.position}")
        self._summary.setText(" · ".join(parts))

    def visible_names(self) -> list[str]:
        return [
            self._list.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(self._list.count())
        ]
