"""The per-test metrics readout: headline facts, the no-reference CV metric
table across every comparator, the ground-truth table, and the pipeline config.

This is the data ``other/discovery.py`` has always loaded and the old dashboard
never displayed (issue #123 defect 7). Winning cells are tinted by direction —
read off ``bench_anime_stitch.py``'s own metric definitions, not their names —
so "who won this metric" is visible without knowing that ``seam_coherence`` is a
banding proxy where lower is better.

Showing these numbers next to the images is also what makes a human rating
*calibratable*: §0.2's still-open item is rank-correlating the automated metrics
against human judgment, which needs the human to have seen both.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..constants.schema import COMPARATOR_TITLES
from ..constants.user_interface import COL_BAD, COL_GOOD, COL_TEXT_DIM, COL_WARN
from ..other import metrics_view as mv
from .theme import subtle

# fallback_reason prefixes worth colouring — a fallback isn't a failure, it's a
# guarded substitution, but it *is* the single most useful fact about a test.
_FALLBACK_COLOR = COL_WARN
_VERDICT_COLORS = {
    "asp_better": COL_GOOD,
    "simple_better": COL_BAD,
    "comparable": COL_WARN,
}


def _metric_table(rows: list[mv.MetricRow], keys: list[str]) -> QTableWidget:
    table = QTableWidget(len(rows), len(keys) + 1)
    table.setHorizontalHeaderLabels(["Metric"] + [COMPARATOR_TITLES.get(k, k) for k in keys])
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    table.setShowGrid(True)
    header = table.horizontalHeader()
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    for col in range(1, len(keys) + 1):
        header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

    for r, row in enumerate(rows):
        arrow = {mv.HIGHER_BETTER: " ↑", mv.LOWER_BETTER: " ↓"}.get(row.direction, "")
        name = QTableWidgetItem(row.label + arrow)
        name.setToolTip(
            {
                mv.HIGHER_BETTER: "Higher is better",
                mv.LOWER_BETTER: "Lower is better",
            }.get(row.direction, "No preferred direction")
        )
        table.setItem(r, 0, name)
        best = row.best_key()
        for c, key in enumerate(keys, start=1):
            cell = QTableWidgetItem(row.formatted(key))
            cell.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if best is not None and key == best:
                cell.setForeground(QColor(COL_GOOD))
                font = cell.font()
                font.setBold(True)
                cell.setFont(font)
            elif row.values.get(key) is None:
                cell.setForeground(QColor(COL_TEXT_DIM))
            table.setItem(r, c, cell)
    table.setFixedHeight(min(420, 26 * len(rows) + 34))
    return table


class MetricsPanel(QWidget):
    """Scrollable stack of metric sections for the current test."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._inner = QWidget()
        self._layout = QVBoxLayout(self._inner)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(self._inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self._sections: list[QWidget] = []

    def _reset(self) -> None:
        for section in self._sections:
            section.setParent(None)
        self._sections = []

    def _add_section(self, widget: QWidget) -> None:
        self._layout.addWidget(widget)
        self._sections.append(widget)

    def set_metrics(self, entry: dict, run_label: str = "") -> None:
        self._reset()
        if not entry:
            self._add_section(subtle(
                "No benchmark result for this test in the loaded results JSON.\n"
                "Images can still be compared and rated; the metric tables need a run "
                "that includes this dataset."
            ))
            self._layout.addStretch(1)
            return

        self._add_section(self._facts_box(entry, run_label))
        keys = mv.present_comparators(entry)
        cv_rows = mv.cv_metric_rows(entry, keys)
        if cv_rows:
            box = QGroupBox("No-reference CV metrics")
            layout = QVBoxLayout(box)
            layout.setContentsMargins(6, 4, 6, 6)
            layout.addWidget(_metric_table(cv_rows, keys))
            self._add_section(box)

        gt_rows = mv.gt_metric_rows(entry)
        box = QGroupBox("Ground-truth comparison")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 6)
        if gt_rows:
            layout.addWidget(_metric_table(gt_rows, ["asp", "simple"]))
        else:
            layout.addWidget(subtle(
                "No ground truth for this test — 42 of the 97 corpus tests are GT-less "
                "by design. Use CQAS and the no-reference metrics above."
            ))
        self._add_section(box)

        config = mv.pipeline_config_rows(entry)
        if config:
            box = QGroupBox("Pipeline configuration for this run")
            grid = QGridLayout(box)
            grid.setContentsMargins(6, 4, 6, 6)
            grid.setSpacing(3)
            for i, (key, value) in enumerate(config):
                label = QLabel(key)
                label.setStyleSheet(f"color: {COL_TEXT_DIM};")
                grid.addWidget(label, i // 2, (i % 2) * 2)
                grid.addWidget(QLabel(value), i // 2, (i % 2) * 2 + 1)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(3, 1)
            self._add_section(box)

        self._layout.addStretch(1)

    def _facts_box(self, entry: dict, run_label: str) -> QGroupBox:
        box = QGroupBox(f"Benchmark result{f' — {run_label}' if run_label else ''}")
        grid = QGridLayout(box)
        grid.setContentsMargins(6, 4, 6, 6)
        grid.setSpacing(4)
        for i, (label, value) in enumerate(mv.headline_facts(entry)):
            name = QLabel(label)
            name.setStyleSheet(f"color: {COL_TEXT_DIM};")
            grid.addWidget(name, i, 0)
            display = QLabel(value)
            display.setWordWrap(True)
            color = None
            if label == "Verdict":
                color = _VERDICT_COLORS.get(value)
            elif label == "Fallback reason" or (label == "Composite" and "fallback" in value):
                color = _FALLBACK_COLOR
            if color:
                display.setStyleSheet(f"color: {color}; font-weight: 600;")
            grid.addWidget(display, i, 1)
        grid.setColumnStretch(1, 1)
        return box
