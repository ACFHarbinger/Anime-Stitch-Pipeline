"""Unit tests for the scoring panel and defect severity scoreboard UI."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from ..other.schema import RatingEntry
from ..ui.scoring_panel import DefectSeverityRow, ScoringPanel


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_defect_severity_row_buttons(qapp):
    row = DefectSeverityRow("torn_anatomy", "Torn anatomy", "severed body part")
    assert row._severity == 0

    # Click severity 2
    row._buttons[2].click()
    assert row._severity == 2
    assert row._buttons[2].isChecked()
    assert not row._buttons[0].isChecked()

    # Set severity 0
    row.set_severity(0)
    assert row._severity == 0
    assert row._buttons[0].isChecked()


def test_scoring_panel_defect_sync(qapp):
    panel = ScoringPanel()
    panel.set_comparators(["asp", "simple"])

    entry = RatingEntry()
    entry.set_severity("asp", "torn_anatomy", 3)
    panel.load_entry(entry)

    assert panel._severity_rows["torn_anatomy"]._severity == 3
    assert panel._defect_buttons["torn_anatomy"].isChecked()

    # Change severity via row
    panel._severity_rows["torn_anatomy"]._buttons[1].click()
    assert panel.entry().severity("asp", "torn_anatomy") == 1


def test_grading_one_output_does_not_change_other_outputs_rows(qapp):
    """Grading a defect on one output must leave every other output's severity
    row alone — a 0 elsewhere reads as "absent", not "legacy: ungraded"."""
    panel = ScoringPanel()
    panel.set_comparators(["asp", "simple", "baseline"])
    panel.load_entry(RatingEntry())

    panel._set_severity_image("simple")
    panel._severity_rows["ghosting"]._buttons[2].click()

    assert panel.entry().defect_severity == {"simple": {"ghosting": 2}}
    for other in ("asp", "baseline"):
        panel._set_severity_image(other)
        assert panel._severity_rows["ghosting"]._severity == 0  # absent, not None


def test_legacy_defect_with_no_per_output_severity_reads_ungraded(qapp):
    """A pre-per-output entry (defect tagged, no defect_severity) still shows
    every output's row as ungraded so the reviewer knows to grade it."""
    panel = ScoringPanel()
    panel.set_comparators(["asp", "simple"])
    entry = RatingEntry()
    entry.defects = ["banding"]  # legacy tag, no defect_severity
    panel.load_entry(entry)

    for img in ("asp", "simple"):
        panel._set_severity_image(img)
        assert panel._severity_rows["banding"]._severity is None
