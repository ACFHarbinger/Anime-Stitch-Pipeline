"""Unit tests for the scoring panel and defect severity scoreboard UI."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from ..constants.schema import DEFECTS, IMAGE_ASP
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
