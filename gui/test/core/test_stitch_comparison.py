"""Regression coverage for the Stitch tab's ASP/SCANS comparison preview."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from asp_gui.elements import StitchTab
from asp_gui.elements._thumb_workers import _ScansComparisonTask
from PySide6.QtGui import QPixmap
from PySide6.QtTest import QTest

pytestmark = pytest.mark.gui


def _write_image(path, value: int) -> None:
    cv2.imwrite(str(path), np.full((24, 32, 3), value, dtype=np.uint8))


def test_scans_quality_metrics_include_all_comparison_scores(tmp_path):
    path = tmp_path / "panorama.png"
    _write_image(path, 128)

    metrics = _ScansComparisonTask._quality_metrics(str(path))

    assert "Sharpness:" in metrics
    assert "Ghosting:" in metrics
    assert "Seam gradient:" in metrics


def test_result_toggle_switches_between_asp_and_scans(q_app, tmp_path):
    asp_path = tmp_path / "asp.png"
    scans_path = tmp_path / "scans.png"
    _write_image(asp_path, 80)
    _write_image(scans_path, 180)
    tab = StitchTab()
    tab._result_pix = QPixmap(str(asp_path))
    tab._scans_pix = QPixmap(str(scans_path))
    tab._comparison_metrics = {"ASP": "asp metrics", "SCANS": "scans metrics"}
    tab._result_preview_label.resize(240, 160)
    tab._btn_before_after.setEnabled(True)

    tab._update_result_preview()
    assert tab._result_metrics_label.text() == "ASP: asp metrics"

    tab._btn_before_after.click()
    QTest.qWait(120)

    assert tab._btn_before_after.isChecked()
    assert tab._btn_before_after.text() == "SCANS ▶"
    assert tab._result_metrics_label.text() == "SCANS: scans metrics"
