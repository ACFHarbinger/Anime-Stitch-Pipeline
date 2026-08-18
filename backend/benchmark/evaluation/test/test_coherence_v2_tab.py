"""Unit tests for CoherenceV2Tab in the ASP benchmark evaluation inspector."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    from benchmark.evaluation.ui.coherence_v2_tab import CoherenceV2Tab
except ImportError:
    from submodules.ASP.backend.benchmark.evaluation.ui.coherence_v2_tab import (
        CoherenceV2Tab,
    )


@pytest.mark.inspector_ui
def test_coherence_v2_tab_initialization(qapp):
    tab = CoherenceV2Tab()
    assert tab is not None
    assert tab.case_list is not None
    assert tab.combo_mode is not None
    assert tab.combo_filter is not None
    assert tab.image_panel is not None


@pytest.mark.inspector_ui
def test_coherence_v2_tab_filter_toggle(qapp):
    tab = CoherenceV2Tab()
    # Mock some cases
    tab._cases = [
        {"name": "asp_test04", "engineering_verdict": "improves_anatomy"},
        {"name": "asp_test06", "engineering_verdict": "parity"},
    ]
    tab._redset_data = {"cases": [{"name": "asp_test04"}]}
    tab._populate_cases()

    # Filter 0: All Cases
    tab.combo_filter.setCurrentIndex(0)
    assert tab.case_list.count() == 2

    # Filter 1: Redset Only
    tab.combo_filter.setCurrentIndex(1)
    assert tab.case_list.count() == 1
    assert "asp_test04" in tab.case_list.item(0).text()

    # Filter 2: Improved Anatomy Only
    tab.combo_filter.setCurrentIndex(2)
    assert tab.case_list.count() == 1
    assert "asp_test04" in tab.case_list.item(0).text()


@pytest.mark.inspector_ui
def test_coherence_v2_tab_view_modes(qapp):
    tab = CoherenceV2Tab()
    tab._img_default = np.zeros((100, 100, 3), dtype=np.uint8)
    tab._img_v2 = np.ones((100, 100, 3), dtype=np.uint8) * 128

    for mode_idx in range(tab.combo_mode.count()):
        tab.combo_mode.setCurrentIndex(mode_idx)
        tab._refresh_view()
        # Verify status text updated
        assert tab.lbl_status.text() != ""


@pytest.mark.inspector_ui
def test_coherence_v2_tab_pixel_hover_readout(qapp):
    tab = CoherenceV2Tab()
    assert tab.lbl_pixel.text() == "Pixel: —"

    tab._on_pixel_hovered(12, 34, (10, 20, 30))  # BGR
    assert "(12, 34)" in tab.lbl_pixel.text()
    assert "R= 30" in tab.lbl_pixel.text()
    assert "G= 20" in tab.lbl_pixel.text()
    assert "B= 10" in tab.lbl_pixel.text()
    assert "#1e140a" in tab.lbl_pixel.text()

    tab._on_pixel_hovered(-1, -1, None)
    assert tab.lbl_pixel.text() == "Pixel: —"


@pytest.mark.inspector_ui
def test_coherence_v2_tab_set_context(qapp):
    tab = CoherenceV2Tab()
    tab._cases = [
        {"name": "asp_test04", "engineering_verdict": "improves_anatomy"},
        {"name": "asp_test06", "engineering_verdict": "parity"},
    ]
    tab._populate_cases()
    assert tab.case_list.count() == 2

    tab.set_context(images={}, metrics={}, name="asp_test06")
    assert tab.case_list.currentRow() == 1
