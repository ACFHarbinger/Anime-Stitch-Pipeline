"""
Offscreen tests for the top-level Rate/Analyze split and collapsible sidebars
(issue #123 followup).

Deliberately does *not* construct a full ``InspectorWindow``: doing so inside
this suite's pytest session (which loads ``backend/test/conftest.py``'s heavy
ML-library imports before any test runs) segfaults during widget construction
— confirmed to be specific to that combination, not a product bug, by building
the identical window via the identical code path as a plain script outside
pytest, which works cleanly every time. Every other test file in this package
tests components directly for the same underlying reason; this one follows
suit and tests the two pieces of new logic in isolation instead:

  - ``InspectorToolbar``'s collapsible-sidebar toggle signals.
  - the generic QSplitter hide/show behaviour ``main_window.py``'s
    ``_build_rate_tab`` relies on for the toggle to actually free the space
    (rather than leaving a stale gap — the same class of bug the panel-grid
    reflow fix addressed) and to restore it correctly afterwards.
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _repo_root)

pytest.importorskip("PySide6", reason="the inspector needs PySide6")

from asp_backend.evaluation.ui.toolbar import InspectorToolbar  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QSplitter  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def toolbar(qapp):
    widget = InspectorToolbar()
    yield widget
    widget.deleteLater()


def test_sidebar_toggles_start_checked(toolbar):
    assert toolbar._queue_toggle.isChecked() is True
    assert toolbar._side_toggle.isChecked() is True


def test_unchecking_queue_toggle_emits_false(toolbar):
    seen = []
    toolbar.queuePanelToggled.connect(seen.append)
    toolbar._queue_toggle.setChecked(False)
    assert seen == [False]


def test_unchecking_side_toggle_emits_false(toolbar):
    seen = []
    toolbar.sidePanelToggled.connect(seen.append)
    toolbar._side_toggle.setChecked(False)
    assert seen == [False]


def test_rechecking_a_toggle_emits_true(toolbar):
    seen = []
    toolbar.queuePanelToggled.connect(seen.append)
    toolbar._queue_toggle.setChecked(False)
    toolbar._queue_toggle.setChecked(True)
    assert seen == [False, True]


def test_the_two_sidebar_toggles_are_independent(toolbar):
    queue_seen, side_seen = [], []
    toolbar.queuePanelToggled.connect(queue_seen.append)
    toolbar.sidePanelToggled.connect(side_seen.append)
    toolbar._queue_toggle.setChecked(False)
    assert queue_seen == [False]
    assert side_seen == []


# ---------------------------------------------------------------------------
# The QSplitter behaviour main_window.py's toggle wiring depends on: hiding a
# child frees its space for the others, and showing it again restores the
# pre-hide proportions — both confirmed empirically before relying on it,
# since "hide the widget" alone was exactly the wrong fix for the panel-grid
# reflow bug (the widget disappeared but the grid didn't reclaim its space).
# ---------------------------------------------------------------------------


@pytest.fixture()
def splitter(qapp):
    widget = QSplitter(Qt.Orientation.Horizontal)
    labels = [QLabel("A"), QLabel("B"), QLabel("C")]
    for label in labels:
        widget.addWidget(label)
    widget.setSizes([200, 600, 200])
    widget.resize(1000, 400)
    widget.show()
    qapp.processEvents()
    yield widget, labels
    widget.close()


def test_hiding_a_splitter_child_frees_its_space(splitter, qapp):
    widget, labels = splitter
    labels[0].hide()
    qapp.processEvents()  # the splitter's layout recalculation is deferred
    sizes = widget.sizes()
    assert sizes[0] == 0
    assert sizes[1] > 600  # B grows to absorb A's freed width


def test_showing_a_hidden_splitter_child_restores_its_proportion(splitter, qapp):
    widget, labels = splitter
    before = widget.sizes()
    labels[0].hide()
    qapp.processEvents()
    labels[0].show()
    qapp.processEvents()
    after = widget.sizes()
    assert after[0] > 0
    assert after == pytest.approx(before, abs=2)


def test_hiding_two_of_three_children_leaves_the_middle_one_full_width(splitter, qapp):
    widget, labels = splitter
    labels[0].hide()
    labels[2].hide()
    qapp.processEvents()
    sizes = widget.sizes()
    assert sizes[0] == 0 and sizes[2] == 0
    assert sizes[1] == sum(sizes)


def test_image_panel_size_hints_bounded(qapp):
    import numpy as np
    from asp_backend.evaluation.ui.image_panel import ImagePanel

    panel = ImagePanel("asp", "ASP")
    # Even with a huge native image, sizeHint should be bounded (not 4000px)
    huge_img = np.zeros((3000, 4000, 3), dtype=np.uint8)
    panel.set_image(huge_img)

    hint = panel.sizeHint()
    assert hint.width() <= 800
    assert hint.height() <= 600

    min_hint = panel.minimumSizeHint()
    assert min_hint.width() <= 300
    assert min_hint.height() <= 200
    panel.deleteLater()

