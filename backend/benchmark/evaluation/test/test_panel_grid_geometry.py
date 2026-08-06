"""
backend/benchmark/evaluation/test/test_panel_grid_geometry.py
=============================================================
Unit tests for PanelGrid geometry, size policy, and layout modes.

Regression suite for issue #153 — verifies that the internal QStackedWidget
instances inside PanelGrid retain ``Ignored`` horizontal size policy so their
sizeHint (which reflects the maximum of all children's sizeHint) never
propagates a large native-image width up through the layout tree.

Run:
    pytest backend/benchmark/evaluation/test/test_panel_grid_geometry.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QSizePolicy

pytestmark = pytest.mark.inspector_ui


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid():
    from asp_backend_evaluation.ui.panel_grid import PanelGrid
    return PanelGrid()


def _solid(h: int, w: int, val: int = 128) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def _load_images(grid, images: dict) -> None:
    grid.set_images(images)
    QApplication.processEvents()


# ---------------------------------------------------------------------------
# Size policy — regression guard for root cause #4 of issue #153
# ---------------------------------------------------------------------------

class TestPanelGridSizePolicy:
    """PanelGrid and its internal QStackedWidgets must have Ignored H policy."""

    def test_panel_grid_horizontal_policy_is_ignored(self, qapp):
        grid = _make_grid()
        assert grid.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored, (
            "PanelGrid horizontal sizePolicy must be Ignored (issue #153)."
        )

    def test_host_stack_horizontal_policy_is_ignored(self, qapp):
        """The internal _host_stack QStackedWidget must also be Ignored.

        Regression guard for issue #153 root cause #4: QStackedWidget defaults
        to Preferred, which causes sizeHint() to report the max child size
        (native image dimensions) to the parent QLayout even when PanelGrid
        itself is Ignored.
        """
        grid = _make_grid()
        assert grid._host_stack.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored, (
            "_host_stack horizontal sizePolicy must be Ignored (issue #153 root cause #4). "
            "A Preferred policy allows the QStackedWidget to report native image width "
            "as its sizeHint, propagating a bloated width request up to the QSplitter."
        )

    def test_stack_horizontal_policy_is_ignored(self, qapp):
        """The single-panel _stack QStackedWidget must also be Ignored."""
        grid = _make_grid()
        assert grid._stack.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored, (
            "_stack horizontal sizePolicy must be Ignored (issue #153 root cause #4)."
        )

    def test_panel_cells_horizontal_policy_is_ignored(self, qapp):
        """Every _PanelCell must have Ignored horizontal policy."""
        grid = _make_grid()
        _load_images(grid, {"asp": _solid(1200, 800), "simple": _solid(900, 600)})
        for key, cell in grid.cells.items():
            pol = cell.sizePolicy().horizontalPolicy()
            assert pol == QSizePolicy.Policy.Ignored, (
                f"_PanelCell[{key}] has {pol.name} horizontal policy — must be Ignored."
            )


# ---------------------------------------------------------------------------
# Image loading and visibility
# ---------------------------------------------------------------------------

class TestPanelGridImageLoading:

    def test_set_images_creates_panels(self, qapp):
        """Panels for each loaded comparator must exist in grid.panels."""
        grid = _make_grid()
        images = {"asp": _solid(1200, 800), "simple": _solid(900, 600)}
        _load_images(grid, images)
        # PanelGrid pre-allocates all known comparator slots; the loaded keys
        # must be a subset (not necessarily equal to) panels.keys().
        assert {"asp", "simple"}.issubset(set(grid.panels.keys())), (
            f"Loaded comparators not in grid.panels: {set(grid.panels.keys())}"
        )

    def test_set_images_clears_old_panels(self, qapp):
        grid = _make_grid()
        _load_images(grid, {"asp": _solid(1200, 800), "gt": _solid(900, 600)})
        _load_images(grid, {"simple": _solid(800, 600)})
        # Only the new panels should exist (old gt panel should be gone from visible set)
        visible = grid.visible()
        assert "gt" not in visible

    def test_visible_returns_subset_of_available(self, qapp):
        grid = _make_grid()
        _load_images(grid, {"asp": _solid(1200, 800), "simple": _solid(900, 600), "gt": _solid(800, 600)})
        visible = grid.visible()
        available = grid.available()
        assert set(visible).issubset(set(available))

    def test_set_visible_filters_panels(self, qapp):
        grid = _make_grid()
        _load_images(grid, {"asp": _solid(1200, 800), "simple": _solid(900, 600), "gt": _solid(800, 600)})
        grid.set_visible(["asp", "simple"])
        QApplication.processEvents()
        assert grid.visible() == ["asp", "simple"]

    def test_fit_all_no_image_does_not_raise(self, qapp):
        """fit_all on a grid with no images must not raise."""
        grid = _make_grid()
        grid.fit_all()


# ---------------------------------------------------------------------------
# fit_all — regression guard for issue #153 re-entrancy + overflow
# ---------------------------------------------------------------------------

class TestPanelGridFitAll:

    def test_fit_all_does_not_expand_grid_widget(self, qapp):
        """After fit_all, PanelGrid must not grow beyond its allocated size.

        Regression guard for issue #153: if fit_all caused a layout propagation
        (via QGraphicsView internal geometry updates) that let PanelGrid negotiate
        more horizontal space, the QSplitter would overflow the window boundary.
        """
        grid = _make_grid()
        grid.resize(600, 500)
        QApplication.processEvents()
        _load_images(grid, {"asp": _solid(1200, 800), "simple": _solid(900, 600)})

        grid.fit_all()
        QApplication.processEvents()

        assert grid.width() <= 602, (
            f"PanelGrid grew beyond its allocated width after fit_all: {grid.width()}"
        )

    def test_fit_all_called_twice_is_idempotent(self, qapp):
        """Calling fit_all twice must produce the same _fit_scale values."""
        grid = _make_grid()
        grid.resize(500, 500)
        QApplication.processEvents()
        _load_images(grid, {"asp": _solid(1200, 800)})

        grid.fit_all()
        QApplication.processEvents()
        scale_1 = grid.panels["asp"]._fit_scale

        grid.fit_all()
        QApplication.processEvents()
        scale_2 = grid.panels["asp"]._fit_scale

        assert abs(scale_1 - scale_2) < 1e-6, (
            f"fit_all is not idempotent: {scale_1:.6f} != {scale_2:.6f}"
        )

    def test_fit_all_after_resize_stays_in_bounds(self, qapp):
        """After a resize followed by fit_all, no panel must overflow its viewport."""
        grid = _make_grid()
        _load_images(grid, {"asp": _solid(1200, 800), "simple": _solid(600, 2000)})

        for w, h in [(400, 400), (800, 600), (1200, 900)]:
            grid.resize(w, h)
            QApplication.processEvents()
            grid.fit_all()
            QApplication.processEvents()

            for key in grid.visible():
                panel = grid.panels[key]
                vw = panel.viewport().width()
                vh = panel.viewport().height()
                img_h, img_w = panel._image_bgr.shape[:2]
                s = panel._fit_scale
                assert s * img_w <= vw + 2, (
                    f"[{key}] Width overflow at grid({w},{h}): "
                    f"{s * img_w:.1f} > viewport_w={vw}"
                )
                assert s * img_h <= vh + 2, (
                    f"[{key}] Height overflow at grid({w},{h}): "
                    f"{s * img_h:.1f} > viewport_h={vh}"
                )
