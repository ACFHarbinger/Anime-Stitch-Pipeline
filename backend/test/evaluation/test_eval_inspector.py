"""
Offscreen smoke tests for the PySide6 evaluation inspector (issue #123).

Runs under ``QT_QPA_PLATFORM=offscreen`` (set before PySide6 is imported), the
same convention ``backend/benchmark/bench_gui_thumbnails.py`` uses — no real
display is needed or used, and no event loop is started.

Deliberately targeted at the defects that had no test before, all of which were
invisible to a "does the window build" check:

  defect 1 — Pixel Value Mode was a no-op: ``set_display_mode`` stored a string
             that nothing ever read, and PIXEL_GRID_ZOOM_THRESHOLD was imported
             and unused.
  defect 6 — zoom was discontinuous, because ``fit_to_view`` set ``_zoom = 1.0``
             while the transform was the fit scale; and sync mirrored only the
             scale factor, never the scroll offset, so panned panels drifted.
  defect 8 — the panel row was hard-wired to exactly three comparators.
  defect 9 — the pair visualisations ignored the source selector.
  defect 10 — results were hard-scaled into a QLabel with no zoom.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _repo_root)

pytest.importorskip("PySide6", reason="the inspector needs PySide6")

from asp_backend_evaluation.constants.user_interface import (  # noqa: E402
    DISPLAY_PIXEL,
    DISPLAY_RAW,
    LAYOUT_GRID,
    LAYOUT_ROW,
    LAYOUT_STACK,
    MODE_BBOX,
    MODE_POINT,
    MODE_PROBE,
    PIXEL_GRID_ZOOM_THRESHOLD,
    ZOOM_MAX,
    ZOOM_MIN,
)
from asp_backend_evaluation.ui.image_panel import ImagePanel  # noqa: E402
from asp_backend_evaluation.ui.panel_grid import PanelGrid  # noqa: E402
from asp_backend_evaluation.ui.scoring_panel import ScoringPanel  # noqa: E402
from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _image(h=400, w=300, value=90) -> np.ndarray:
    img = np.full((h, w, 3), value, dtype=np.uint8)
    img[h // 2:, :] = value + 60  # a horizontal seam to look at
    return img


@pytest.fixture()
def panel(qapp):
    widget = ImagePanel("asp", "ASP")
    widget.resize(320, 240)
    # show() so the viewport reaches its final geometry — an unshown
    # QAbstractScrollArea keeps a default viewport size, which would make the
    # fit-scale assertions below measure the wrong box.
    widget.show()
    qapp.processEvents()
    widget.set_image(_image())
    yield widget
    widget.close()


def _expected_fit_scale(widget: ImagePanel) -> float:
    size = widget.viewport().size()
    w, h = widget.image_size()
    return min(size.width() / w, size.height() / h)


# ---------------------------------------------------------------------------
# defect 6 — zoom must be continuous from the fitted view
# ---------------------------------------------------------------------------


def test_fit_sets_zoom_to_one_and_native_scale_to_the_fit_scale(panel):
    panel.fit_to_view()
    assert panel.zoom() == pytest.approx(1.0)
    # The old bug: _zoom said 1.0 while the real device scale was the fit scale,
    # so the two disagreed and the first wheel notch jumped straight to native
    # 1.15x. The two must now be consistent by construction.
    assert panel.native_scale() == pytest.approx(_expected_fit_scale(panel), rel=1e-6)


def test_one_zoom_step_scales_the_fitted_view_not_the_native_size(panel):
    panel.fit_to_view()
    fit_scale = panel.native_scale()
    panel.set_zoom(panel.zoom() * 1.15)
    assert panel.zoom() == pytest.approx(1.15)
    # Continuous: 1.15x the *fitted* scale. The old behaviour landed on 1.15
    # native regardless of the fit, which on a 1700px panorama is a violent jump.
    assert panel.native_scale() == pytest.approx(fit_scale * 1.15, rel=1e-3)


def test_zoom_of_a_large_panorama_starts_far_below_native(qapp):
    """The case that made defect 6 visible: a benchmark panorama is ~1700-2000px
    on its long edge, so its fitted view is a small fraction of native."""
    widget = ImagePanel("asp", "ASP")
    widget.resize(400, 300)
    widget.show()
    qapp.processEvents()
    widget.set_image(_image(1704, 1703))
    try:
        assert widget.zoom() == pytest.approx(1.0)
        assert widget.native_scale() < 0.3
        widget.set_zoom(1.15)
        assert widget.native_scale() < 0.35
    finally:
        widget.close()


@pytest.mark.parametrize("requested,expected", [(1e-6, ZOOM_MIN), (1e6, ZOOM_MAX)])
def test_zoom_is_clamped(panel, requested, expected):
    panel.set_zoom(requested)
    assert panel.zoom() == pytest.approx(expected)


def test_external_view_does_not_re_broadcast(panel):
    """The guard that stops a locked pair from looping."""
    seen = []
    panel.viewChanged.connect(lambda *args: seen.append(args))
    panel.apply_external_view(2.0, 0.5, 0.5)
    assert seen == []
    assert panel.zoom() == pytest.approx(2.0)


def test_zoom_emits_the_normalized_centre_so_pan_can_be_mirrored(panel):
    seen = []
    panel.viewChanged.connect(lambda z, cx, cy: seen.append((z, cx, cy)))
    panel.set_zoom(3.0)
    assert seen
    zoom, cx, cy = seen[-1]
    assert zoom == pytest.approx(3.0)
    assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0


# ---------------------------------------------------------------------------
# defect 1 — Pixel Value Mode
# ---------------------------------------------------------------------------


def test_display_mode_round_trips(panel):
    assert panel.display_mode() == DISPLAY_RAW
    panel.set_display_mode(DISPLAY_PIXEL)
    assert panel.display_mode() == DISPLAY_PIXEL


def test_pixel_overlay_paints_without_error_above_the_grid_threshold(panel, qapp):
    """The old mode stored the string and repainted; nothing read it. Rendering
    the viewport exercises drawForeground's grid + numeric-label path."""
    from PySide6.QtGui import QPainter, QPixmap

    panel.set_display_mode(DISPLAY_PIXEL)
    panel.set_zoom(400.0)  # comfortably past the text threshold too
    assert panel.native_scale() > PIXEL_GRID_ZOOM_THRESHOLD
    target = QPixmap(panel.viewport().size())
    target.fill()
    painter = QPainter(target)
    panel.render(painter)
    painter.end()


def test_pixel_probe_reports_the_underlying_value(panel):
    found = panel.pixel_at(QPoint(panel.viewport().width() // 2, panel.viewport().height() // 2))
    assert found is not None
    x, y, bgr = found
    assert 0 <= x < 300 and 0 <= y < 400
    assert tuple(bgr) in {(90, 90, 90), (150, 150, 150)}


def test_pixel_probe_off_image_returns_none(panel):
    assert panel.pixel_at(QPoint(-50, -50)) is None


def test_pinning_a_pixel_emits_its_value(panel):
    seen = []
    panel.pixelPinned.connect(lambda x, y, bgr: seen.append((x, y, bgr)))
    panel.set_mode(MODE_PROBE)
    panel._pin_pixel(QPoint(panel.viewport().width() // 2, panel.viewport().height() // 2))
    assert len(seen) == 1


def test_pixel_region_crops_a_normalized_rect(panel):
    region = panel.pixel_region(0.0, 0.0, 0.5, 0.25)
    assert region is not None
    assert region.shape[0] == pytest.approx(100, abs=1) and region.shape[1] == pytest.approx(150, abs=1)


def test_pixel_region_of_a_degenerate_rect_is_none(panel):
    assert panel.pixel_region(0.5, 0.5, 0.0, 0.0) is None


def test_visible_region_is_normalized_and_bounded(panel):
    panel.fit_to_view()
    region = panel.visible_region_norm()
    assert region is not None
    x, y, w, h = region
    assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0


# ---------------------------------------------------------------------------
# Annotation geometry
# ---------------------------------------------------------------------------


def test_bbox_is_emitted_in_normalized_coordinates(panel):
    from PySide6.QtCore import QRectF

    seen = []
    panel.bboxDrawn.connect(seen.append)
    panel.set_mode(MODE_BBOX)
    panel._finish_bbox(QRectF(30, 100, 150, 200))  # scene (= image pixel) coords
    assert len(seen) == 1
    data = seen[0]
    assert data["x"] == pytest.approx(30 / 300)
    assert data["y"] == pytest.approx(100 / 400)
    assert data["w"] == pytest.approx(150 / 300)
    assert data["h"] == pytest.approx(200 / 400)


# ---------------------------------------------------------------------------
# Link tool — point-or-region endpoints, followup feedback
#
# Driven through the *real* mousePressEvent/mouseMoveEvent/mouseReleaseEvent
# methods (called directly, not via QApplication.sendEvent): a deeply nested
# QGraphicsView inside a QMainWindow doesn't reliably receive synthetic
# sendEvent-posted mouse events under the offscreen QPA (confirmed by testing
# — the isolated top-level-widget case works, the nested case silently drops
# the event before it reaches the override), which is exactly why the
# pre-existing bbox test above already calls _finish_bbox() directly rather
# than simulating a drag. Calling the event handlers themselves is the
# reliable middle ground: it exercises the actual click-vs-drag threshold
# logic, just without depending on Qt's own event-delivery routing.
# ---------------------------------------------------------------------------


def _mouse_event(kind, pos, panel, buttons=None):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    button = Qt.MouseButton.LeftButton if kind != QMouseEvent.Type.MouseMove else Qt.MouseButton.NoButton
    if buttons is None:
        buttons = Qt.MouseButton.LeftButton if kind != QMouseEvent.Type.MouseButtonRelease else Qt.MouseButton.NoButton
    return QMouseEvent(kind, QPointF(pos), QPointF(panel.mapToGlobal(pos)), button, buttons, Qt.KeyboardModifier.NoModifier)


def _click(panel, pos):
    from PySide6.QtGui import QMouseEvent

    panel.mousePressEvent(_mouse_event(QMouseEvent.Type.MouseButtonPress, pos, panel))
    panel.mouseReleaseEvent(_mouse_event(QMouseEvent.Type.MouseButtonRelease, pos, panel))


def _drag(panel, start, end):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QMouseEvent

    panel.mousePressEvent(_mouse_event(QMouseEvent.Type.MouseButtonPress, start, panel))
    panel.mouseMoveEvent(_mouse_event(QMouseEvent.Type.MouseMove, end, panel, buttons=Qt.MouseButton.LeftButton))
    panel.mouseReleaseEvent(_mouse_event(QMouseEvent.Type.MouseButtonRelease, end, panel))


def _viewport_center(panel):
    return QPoint(panel.viewport().width() // 2, panel.viewport().height() // 2)


def test_a_real_click_in_link_mode_emits_a_point(panel):
    panel.set_mode(MODE_POINT)
    points, regions = [], []
    panel.pointPicked.connect(lambda x, y: points.append((x, y)))
    panel.regionPicked.connect(lambda x, y, w, h: regions.append((x, y, w, h)))
    _click(panel, _viewport_center(panel))
    assert len(points) == 1 and regions == []


def test_a_real_drag_in_link_mode_emits_a_region(panel):
    panel.set_mode(MODE_POINT)
    points, regions = [], []
    panel.pointPicked.connect(lambda x, y: points.append((x, y)))
    panel.regionPicked.connect(lambda x, y, w, h: regions.append((x, y, w, h)))
    center = _viewport_center(panel)
    _drag(panel, center, center + QPoint(40, 30))
    assert points == [] and len(regions) == 1
    x, y, w, h = regions[0]
    assert w > 0 and h > 0


def test_a_tiny_drag_still_counts_as_a_click(panel):
    """Below the 4px threshold — matches the defect-bbox tool's own floor —
    a shaky click must not be misread as an accidental region."""
    panel.set_mode(MODE_POINT)
    points, regions = [], []
    panel.pointPicked.connect(lambda x, y: points.append((x, y)))
    panel.regionPicked.connect(lambda x, y, w, h: regions.append((x, y, w, h)))
    center = _viewport_center(panel)
    _drag(panel, center, center + QPoint(1, 1))
    assert len(points) == 1 and regions == []


def test_region_picked_is_normalized(panel):
    panel.set_mode(MODE_POINT)
    seen = []
    panel.regionPicked.connect(lambda x, y, w, h: seen.append((x, y, w, h)))
    panel._finish_region(_qrectf(30, 100, 150, 200))
    x, y, w, h = seen[0]
    assert (x, y, w, h) == pytest.approx((30 / 300, 100 / 400, 150 / 300, 200 / 400))


def _qrectf(x, y, w, h):
    from PySide6.QtCore import QRectF

    return QRectF(x, y, w, h)


# ---------------------------------------------------------------------------
# EdgeBuilder — pure logic, no Qt event dispatch needed
# ---------------------------------------------------------------------------


def test_edge_builder_needs_two_before_it_can_finish():
    from asp_backend_evaluation.ui.annotations import EdgeBuilder

    builder = EdgeBuilder()
    assert builder.can_finish() is False
    builder.add("asp", 0.1, 0.2)
    assert builder.can_finish() is False
    builder.add("simple", 0.15, 0.2)
    assert builder.can_finish() is True


def test_edge_builder_supports_chains_beyond_two():
    from asp_backend_evaluation.ui.annotations import EdgeBuilder

    builder = EdgeBuilder()
    for key in ("asp", "simple", "overmix", "ground_truth"):
        builder.add(key, 0.1, 0.1)
    assert builder.count() == 4
    edge = builder.finish("four-way seam")
    assert [p.image for p in edge.points] == ["asp", "simple", "overmix", "ground_truth"]
    assert builder.count() == 0  # finishing clears it


def test_edge_builder_finish_below_two_returns_none_and_keeps_state():
    from asp_backend_evaluation.ui.annotations import EdgeBuilder

    builder = EdgeBuilder()
    builder.add("asp", 0.1, 0.1)
    assert builder.finish("too soon") is None
    assert builder.count() == 1  # not cleared — nothing was actually finished


def test_edge_builder_reset_discards_pending_points():
    from asp_backend_evaluation.ui.annotations import EdgeBuilder

    builder = EdgeBuilder()
    builder.add("asp", 0.1, 0.1)
    builder.add("simple", 0.2, 0.2)
    builder.reset()
    assert builder.count() == 0
    assert builder.can_finish() is False


def test_edge_builder_mixes_points_and_regions():
    from asp_backend_evaluation.ui.annotations import EdgeBuilder

    builder = EdgeBuilder()
    builder.add("asp", 0.1, 0.1)  # a point
    builder.add("simple", 0.2, 0.2, 0.05, 0.05)  # a region
    edge = builder.finish("mixed")
    assert edge.points[0].is_region is False
    assert edge.points[1].is_region is True


# ---------------------------------------------------------------------------
# EdgeOverlay — renders a pending (unfinished) chain without crashing
# ---------------------------------------------------------------------------


def test_overlay_renders_a_pending_chain_across_panels(qapp):
    from asp_backend_evaluation.other.schema import EdgePoint
    from asp_backend_evaluation.ui.annotations import EdgeOverlay

    a = ImagePanel("asp", "ASP")
    b = ImagePanel("simple", "Simple")
    a.resize(200, 200)
    b.resize(200, 200)
    a.show()
    b.show()
    qapp.processEvents()
    a.set_image(_image())
    b.set_image(_image(300, 250, 50))
    overlay = EdgeOverlay()
    overlay.register_panels({"asp": a, "simple": b})
    overlay.resize(200, 200)
    overlay.set_pending([
        EdgePoint(image="asp", x=0.1, y=0.1),
        EdgePoint(image="simple", x=0.2, y=0.2, w=0.1, h=0.1),
    ])
    overlay.grab()  # must not raise
    a.close()
    b.close()


def test_restoring_bboxes_after_a_reload_does_not_accumulate(panel):
    from asp_backend_evaluation.other.schema import BoundingBox

    boxes = [BoundingBox(image="asp", x=0.1, y=0.1, w=0.2, h=0.2)]
    panel.restore_bboxes(boxes)
    first = len(panel._overlay_items)
    panel.clear_overlays()
    panel.restore_bboxes(boxes)
    assert len(panel._overlay_items) == first


def test_bboxes_for_other_panels_are_ignored(panel):
    from asp_backend_evaluation.other.schema import BoundingBox

    panel.restore_bboxes([BoundingBox(image="simple", x=0.1, y=0.1, w=0.2, h=0.2)])
    assert panel._overlay_items == []


def test_a_panel_with_no_image_is_inert(qapp):
    empty = ImagePanel("hugin", "Hugin")
    empty.set_image(None)
    assert empty.has_image() is False
    assert empty.image_size() is None
    assert empty.pixel_region(0, 0, 1, 1) is None
    assert empty.visible_region_norm() is None
    empty.fit_to_view()  # must not raise
    empty.set_zoom(4.0)


# ---------------------------------------------------------------------------
# defect 8 — N-way grid
# ---------------------------------------------------------------------------


@pytest.fixture()
def grid(qapp):
    widget = PanelGrid()
    widget.resize(1200, 400)
    widget.set_images({
        "asp": _image(), "simple": _image(420, 320, 100),
        "overmix": _image(500, 380, 110), "ground_truth": _image(440, 300, 120),
    })
    return widget


def test_grid_shows_every_available_comparator(grid):
    assert grid.available() == ["asp", "simple", "overmix", "ground_truth"]
    assert grid.visible() == ["asp", "simple", "overmix", "ground_truth"]


def test_grid_drops_comparators_a_test_lacks(grid):
    grid.set_images({"asp": _image(), "simple": _image()})
    assert grid.available() == ["asp", "simple"]
    assert grid.visible() == ["asp", "simple"]
    assert grid.panels["overmix"].has_image() is False


def test_visibility_selection_keeps_display_order(grid):
    grid.set_visible(["ground_truth", "asp"])
    assert grid.visible() == ["asp", "ground_truth"]


@pytest.mark.parametrize("mode", [LAYOUT_GRID, LAYOUT_STACK])
def test_layout_modes_apply(grid, mode):
    grid.set_layout_mode(mode)
    assert grid.layout_mode() == mode


def test_locking_mirrors_zoom_and_centre_across_panels(grid):
    grid.set_locked(True)
    grid.set_focus("asp")
    grid.panels["asp"].set_zoom(4.0)
    for key in grid.visible():
        assert grid.panels[key].zoom() == pytest.approx(4.0)


def test_locked_panels_share_a_normalized_centre_despite_different_canvases(grid):
    grid.set_locked(True)
    grid.set_focus("asp")
    grid.panels["asp"].set_zoom(6.0)
    centres = [grid.panels[k].center_norm() for k in grid.visible()]
    xs = [c[0] for c in centres]
    ys = [c[1] for c in centres]
    assert max(xs) - min(xs) < 0.1
    assert max(ys) - min(ys) < 0.1


def test_unlocking_leaves_other_panels_alone(grid):
    grid.set_locked(False)
    grid.panels["simple"].set_zoom(1.0)
    grid.panels["asp"].set_zoom(5.0)
    assert grid.panels["simple"].zoom() == pytest.approx(1.0)


def test_focus_cycles_only_through_visible_panels(grid):
    grid.set_visible(["asp", "simple"])
    grid.set_focus("asp")
    grid.cycle_focus(1)
    assert grid.focus_key() == "simple"
    grid.cycle_focus(1)
    assert grid.focus_key() == "asp"


def test_fit_all_resets_every_visible_panel(grid):
    grid.panels["asp"].set_zoom(8.0)
    grid.fit_all()
    assert all(grid.panels[k].zoom() == pytest.approx(1.0) for k in grid.visible())


# ---------------------------------------------------------------------------
# Reflow bug — followup feedback: unchecking comparators left a stale blank
# gap instead of the remaining panels filling the freed width. Root cause was
# QGridLayout remembering a stretch factor per column index for the life of
# the layout object; takeAt() removing items doesn't clear it.
# ---------------------------------------------------------------------------


def test_hiding_panels_clears_the_freed_columns_stretch(grid):
    """Regression test for the "large chunk empty when several stitcher
    engines are unselected" bug — also why unchecking Ground Truth looked
    like it did nothing, since the freed column stayed reserved as blank
    space instead of the remaining panels reflowing into it."""
    grid.set_layout_mode(LAYOUT_ROW)
    assert grid.visible() == ["asp", "simple", "overmix", "ground_truth"]
    for col in range(4):
        assert grid._grid.columnStretch(col) == 1

    grid.set_visible(["asp", "simple"])
    assert grid.visible() == ["asp", "simple"]
    # The two remaining columns keep their stretch; every column beyond them
    # (previously used by overmix/ground_truth) must be back to zero, or Qt
    # still reserves their share of width as a blank gap.
    assert grid._grid.columnStretch(0) == 1
    assert grid._grid.columnStretch(1) == 1
    for col in range(2, 4):
        assert grid._grid.columnStretch(col) == 0


def test_ground_truth_checkbox_off_reflows_the_remaining_panels(grid):
    """The same bug from the user's perspective: hiding Ground Truth must
    make the other panels visibly take up the freed width, not just remove
    the widget while leaving a static-sized gap."""
    grid.resize(1200, 400)
    grid.set_visible(["asp", "simple", "overmix", "ground_truth"])
    grid.show()
    for col in range(4):
        assert grid._grid.columnStretch(col) == 1

    grid.set_visible(["asp", "simple", "overmix"])
    assert "ground_truth" not in grid.visible()
    assert grid.cells["ground_truth"].isVisible() is False
    for col in range(3):
        assert grid._grid.columnStretch(col) == 1
    assert grid._grid.columnStretch(3) == 0


# ---------------------------------------------------------------------------
# Drag-to-reorder panels
# ---------------------------------------------------------------------------


def test_reorder_moves_source_into_targets_slot(grid):
    grid.reorder("ground_truth", "asp")
    assert grid.order() == ["ground_truth", "asp", "simple", "overmix", "hugin"]
    assert grid.visible() == ["ground_truth", "asp", "simple", "overmix"]


def test_reorder_lets_ground_truth_sit_between_two_stitchers(grid):
    """The concrete case requested: GT between two stitcher outputs instead
    of always trailing on the right."""
    grid.reorder("ground_truth", "simple")
    assert grid.order().index("asp") < grid.order().index("ground_truth") < grid.order().index("simple")


def test_reorder_is_a_noop_for_identical_keys(grid):
    before = grid.order()
    grid.reorder("asp", "asp")
    assert grid.order() == before


@pytest.mark.parametrize("source,target", [("nope", "asp"), ("asp", "nope"), ("nope", "also_nope")])
def test_reorder_ignores_unknown_keys(grid, source, target):
    before = grid.order()
    grid.reorder(source, target)
    assert grid.order() == before


def test_reorder_updates_the_actual_grid_layout_positions(grid):
    grid.set_layout_mode(LAYOUT_ROW)
    grid.reorder("ground_truth", "asp")
    columns = [
        grid._grid.itemAtPosition(0, c).widget().key
        for c in range(grid._grid.columnCount())
        if grid._grid.itemAtPosition(0, c)
    ]
    assert columns == ["ground_truth", "asp", "simple", "overmix"]


def test_custom_order_persists_across_a_new_tests_images(grid):
    """Panel order is a session-wide preference, not a per-test fact — it
    must survive navigating to the next test in the queue."""
    grid.reorder("ground_truth", "asp")
    grid.set_images({"asp": _image(), "simple": _image(), "overmix": _image()})
    assert grid.order()[0] == "ground_truth"
    assert grid.visible() == ["asp", "simple", "overmix"]  # this test has no GT


def test_set_order_replaces_and_appends_missing_keys(grid):
    grid.set_order(["ground_truth", "simple"])
    assert grid.order() == ["ground_truth", "simple", "asp", "overmix", "hugin"]


def test_set_order_ignores_unknown_keys(grid):
    grid.set_order(["ground_truth", "not_a_real_key", "asp"])
    assert grid.order() == ["ground_truth", "asp", "simple", "overmix", "hugin"]


def test_order_changed_signal_fires_on_reorder(grid):
    seen = []
    grid.orderChanged.connect(seen.append)
    grid.reorder("ground_truth", "asp")
    assert seen and seen[0][0] == "ground_truth"


def test_order_changed_signal_does_not_fire_on_a_noop_reorder(grid):
    seen = []
    grid.orderChanged.connect(seen.append)
    grid.reorder("asp", "asp")
    assert seen == []


# ---------------------------------------------------------------------------
# Theme cascade (issue #123 followup — light/dark Settings toggle)
#
# ``set_focused``/``score_chip_style`` build inline (non-cascaded) stylesheets
# per instance, so a theme switch needs each widget to re-derive its own style
# from ``current_palette()`` explicitly — these pin that the cascade actually
# reaches every already-built cell/button rather than leaving it stuck on
# whatever palette was active at construction time.
# ---------------------------------------------------------------------------


@pytest.fixture()
def restore_dark_theme(qapp):
    from asp_backend_evaluation.ui.theme import apply_theme

    yield
    apply_theme(qapp, "dark")  # theme is module-level state; don't leak into other tests


def test_grid_refresh_theme_updates_the_focused_chips_inline_style(grid, restore_dark_theme):
    from asp_backend_evaluation.ui.theme import apply_theme, current_palette

    grid.set_focus("asp")
    apply_theme(grid, "light")
    grid.refresh_theme()
    light_accent = current_palette()["accent"]
    assert light_accent in grid.cells["asp"].title_label.styleSheet()


def test_grid_refresh_theme_also_updates_unfocused_cells(grid, restore_dark_theme):
    from asp_backend_evaluation.ui.theme import apply_theme, current_palette

    grid.set_focus("asp")
    apply_theme(grid, "light")
    grid.refresh_theme()
    light_border = current_palette()["border"]
    assert light_border in grid.cells["simple"].title_label.styleSheet()


# ---------------------------------------------------------------------------
# Scoring panel
# ---------------------------------------------------------------------------


@pytest.fixture()
def scoring(qapp):
    widget = ScoringPanel()
    widget.set_comparators(["asp", "simple", "overmix"])
    from asp_backend_evaluation.other.schema import RatingEntry

    widget.load_entry(RatingEntry())
    return widget


def test_keyboard_scoring_updates_the_entry(scoring):
    assert scoring.score_focused("asp", 3) is True
    assert scoring.entry().asp == 3
    assert scoring.blocks["asp"].rows["coherence"].score() == 3


def test_scoring_an_absent_comparator_reports_failure(scoring):
    assert scoring.score_focused("hugin", 3) is False


def test_only_coherence_is_required(scoring):
    assert scoring.missing_required() == ["ASP coherence", "Simple coherence"]
    scoring.score_focused("asp", 2)
    scoring.score_focused("simple", 4)
    assert scoring.missing_required() == []


def test_optional_dimensions_do_not_block_completion(scoring):
    scoring.score_focused("asp", 2)
    scoring.score_focused("simple", 4)
    scoring.score_focused("asp", 1, dimension="sharpness")
    assert scoring.missing_required() == []
    assert scoring.entry().score("asp", "sharpness") == 1


def test_preference_toggles_off_when_reselected(scoring):
    scoring.set_preference("asp")
    assert scoring.entry().preference == "asp"
    scoring.set_preference("asp", toggle=True)
    assert scoring.entry().preference is None


def test_defect_tags_toggle_by_index(scoring):
    key = scoring.toggle_defect_index(0)
    assert key == "torn_anatomy"
    assert scoring.entry().defects == ["torn_anatomy"]
    scoring.toggle_defect_index(0)
    assert scoring.entry().defects == []


def test_out_of_range_defect_index_is_ignored(scoring):
    assert scoring.toggle_defect_index(99) is None


def test_collapsed_dimensions_still_show_a_scored_row(scoring):
    """Collapsing must never hide recorded data.

    Asserted with isHidden() rather than isVisible(): the latter is False for
    any widget whose ancestors aren't shown, so on an unshown panel it would
    pass for the wrong reason. isHidden() reflects the explicit setVisible state.
    """
    scoring.score_focused("asp", 3, dimension="seams")
    scoring.blocks["asp"].set_detailed(False)
    assert scoring.blocks["asp"].rows["seams"].isHidden() is False
    assert scoring.blocks["asp"].rows["framing"].isHidden() is True
    assert scoring.blocks["asp"].rows["coherence"].isHidden() is False


def test_expanding_dimensions_shows_every_row(scoring):
    scoring.blocks["asp"].set_detailed(True)
    assert all(not row.isHidden() for row in scoring.blocks["asp"].rows.values())


def test_loading_an_entry_does_not_emit_changes(scoring):
    from asp_backend_evaluation.other.schema import RatingEntry

    seen = []
    scoring.changed.connect(lambda: seen.append(1))
    entry = RatingEntry()
    entry.set_score("asp", "coherence", 4)
    scoring.load_entry(entry)
    assert seen == []


def test_comparators_are_rebuilt_per_test(scoring):
    scoring.set_comparators(["asp", "simple"])
    assert set(scoring.blocks) == {"asp", "simple"}


def test_scoring_panel_refresh_theme_updates_every_chip(scoring, restore_dark_theme):
    from asp_backend_evaluation.ui.theme import apply_theme, current_palette

    apply_theme(scoring, "light")
    scoring.refresh_theme()
    light_border = current_palette()["border"]
    for block in scoring.blocks.values():
        for row in block.rows.values():
            for btn in row._buttons:
                assert light_border in btn.styleSheet()


def test_scoring_panel_refresh_theme_updates_optional_dimension_label(scoring, restore_dark_theme):
    from asp_backend_evaluation.ui.theme import apply_theme, current_palette

    apply_theme(scoring, "light")
    scoring.refresh_theme()
    light_dim = current_palette()["text_dim"]
    seams_row = scoring.blocks["asp"].rows["seams"]
    assert light_dim in seams_row._name_label.styleSheet()
