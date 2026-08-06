"""
backend/benchmark/evaluation/test/test_inspector_resize.py
==========================================================
Integration tests for InspectorWindow resize debounce and geometry overflow.

These are the primary regression guards for issue #153 — "InspectorWindow UI
overflows window boundaries after maximize/resize".

All three root causes are exercised:

  Root cause #1 — size policies (see test_image_panel_geometry.py and
    test_panel_grid_geometry.py which cover this in isolation).

  Root cause #2 — QTimer.singleShot(0) bypassing _resize_timer:
    Verified by TestScheduleFit — checks that _schedule_fit never shortens
    an active resize debounce, and that rapid concurrent calls collapse into
    a single _fit_all invocation.

  Root cause #3 — _fit_all re-entrancy:
    Verified by TestFitAllReentrancy — checks that a nested _fit_all call
    (e.g. triggered by a QGraphicsView internal geometry event) is a no-op.

  Root cause #4 — QStackedWidget size policies (see test_panel_grid_geometry.py).

  Root cause #5 — QGraphicsView internal updateGeometry propagation:
    Covered by TestInspectorWindowGeometry which builds the full window and
    verifies the window geometry never exceeds the screen after a simulated
    resize + fit cycle.

Run:
    pytest backend/benchmark/evaluation/test/test_inspector_resize.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

pytestmark = pytest.mark.inspector_ui


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid(h: int = 1200, w: int = 800, val: int = 128) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def _process(n: int = 5) -> None:
    """Drain the Qt event queue *n* times to let deferred timers fire."""
    for _ in range(n):
        QApplication.processEvents()


# ---------------------------------------------------------------------------
# _schedule_fit — regression guard for root cause #2 of issue #153
# ---------------------------------------------------------------------------

class TestScheduleFit:
    """_schedule_fit must never shorten an active resize debounce.

    The core contract: if _resize_timer has N ms remaining and
    _schedule_fit(0) is called, the timer must NOT be shortened to 0ms.
    This prevents singleShot(0) equivalents from firing _fit_all while the
    window manager is still mid-animation.
    """

    def _make_window(self, tmp_path: str):
        """Build an InspectorWindow with an empty corpus directory."""
        from asp_backend.evaluation.ui.main_window import InspectorWindow

        return InspectorWindow(
            base_dir=tmp_path,
            out_path=os.path.join(tmp_path, "eval.json"),
            redo=False,
            repo_root=tmp_path,
        )

    def test_schedule_fit_does_not_shorten_active_debounce(self, qapp, tmp_path):
        """If a 150ms resize debounce is counting down, _schedule_fit(0) is a no-op."""
        window = self._make_window(str(tmp_path))

        # Simulate: resizeEvent was just received — starts the 150ms debounce.
        window._resize_timer.start(150)
        remaining_before = window._resize_timer.remainingTime()
        assert remaining_before > 0

        # Now call _schedule_fit(0) — as if a dataset-load completed during
        # the resize animation.
        fit_call_count = [0]
        original_fit = window._fit_all
        def counting_fit():
            fit_call_count[0] += 1
            original_fit()
        window._fit_all = counting_fit
        window._resize_timer.timeout.disconnect()
        window._resize_timer.timeout.connect(counting_fit)

        window._schedule_fit(0)

        # The timer should still have remaining time (not reset to 0).
        remaining_after = window._resize_timer.remainingTime()
        assert remaining_after > 0, (
            "_schedule_fit(0) must not shorten an active 150ms resize debounce. "
            "If it does, _fit_all fires mid-animation and causes window overflow."
        )

        window.close()

    def test_schedule_fit_fires_when_no_debounce_active(self, qapp, tmp_path):
        """If no timer is running, _schedule_fit(0) starts one immediately."""
        window = self._make_window(str(tmp_path))

        assert not window._resize_timer.isActive(), "Timer should be idle initially."

        window._schedule_fit(0)

        assert window._resize_timer.isActive(), (
            "_schedule_fit(0) must start _resize_timer when no debounce is active."
        )

        window._resize_timer.stop()
        window.close()

    def test_multiple_schedule_fit_calls_collapse_into_one(self, qapp, tmp_path):
        """Rapid _schedule_fit calls must collapse into a single _fit_all call."""
        window = self._make_window(str(tmp_path))

        fit_count = [0]
        original_fit = window._fit_all
        def counting_fit():
            fit_count[0] += 1
            original_fit()

        # Patch _fit_all via the timer signal (schedule_fit goes through _resize_timer)
        window._resize_timer.timeout.disconnect()
        window._resize_timer.timeout.connect(counting_fit)

        # Fire 5 rapid schedule_fit(0) calls (simulating 5 dataset-related events)
        for _ in range(5):
            window._schedule_fit(0)

        # Let the timer fire exactly once
        _process(10)
        QApplication.instance().processEvents()

        assert fit_count[0] <= 1, (
            f"_fit_all was called {fit_count[0]} times for 5 rapid _schedule_fit(0) "
            f"calls — it must collapse into at most 1 call via QTimer debounce."
        )

        window.close()


# ---------------------------------------------------------------------------
# _fit_all re-entrancy guard — root cause #5 of issue #153
# ---------------------------------------------------------------------------

class TestFitAllReentrancy:
    """_fit_all must be re-entrancy safe via the _fitting guard."""

    def _make_window(self, tmp_path: str):
        from asp_backend.evaluation.ui.main_window import InspectorWindow
        return InspectorWindow(
            base_dir=tmp_path,
            out_path=os.path.join(tmp_path, "eval.json"),
            redo=False,
            repo_root=tmp_path,
        )

    def test_fitting_flag_starts_false(self, qapp, tmp_path):
        window = self._make_window(str(tmp_path))
        assert window._fitting is False
        window.close()

    def test_fitting_flag_is_reset_after_fit_all(self, qapp, tmp_path):
        window = self._make_window(str(tmp_path))
        window._fit_all()
        _process()
        assert window._fitting is False, (
            "_fitting flag must be reset to False after _fit_all returns."
        )
        window.close()

    def test_reentrant_fit_all_is_noop(self, qapp, tmp_path):
        """Calling _fit_all while _fitting=True must be a no-op."""
        window = self._make_window(str(tmp_path))

        call_count = [0]
        original_grid_fit = window.grid.fit_all

        def guarded_grid_fit():
            call_count[0] += 1
            original_grid_fit()

        window.grid.fit_all = guarded_grid_fit

        # Simulate re-entrancy: set _fitting=True then call _fit_all again.
        window._fitting = True
        window._fit_all()
        _process()

        assert call_count[0] == 0, (
            "Re-entrant _fit_all call (with _fitting=True) must not invoke grid.fit_all(). "
            "This guard prevents double-transform application from QGraphicsView internals."
        )

        # Reset and verify normal call works
        window._fitting = False
        window._fit_all()
        assert call_count[0] == 1, "Non-reentrant _fit_all must invoke grid.fit_all()."

        window.close()


# ---------------------------------------------------------------------------
# InspectorWindow geometry — full-stack overflow regression
# ---------------------------------------------------------------------------

class TestInspectorWindowGeometry:
    """Full-window geometry tests — verify the window never expands past screen.

    These tests construct the complete InspectorWindow widget (under offscreen
    QPA) and simulate resize events, verifying that:
      - The window does not grow wider than its own set size after fit_all.
      - The QSplitter's total matches the window's client area.
    """

    def _make_window(self, tmp_path: str, w: int = 1200, h: int = 800):
        from asp_backend.evaluation.ui.main_window import InspectorWindow

        window = InspectorWindow(
            base_dir=tmp_path,
            out_path=os.path.join(tmp_path, "eval.json"),
            redo=False,
            repo_root=tmp_path,
        )
        window.resize(w, h)
        _process()
        return window

    def test_window_width_does_not_grow_after_fit_all(self, qapp, tmp_path):
        """After fit_all, window width must not exceed the size we set."""
        window = self._make_window(str(tmp_path), w=1200, h=800)
        window._fit_all()
        _process()

        assert window.width() <= 1202, (
            f"Window grew beyond set width after _fit_all: {window.width()} > 1200. "
            "This is the primary symptom of issue #153."
        )
        window.close()

    def test_window_stable_through_multiple_resize_cycles(self, qapp, tmp_path):
        """Window width must remain stable across several resize + fit_all cycles."""
        window = self._make_window(str(tmp_path))

        for target_w, target_h in [(800, 600), (1400, 900), (1000, 700), (600, 500)]:
            window.resize(target_w, target_h)
            _process()
            window._fit_all()
            _process()

            assert window.width() <= target_w + 2, (
                f"Window width grew beyond {target_w} after resize+fit: {window.width()}"
            )

        window.close()

    def test_splitter_sizes_sum_to_window_client_width(self, qapp, tmp_path):
        """The body_splitter child sizes must sum close to the available width.

        A mismatch here (splitter wider than window) would be another indicator
        of the layout overflow that caused issue #153.
        """
        window = self._make_window(str(tmp_path), w=1200, h=800)
        window._fit_all()
        _process()

        splitter_total = sum(window.body_splitter.sizes())
        available = window.body_splitter.width()

        assert abs(splitter_total - available) <= 10, (
            f"body_splitter sizes sum ({splitter_total}) != splitter width ({available}). "
            "A large discrepancy indicates the splitter expanded beyond the window."
        )
        window.close()

    @pytest.mark.regression
    def test_window_does_not_overflow_after_simulated_maximize(self, qapp, tmp_path):
        """Simulated maximize: resize to a large size, then run the debounce timer.

        This reproduces the exact sequence that triggered issue #153:
          1. Window is resized (maximize).
          2. _resize_timer fires (150ms debounce).
          3. _fit_all is called.

        The window must not grow beyond the target size at step 3.
        """
        from PySide6.QtGui import QGuiApplication

        screen = QGuiApplication.primaryScreen()
        if screen is None:
            pytest.skip("No screen available (offscreen QPA may not report screen geometry)")

        screen_w = screen.geometry().width() or 1920
        screen_h = screen.geometry().height() or 1080

        window = self._make_window(str(tmp_path), w=800, h=600)

        # Step 1: simulate maximize by setting to screen size
        window.resize(screen_w, screen_h)
        _process()
        size_after_resize = window.width()

        # Step 2: simulate _resize_timer firing (the debounce completing)
        window._fit_all()
        _process()
        size_after_fit = window.width()

        assert size_after_fit <= size_after_resize + 2, (
            f"Window grew from {size_after_resize}px to {size_after_fit}px after "
            f"_fit_all was called post-maximize — this is the exact symptom of issue #153."
        )
        window.close()


# ---------------------------------------------------------------------------
# Root cause #6 — unwrapped footer/header labels forcing a wide minimum size
# ---------------------------------------------------------------------------

class TestElidingLabels:
    """The footer key-hint crib sheet (``KEY_HINTS``) and the header/footer
    status labels used a plain ``QLabel`` with word-wrap off and no elision.
    A ``QLabel`` in that configuration reports its full unwrapped text width
    as its ``minimumSizeHint`` — for the joined ``KEY_HINTS`` string that is
    ~1680px, which propagated up through the footer row into
    ``QMainWindow``'s minimum size and forced the window wider than most
    screens, independent of (and unfixed by) the panel-grid/scoring-sidebar
    size-policy fixes for issue #153's original five root causes.
    """

    def _make_window(self, tmp_path: str, w: int = 1200, h: int = 800):
        from asp_backend.evaluation.ui.main_window import InspectorWindow

        window = InspectorWindow(
            base_dir=tmp_path,
            out_path=os.path.join(tmp_path, "eval.json"),
            redo=False,
            repo_root=tmp_path,
        )
        window.resize(w, h)
        _process()
        return window

    @pytest.mark.regression
    def test_window_minimum_width_is_small(self, qapp, tmp_path):
        """The window's minimum width must stay small — the full KEY_HINTS
        text (~1680px unwrapped) must never set the floor for how narrow the
        window can go."""
        window = self._make_window(str(tmp_path))
        min_width = window.minimumSizeHint().width()

        assert min_width < 800, (
            f"InspectorWindow minimumSizeHint width is {min_width}px — the footer "
            "hints label is forcing a wide minimum again (issue #153 root cause #6)."
        )
        window.close()

    def test_window_can_shrink_below_key_hints_text_width(self, qapp, tmp_path):
        """A window narrower than the full KEY_HINTS text must actually apply
        — the label must elide instead of clamping the window wider."""
        window = self._make_window(str(tmp_path), w=1200, h=800)
        window.resize(700, 600)
        _process()

        assert window.width() <= 702, (
            f"Window could not shrink to 700px (stuck at {window.width()}px) — "
            "an unwrapped/un-elided label is still forcing a wide minimum."
        )
        window.close()

    def test_hints_label_elides_when_narrow(self, qapp, tmp_path):
        """The rendered hints text must shrink to fit the label's own width,
        not overflow it — the full crib sheet remains available as a tooltip."""
        window = self._make_window(str(tmp_path), w=1200, h=800)
        from asp_backend.evaluation.constants.user_interface import KEY_HINTS

        full_text = "   ".join(f"{k}: {v}" for k, v in KEY_HINTS)
        hints_label = None
        for lbl in window.findChildren(type(window.pixel_label)):
            if lbl.toolTip() == full_text:
                hints_label = lbl
        assert hints_label is not None, "Could not find the footer hints label."

        hints_label.resize(150, 15)
        _process()

        assert len(hints_label.text()) < len(full_text), (
            "Hints label did not elide at a narrow width — its rendered text "
            "is as long as (or longer than) the full crib sheet."
        )
        assert hints_label.toolTip() == full_text, (
            "The full crib sheet must remain available via tooltip after eliding."
        )
        window.close()
        window.close()
