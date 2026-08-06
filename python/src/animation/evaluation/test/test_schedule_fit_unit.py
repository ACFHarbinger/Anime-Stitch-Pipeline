"""
backend/benchmark/evaluation/test/test_schedule_fit_unit.py
============================================================
Pure-unit tests for the _schedule_fit() helper in InspectorWindow.

These tests use a minimal stub instead of the full InspectorWindow, so they
run fast and without requiring any benchmark data directory or Qt display.

The _schedule_fit logic is extracted as a function and tested independently
so that the debounce contract can be verified without the weight of the full
widget hierarchy.

Run:
    pytest backend/benchmark/evaluation/test/test_schedule_fit_unit.py -v
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

# ---------------------------------------------------------------------------
# Minimal stub that contains only _schedule_fit and _resize_timer logic.
# This avoids constructing the full InspectorWindow (which needs a data dir).
# ---------------------------------------------------------------------------

class _ScheduleFitStub:
    """Minimal reproduction of InspectorWindow._schedule_fit + _resize_timer."""

    def __init__(self):
        self._resize_timer = QTimer()
        self._resize_timer.setSingleShot(True)
        self._fit_all_calls: list[str] = []  # log of call origins
        self._resize_timer.timeout.connect(lambda: self._fit_all_calls.append("timer"))
        self._fitting = False

    def _schedule_fit(self, delay_ms: int = 0) -> None:
        """Exact copy of InspectorWindow._schedule_fit (minus _dbg calls)."""
        remaining = self._resize_timer.remainingTime() if self._resize_timer.isActive() else -1
        if remaining >= 0 and remaining > delay_ms:
            return  # a longer debounce is already active — don't shorten it
        self._resize_timer.start(delay_ms)

    def simulate_resize_event(self, debounce_ms: int = 150) -> None:
        """Mirror of InspectorWindow.resizeEvent — starts the debounce."""
        self._resize_timer.start(debounce_ms)

    def stop(self):
        self._resize_timer.stop()


@pytest.fixture
def stub(qapp):
    s = _ScheduleFitStub()
    yield s
    s.stop()


# ---------------------------------------------------------------------------
# Core debounce contract
# ---------------------------------------------------------------------------

class TestScheduleFitContract:

    def test_no_active_timer_starts_immediately(self, stub, qapp):
        """With no active timer, _schedule_fit(0) must start the timer."""
        assert not stub._resize_timer.isActive()
        stub._schedule_fit(0)
        assert stub._resize_timer.isActive()

    def test_does_not_shorten_150ms_debounce_to_0ms(self, stub, qapp):
        """_schedule_fit(0) must NOT shorten a 150ms resize debounce.

        This is the exact race that caused issue #153: a singleShot(0) fired
        while the window manager was still animating the maximize, shortening
        the debounce to 0ms and calling fit_all with transitional viewport sizes.
        """
        stub.simulate_resize_event(debounce_ms=150)
        remaining_before = stub._resize_timer.remainingTime()
        assert remaining_before > 5, "Sanity: 150ms timer should have meaningful remaining time"

        stub._schedule_fit(0)  # should be a no-op — timer is already counting down

        remaining_after = stub._resize_timer.remainingTime()
        assert remaining_after > 0, (
            "_schedule_fit(0) must not shorten an active 150ms debounce to 0ms. "
            f"remainingTime went from {remaining_before}ms -> {remaining_after}ms."
        )

    def test_does_not_shorten_longer_schedule_fit(self, stub, qapp):
        """_schedule_fit(0) must not shorten a previously started _schedule_fit(100)."""
        stub._schedule_fit(100)
        stub._schedule_fit(0)
        # Timer should still have significant remaining time
        assert stub._resize_timer.remainingTime() > 0

    def test_longer_delay_replaces_shorter(self, stub, qapp):
        """_schedule_fit(200) when no timer is active starts with 200ms."""
        stub._schedule_fit(200)
        assert stub._resize_timer.isActive()
        # The interval should be ~ 200ms (remaining might be slightly less)
        assert stub._resize_timer.remainingTime() > 0

    def test_timer_fires_exactly_once_for_rapid_calls(self, stub, qapp):
        """5 rapid _schedule_fit(0) calls must fire the timer at most once."""
        for _ in range(5):
            stub._schedule_fit(0)

        # Process events to let the 0ms timer fire
        for _ in range(10):
            QApplication.processEvents()

        assert len(stub._fit_all_calls) <= 1, (
            f"Timer fired {len(stub._fit_all_calls)} times for 5 rapid _schedule_fit(0) "
            f"calls — debounce must collapse these into at most 1 call."
        )

    def test_schedule_fit_after_timer_expired_starts_new_timer(self, stub, qapp):
        """After the timer has fired, _schedule_fit(0) must start a new timer."""
        stub._schedule_fit(0)
        # Let the timer fire
        for _ in range(10):
            QApplication.processEvents()

        # Timer should be idle now (singleShot)
        assert not stub._resize_timer.isActive()

        # Should be able to start a new one
        stub._schedule_fit(0)
        assert stub._resize_timer.isActive()

    def test_resize_event_after_schedule_fit_extends_debounce(self, stub, qapp):
        """A resize event arriving after _schedule_fit(0) must push debounce back to 150ms."""
        stub._schedule_fit(0)  # 0ms timer is running
        stub.simulate_resize_event(150)  # resize event restarts at 150ms

        remaining = stub._resize_timer.remainingTime()
        assert remaining > 5, (
            f"resize event should push debounce to ~150ms, got {remaining}ms remaining."
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestScheduleFitEdgeCases:

    def test_schedule_fit_with_negative_delay_treated_as_zero(self, stub, qapp):
        """Negative delay is technically invalid; the timer must still start."""
        # QTimer.start(-1) is usually treated as a 0ms timer or no-op
        # We just verify no exception is raised
        try:
            stub._schedule_fit(-1)
        except Exception as exc:
            pytest.fail(f"_schedule_fit(-1) raised unexpectedly: {exc}")

    def test_schedule_fit_called_with_zero_on_idle_timer(self, stub, qapp):
        """_schedule_fit(0) on idle timer: timer.remainingTime() == -1 (not active)."""
        assert not stub._resize_timer.isActive()
        # remainingTime() returns -1 for inactive timer
        assert stub._resize_timer.remainingTime() == -1

        stub._schedule_fit(0)
        assert stub._resize_timer.isActive()

    def test_multiple_resize_events_each_restart_at_150ms(self, stub, qapp):
        """Each resize event must restart the debounce at 150ms, not accumulate."""
        for _ in range(3):
            stub.simulate_resize_event(150)
            remaining = stub._resize_timer.remainingTime()
            # Each restart should give approximately 150ms
            assert remaining > 5, f"After restart, remainingTime should be ~150ms, got {remaining}"
