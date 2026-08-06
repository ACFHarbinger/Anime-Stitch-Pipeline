"""Tests for the Hybrid Stitch first-run onboarding wizard (roadmap Phase 6.1,
issue #1): a dismissible, re-invocable ``QWizard`` walkthrough over
``RealHybridStitchPanel``.
"""

from __future__ import annotations

import pytest
from asp_gui.tabs.stencil.hybrid_stitch_panel import RealHybridStitchPanel
from asp_gui.tabs.stencil.onboarding_wizard import (
    HybridStitchOnboardingWizard,
    hybrid_stitch_onboarding_seen,
    mark_hybrid_stitch_onboarding_seen,
)
from gui.src.windows.settings.app_settings import AppSettings

pytestmark = pytest.mark.gui


class _FakeSettingsStore:
    """In-memory stand-in for QSettings so tests never touch real user
    settings on disk, following this file's own monkeypatch convention
    (see gui/test/dialogs/test_directory_import_dialog.py)."""

    def __init__(self):
        self._data: dict = {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value


@pytest.fixture
def fake_settings(monkeypatch):
    store = _FakeSettingsStore()
    monkeypatch.setattr(AppSettings, "get", staticmethod(store.get))
    monkeypatch.setattr(AppSettings, "set", staticmethod(store.set))
    return store


class TestOnboardingSeenFlag:
    def test_unseen_by_default(self, fake_settings):
        assert hybrid_stitch_onboarding_seen() is False

    def test_mark_seen_persists(self, fake_settings):
        mark_hybrid_stitch_onboarding_seen()
        assert hybrid_stitch_onboarding_seen() is True

    def test_string_true_value_is_treated_as_seen(self, fake_settings):
        # QSettings backends (e.g. ini format) can round-trip booleans as
        # the strings "true"/"false" -- mirror AppSettings.recursive_scan's
        # own handling of that case.
        fake_settings.set("onboarding/hybrid_stitch_panel_seen", "true")
        assert hybrid_stitch_onboarding_seen() is True


class TestHybridStitchOnboardingWizard:
    def test_has_a_page_per_panel_section(self, q_app):
        wizard = HybridStitchOnboardingWizard(panel=None)
        # Welcome + sidebar + 5 tool tabs (Control Points, Color Correct,
        # Seam Painter, Mesh Warp, Render).
        assert wizard.pageIds() == list(range(7))

    def test_pages_reference_real_tool_tab_names(self, q_app):
        wizard = HybridStitchOnboardingWizard(panel=None)
        tab_names = set(wizard._tab_for_page.values())
        assert tab_names == {
            "Control Points",
            "Color Correct",
            "Seam Painter",
            "Mesh Warp",
            "Render",
        }

    def test_changing_page_switches_the_live_panel_tab(self, q_app):
        panel = RealHybridStitchPanel()
        wizard = HybridStitchOnboardingWizard(panel=panel, parent=panel)

        seam_page_id = next(
            pid for pid, name in wizard._tab_for_page.items() if name == "Seam Painter"
        )
        wizard._on_page_changed(seam_page_id)
        assert panel._tools.tabText(panel._tools.currentIndex()) == "Seam Painter"

    def test_dismissing_marks_seen(self, q_app, fake_settings):
        panel = RealHybridStitchPanel()
        assert hybrid_stitch_onboarding_seen() is False
        panel._show_onboarding_wizard()
        assert panel._onboarding_wizard is not None
        panel._onboarding_wizard.reject()  # Cancel / dismiss at any step
        assert hybrid_stitch_onboarding_seen() is True
        assert panel._onboarding_wizard is None


class TestHybridStitchPanelOnboardingIntegration:
    def test_help_button_present_and_reinvokes_wizard(self, q_app, fake_settings):
        mark_hybrid_stitch_onboarding_seen()  # simulate a returning user
        panel = RealHybridStitchPanel()
        assert panel._onboarding_wizard is None  # no auto-open on this call path
        panel._help_btn.click()
        assert panel._onboarding_wizard is not None
        panel._onboarding_wizard.reject()

    def test_first_run_auto_opens_wizard(self, q_app, fake_settings):
        panel = RealHybridStitchPanel()
        assert hybrid_stitch_onboarding_seen() is False
        panel._maybe_show_first_run_onboarding()
        assert panel._onboarding_wizard is not None
        panel._onboarding_wizard.reject()

    def test_second_run_does_not_auto_open_wizard(self, q_app, fake_settings):
        mark_hybrid_stitch_onboarding_seen()
        panel = RealHybridStitchPanel()
        panel._maybe_show_first_run_onboarding()
        assert panel._onboarding_wizard is None
