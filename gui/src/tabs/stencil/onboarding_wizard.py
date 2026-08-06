"""First-run guided walkthrough for :class:`RealHybridStitchPanel` (roadmap
Phase 6.1 — "In-app guided walkthrough").

A dismissible, re-invocable ``QWizard`` that tours a new user through the
manual Hybrid Stitch tool: the frame-sequence sidebar, the working-pair
selector, and each tool tab (Control Points / Color Correct / Seam Painter /
Mesh Warp / Render). As the user pages through, the panel's own
``QTabWidget`` is switched to the matching tab so the tour points at the
real, live UI rather than a static screenshot.

Persistence follows the same ``AppSettings`` convention used elsewhere in
this codebase (see ``gui/src/elements/_thumbnail_file_picker.py``'s
favourites/theme reads) rather than inventing a new settings mechanism.
"""

from __future__ import annotations

from gui.src.windows.settings.app_settings import AppSettings
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

# ``AppSettings.get``/``set`` generic namespace -- no typed accessor exists
# for onboarding flags yet, so this follows the "raw access" convention
# documented at the bottom of app_settings.py.
_SEEN_KEY = "onboarding/hybrid_stitch_panel_seen"


def hybrid_stitch_onboarding_seen() -> bool:
    """Return True if the user already dismissed/completed the tour."""
    val = AppSettings.get(_SEEN_KEY, False)
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return bool(val)


def mark_hybrid_stitch_onboarding_seen() -> None:
    """Persist that the tour has been shown, so it won't auto-open again."""
    AppSettings.set(_SEEN_KEY, True)


def _make_page(title: str, subtitle: str, body: str) -> QWizardPage:
    page = QWizardPage()
    page.setTitle(title)
    page.setSubTitle(subtitle)
    layout = QVBoxLayout(page)
    label = QLabel(body)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    layout.addWidget(label)
    layout.addStretch()
    return page


class HybridStitchOnboardingWizard(QWizard):
    """Guided first-run tour of the Hybrid Stitch panel.

    Dismissible at any step (Cancel / window close both count as
    "dismissed") and safe to reopen at any time -- it never blocks normal
    use of the panel underneath it.
    """

    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self._panel = panel
        self.setWindowTitle("Hybrid Stitch — Guided Tour")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self.resize(520, 360)

        self._tab_for_page: dict[int, str] = {}

        self._add_page(
            _make_page(
                "Welcome to Hybrid Stitch",
                "A manual, human-in-the-loop stitching tool.",
                "<p>Hybrid Stitch lets you build or correct a panorama stitch "
                "by hand when the automated pipeline needs help: pick control "
                "points, fix color mismatches, paint seams, and warp a mesh "
                "&mdash; one frame pair at a time.</p>"
                "<p>This short tour walks through each part of the panel. "
                "You can dismiss it at any step &mdash; it won't block "
                "anything &mdash; and reopen it later from the <b>?</b> "
                "button in the panel's toolbar.</p>",
            ),
            None,
        )
        self._add_page(
            _make_page(
                "1. Sequence sidebar",
                "Build the frame sequence you'll stitch.",
                "<p>On the left, <b>Add Frames…</b> loads images into the "
                "sequence list. Drag rows to reorder them, or use "
                "<b>Move Up ↑</b> / <b>Move Down ↓</b>; <b>Remove Selected</b> "
                "and <b>Clear All</b> manage the list.</p>"
                "<p>Below it, <b>Working Pair</b> picks which two frames "
                "(Frame A and Frame B) the tool tabs on the right operate on "
                "&mdash; click <b>Load Pair →</b> to send them to the "
                "editor.</p>",
            ),
            None,
        )
        self._add_page(
            _make_page(
                "2. Control Points",
                "Manually align a frame pair.",
                "<p>Click matching landmarks on Frame A and Frame B to build "
                "a set of correspondences; the editor solves a homography "
                "from them. Once the alignment looks right, use "
                "<b>✔ Accept H</b> in the sidebar to save it for this pair.</p>",
            ),
            "Control Points",
        )
        self._add_page(
            _make_page(
                "3. Color Correct",
                "Match brightness, contrast, and tone across the seam.",
                "<p>Adjust per-frame brightness/contrast/saturation so Frame "
                "A and Frame B blend smoothly at the seam &mdash; useful "
                "when scans or captures were taken under different exposure "
                "or gain.</p>",
            ),
            "Color Correct",
        )
        self._add_page(
            _make_page(
                "4. Seam Painter",
                "Paint exactly where the seam should run.",
                "<p>Once a homography is solved, paint the boundary between "
                "Frame A and Frame B directly &mdash; handy when an "
                "automatic seam would cut across line art or a character "
                "instead of flat background.</p>",
            ),
            "Seam Painter",
        )
        self._add_page(
            _make_page(
                "5. Mesh Warp",
                "Locally deform a frame to fix residual misalignment.",
                "<p>For warps a single homography can't capture, drag the "
                "mesh grid to nudge regions of the image into place before "
                "accepting the pair.</p>",
            ),
            "Mesh Warp",
        )
        self._add_page(
            _make_page(
                "6. Render",
                "Preview and export the accepted pairs.",
                "<p>Once you've accepted homographies (and optionally seams) "
                "for your pairs, the Render tab composites and previews the "
                "result. When you're done, <b>✔ Use as Stitch List</b> in "
                "the sidebar sends the ordered sequence to the main Stitch "
                "tab.</p>",
            ),
            "Render",
        )

        self.currentIdChanged.connect(self._on_page_changed)

    def _add_page(self, page: QWizardPage, tab_name: str | None) -> None:
        page_id = self.addPage(page)
        if tab_name is not None:
            self._tab_for_page[page_id] = tab_name

    def _on_page_changed(self, page_id: int) -> None:
        """Switch the live panel's tool tab to match the current tour page."""
        tab_name = self._tab_for_page.get(page_id)
        if tab_name is None or self._panel is None:
            return
        tools = getattr(self._panel, "_tools", None)
        if tools is None:
            return
        for i in range(tools.count()):
            if tools.tabText(i) == tab_name:
                tools.setCurrentIndex(i)
                break


__all__ = [
    "HybridStitchOnboardingWizard",
    "hybrid_stitch_onboarding_seen",
    "mark_hybrid_stitch_onboarding_seen",
]
