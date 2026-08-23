"""InspectorWindow: the top-level PySide6 window for per-test evaluation.

Assembles the queue sidebar, the N-way panel grid, the tool tabs
(visualisations / comparison / diagnostics / metrics / artifacts), the scoring
form, and the annotation flow, over an ``EvaluationSession`` that owns queue and
persistence semantics.

``RatingDashboard`` remains as an alias so ``bench_eval_dispatch.py`` and any
existing smoke test keep constructing "the window" by its old name.

Keyboard-first by design — §0.1 budgets ~28 s per test, which a mouse-only form
can't reach. The key map lives in ``shortcuts.py`` and the annotation flow in
``annotation_flow.py``, both split out to keep this file within the repo's
500-LoC budget (§5.17 / issues #121-#122); the user-facing crib sheet is
``KEY_HINTS`` in ``constants/user_interface.py``, rendered into the window footer.
"""

from __future__ import annotations

import traceback as _traceback

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..constants.schema import (
    COMPARATOR_TITLES,
    DEFECTS,
)
from ..constants.user_interface import (
    COL_TEXT_DIM,
    DISPLAY_PIXEL,
    DISPLAY_RAW,
    KEY_HINTS,
    MODE_POINT,
)
from ..other import discovery
from ..other.schema import RatingEntry
from ..other.session import EvaluationSession
from ..other.settings import load_settings
from . import shortcuts
from .annotation_flow import AnnotationFlowMixin
from .annotations import AnnotationListWidget, EdgeBuilder, EdgeOverlay
from .artifacts_tab import ArtifactsTab
from .coherence_v2_tab import CoherenceV2Tab
from .compare_tab import ComparisonTab
from .diagnostics_tab import DiagnosticsTab
from .metrics_panel import MetricsPanel
from .panel_grid import PanelGrid
from .queue_panel import QueuePanel
from .scoring_panel import ScoringPanel
from .settings_flow import SettingsFlowMixin
from .theme import apply_theme, heading, subtle
from .toolbar import InspectorToolbar
from .viz_tab import VisualizationTab

# ---------------------------------------------------------------------------
# Debug instrumentation — set to False to silence all [DBG-INSPECTOR] output.
# Tracks resize events, _fit_all calls (with call-stack origin), splitter
# geometry, and individual panel viewport sizes to diagnose the
# "UI expands beyond boundaries after maximize" bug (issue #153).
# ---------------------------------------------------------------------------
_DBG_INSPECTOR: bool = False


def _dbg(*parts) -> None:  # noqa: D103 — internal debug helper
    if _DBG_INSPECTOR:
        print("[DBG-INSPECTOR]", *parts, flush=True)


class _ElidingLabel(QLabel):
    """A QLabel that elides its text instead of forcing a wide minimum size.

    The plain ``QLabel`` used for the footer key-hint crib sheet had
    ``wordWrap`` off and no elision, so its ``minimumSizeHint`` equalled the
    full unwrapped text width (~1680px for ``KEY_HINTS``) — that propagated
    up through the footer row into ``QMainWindow``'s minimum size, forcing
    the window wider than many screens regardless of the panel-grid/scoring
    sidebar fixes for issue #153. The full text is still available via the
    tooltip.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full_text = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setToolTip(text)
        self._update_elided_text()

    def setText(self, text: str) -> None:  # noqa: D102 - Qt override
        self._full_text = text
        self.setToolTip(text)
        self._update_elided_text()

    def resizeEvent(self, event) -> None:  # noqa: D102 - Qt override
        super().resizeEvent(event)
        self._update_elided_text()

    def _update_elided_text(self) -> None:
        metrics = QFontMetrics(self.font())
        elided = metrics.elidedText(self._full_text, Qt.TextElideMode.ElideRight, self.width())
        super().setText(elided)


class _GridHost(QWidget):
    """Holds the panel grid with the cross-panel edge overlay stacked on top,
    keeping the overlay's geometry synced on resize."""

    def __init__(self, grid: PanelGrid, overlay: EdgeOverlay, parent=None):
        super().__init__(parent)
        self._overlay = overlay
        overlay.setParent(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(grid)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        overlay.sync_geometry()
        _dbg(
            f"_GridHost.__init__: sizePolicy={self.sizePolicy().horizontalPolicy().name}/"
            f"{self.sizePolicy().verticalPolicy().name}"
        )

    def resizeEvent(self, event) -> None:  # noqa: D102 - Qt override
        super().resizeEvent(event)
        _dbg(
            f"_GridHost.resizeEvent: old={event.oldSize().width()}x{event.oldSize().height()}"
            f" new={self.width()}x{self.height()}"
        )
        self._overlay.sync_geometry()


class InspectorWindow(AnnotationFlowMixin, SettingsFlowMixin, QMainWindow):
    def __init__(
        self,
        base_dir: str,
        out_path: str,
        redo: bool,
        repo_root: str,
        default_display_mode: str = DISPLAY_RAW,
        results_path: str | None = None,
        theme: str | None = None,
    ):
        super().__init__()
        self.setWindowTitle("Benchmark Evaluation Inspector")
        self.resize(1760, 1040)
        # A CLI --theme wins for this run; otherwise fall back to whatever the
        # Settings dialog last persisted (see settings.py), same override
        # precedence bench_eval_dispatch.py already uses for --out.
        self._settings = load_settings()
        if theme:
            self._settings.theme = theme
        apply_theme(self, self._settings.theme)

        self.base_dir = base_dir
        self.repo_root = repo_root
        self.results_path = results_path
        self._assets = None
        self._edge_builder = EdgeBuilder()

        names = discovery.discover_datasets(base_dir)
        self.session = EvaluationSession(names, out_path, redo=redo)
        self.total_datasets = len(names)

        self._build_ui(default_display_mode)
        shortcuts.install(self)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._fit_all)
        self._fitting: bool = False  # re-entrancy guard for _fit_all
        self.queue_panel.set_session(self.session)
        _dbg("InspectorWindow.__init__: construction complete")
        _dbg(
            f"  window size: {self.width()}x{self.height()}"
            f"  | central widget sizePolicy H={self.centralWidget().sizePolicy().horizontalPolicy().name}"
            f" V={self.centralWidget().sizePolicy().verticalPolicy().name}"
        )
        if hasattr(self, "body_splitter"):
            _dbg(f"  body_splitter sizes: {self.body_splitter.sizes()}")
        if self.session.current:
            self._load_current()
        else:
            self.status_label.setText(f"No datasets found under {base_dir}/output/.")

    # -- compatibility -------------------------------------------------------

    @property
    def out_path(self) -> str:
        return self.session.out_path

    @property
    def evaluations(self) -> dict[str, RatingEntry]:
        return self.session.evaluations

    @property
    def todo(self) -> list[str]:
        """Datasets still lacking a real judgment — the queue the CLI reports."""
        return [n for n in self.session.order if not self.session.is_rated(n)]

    # -- construction --------------------------------------------------------

    def _build_ui(self, default_display_mode: str) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 6)
        outer.setSpacing(6)

        header = QHBoxLayout()
        self.title_label = heading("—")
        self.status_label = _ElidingLabel("")
        self.status_label.setProperty("role", "subtle")
        self.settings_btn = QPushButton("⚙ Settings")
        self.settings_btn.setToolTip("Default save directory, dark/light theme")
        self.settings_btn.clicked.connect(self._open_settings)
        self.load_eval_btn = QPushButton("Load Evaluation…")
        self.load_eval_btn.setToolTip("Resume an existing benchmark evaluation JSON")
        self.load_eval_btn.clicked.connect(self._load_evaluation_file)
        header.addWidget(self.title_label)
        header.addWidget(self.status_label, stretch=1)
        header.addWidget(self.load_eval_btn)
        header.addWidget(self.settings_btn)
        outer.addLayout(header)

        # Two top-level pages: "Rate" (images + scoring, the per-test workflow)
        # and "Analyze" (the tool tabs, now a full page instead of sharing
        # vertical space with the panel grid below it) — freeing up room for
        # both was the point of splitting them apart.
        self.top_tabs = QTabWidget()
        self.top_tabs.addTab(self._build_rate_tab(default_display_mode), "Rate")
        self.top_tabs.addTab(self._build_analyze_tab(), "Analyze")
        outer.addWidget(self.top_tabs, stretch=1)

        self.pixel_label = _ElidingLabel("Pixel: —")
        self.pixel_label.setProperty("role", "mono")
        footer = QHBoxLayout()
        footer.addWidget(self.pixel_label, stretch=1)
        hints = _ElidingLabel("   ".join(f"{k}: {v}" for k, v in KEY_HINTS))
        hints.setStyleSheet(f"color: {COL_TEXT_DIM}; font-size: 11px;")
        footer.addWidget(hints, stretch=3)
        outer.addLayout(footer)

    def _build_rate_tab(self, default_display_mode: str) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(6)

        self.toolbar = InspectorToolbar()
        self.toolbar.modeChanged.connect(self._on_mode_changed)
        self.toolbar.displayModeChanged.connect(self._on_display_mode_changed)
        self.toolbar.layoutChanged.connect(self._on_layout_changed)
        self.toolbar.visibilityChanged.connect(self._on_visibility_changed)
        self.toolbar.lockToggled.connect(lambda locked: self.grid.set_locked(locked))
        self.toolbar.fitRequested.connect(self._fit_all)
        self.toolbar.zoomRequested.connect(lambda step: self.grid.zoom_focused(step))
        self.toolbar.saveRequested.connect(self._save_now)
        page_layout.addWidget(self.toolbar)

        self.grid = PanelGrid()
        self.grid.bboxDrawn.connect(self._on_bbox_drawn)
        self.grid.pointPicked.connect(self._on_point_picked)
        self.grid.regionPicked.connect(self._on_region_picked)
        self.grid.pixelHovered.connect(self._on_pixel_hovered)
        self.grid.pixelPinned.connect(self._on_pixel_pinned)
        self.grid.focusChanged.connect(self._on_focus_changed)
        self.overlay = EdgeOverlay()
        self.overlay.register_panels(self.grid.panels)
        grid_host = _GridHost(self.grid, self.overlay)
        self.grid.set_display_mode(default_display_mode)
        self.toolbar.set_display_pixel(default_display_mode == DISPLAY_PIXEL)

        self.queue_panel = QueuePanel()
        self.queue_panel.datasetSelected.connect(self._on_dataset_selected)
        side_panel = self._build_side_panel()

        self.body_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.body_splitter.addWidget(self.queue_panel)
        self.body_splitter.addWidget(grid_host)
        self.body_splitter.addWidget(side_panel)
        self.body_splitter.setStretchFactor(0, 0)
        self.body_splitter.setStretchFactor(1, 4)
        self.body_splitter.setStretchFactor(2, 0)
        self.body_splitter.setSizes([200, 800, 320])
        # Qt's QSplitter remembers a hidden child's proportion and restores it
        # on show() — confirmed by testing — so plain setVisible() toggling is
        # enough, no manual size bookkeeping needed on either side.
        self.toolbar.queuePanelToggled.connect(self.queue_panel.setVisible)
        self.toolbar.sidePanelToggled.connect(side_panel.setVisible)
        page_layout.addWidget(self.body_splitter, stretch=1)
        return page

    def _build_analyze_tab(self) -> QWidget:
        self.tabs = QTabWidget()
        self.viz_tab = VisualizationTab()
        self.compare_tab = ComparisonTab()
        self.coherence_v2_tab = CoherenceV2Tab(self.repo_root)
        self.diagnostics_tab = DiagnosticsTab(self.repo_root)
        self.metrics_panel = MetricsPanel()
        self.artifacts_tab = ArtifactsTab()
        self.tabs.addTab(self.metrics_panel, "Metrics")
        self.tabs.addTab(self.coherence_v2_tab, "Coherence V2")
        self.tabs.addTab(self.diagnostics_tab, "Diagnostics")
        self.tabs.addTab(self.compare_tab, "Compare")
        self.tabs.addTab(self.viz_tab, "Visualise")
        self.tabs.addTab(self.artifacts_tab, "Artifacts")
        return self.tabs

    def _build_side_panel(self) -> QWidget:
        host = QWidget()
        host.setMinimumWidth(300)
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.scoring_panel = ScoringPanel()
        self.scoring_panel.changed.connect(self._on_scoring_changed)
        layout.addWidget(self.scoring_panel, stretch=3)

        layout.addWidget(subtle("Regions & links"))
        self.annotation_list = AnnotationListWidget()
        self.annotation_list.set_remove_callback(self._on_remove_annotation)
        layout.addWidget(self.annotation_list, stretch=1)

        nav = QHBoxLayout()
        nav.setSpacing(4)
        self.back_btn = QPushButton("‹ Back")
        self.back_btn.clicked.connect(self._go_back)
        self.skip_btn = QPushButton("Skip")
        self.skip_btn.setToolTip("Defer this test — it stays in the queue and Back still works")
        self.skip_btn.clicked.connect(self._go_skip)
        self.next_btn = QPushButton("Save & Next ›")
        self.next_btn.setProperty("role", "primary")
        self.next_btn.clicked.connect(self._go_next)
        for btn in (self.back_btn, self.skip_btn, self.next_btn):
            nav.addWidget(btn)
        layout.addLayout(nav)
        return host

    # -- dataset loading -----------------------------------------------------

    def _load_current(self) -> None:
        name = self.session.current
        if name is None:
            return
        self._edge_builder.reset()
        self._assets = discovery.load_test_assets(
            self.base_dir, name, self.repo_root, self.results_path
        )
        images = discovery.load_images(self._assets)
        entry = self.session.entry(name)

        self.grid.set_images(images)
        self.toolbar.set_comparators(self.grid.available(), self.grid.visible())
        self.grid.restore_bboxes(entry.bboxes)
        self.overlay.set_edges(entry.edges)
        self.overlay.set_pending([])  # an in-progress link never carries across tests

        self.scoring_panel.set_comparators([k for k in self.grid.available() if k in self.scorable()])
        self.scoring_panel.load_entry(entry)
        self.annotation_list.refresh(entry.bboxes, entry.edges)

        run_label = self._run_label()
        self.metrics_panel.set_metrics(self._assets.metrics, run_label)
        self.diagnostics_tab.set_dataset(name)
        for tab in (self.viz_tab, self.compare_tab, self.diagnostics_tab, self.coherence_v2_tab):
            tab.set_context(images, self._assets.metrics, name)
        self.artifacts_tab.set_assets(self._assets)

        self._update_header()
        self.back_btn.setEnabled(self.session.can_go_back())
        # Panels have no final geometry until the event loop runs, so fitting
        # now would use stale viewport sizes.  Route through _resize_timer (not
        # a bare singleShot) so that a concurrent window-manager resize debounce
        # is never shortened — see issue #153 / TROUBLESHOOTING.md §Geometry Overflow.
        _dbg("_load_current: scheduling _fit_all via _schedule_fit(0)")
        self._schedule_fit(0)

    def scorable(self) -> list[str]:
        from ..constants.schema import SCORABLE_KEYS

        return list(SCORABLE_KEYS)

    def _run_label(self) -> str:
        metrics = discovery.load_metrics(self.repo_root, self.results_path)
        metadata = metrics.get("__metadata__") or {}
        timestamp = metadata.get("timestamp", "")
        count = metadata.get("total_datasets")
        if not timestamp:
            return ""
        return f"{timestamp[:16].replace('T', ' ')}, {count} datasets"

    def _update_header(self) -> None:
        name = self.session.current or "—"
        progress = self.session.progress()
        self.title_label.setText(name)
        missing = self.scoring_panel.missing_required()
        state = "complete" if not missing else f"needs {', '.join(missing)}"
        dirty = self.session.is_dirty
        self.status_label.setText(
            f"#{progress.position}/{progress.total} · {progress.rated} rated · "
            f"{progress.skipped} skipped · this test: {state}"
            + (" · unsaved changes" if dirty else "")
        )
        self.toolbar.set_dirty(dirty)
        self.queue_panel.refresh()

    # -- toolbar / view ------------------------------------------------------

    def _on_mode_changed(self, mode: str) -> None:
        self.grid.set_mode(mode)
        if mode != MODE_POINT and self._edge_builder.count() > 0:
            self._edge_builder.reset()
            self.overlay.set_pending([])
            self.status_label.setText("Link cancelled (left Link mode).")

    def _on_display_mode_changed(self, mode: str) -> None:
        self.grid.set_display_mode(mode)

    def _on_layout_changed(self, mode: str) -> None:
        self.grid.set_layout_mode(mode)
        _dbg(f"_on_layout_changed({mode!r}): scheduling _fit_all via _schedule_fit(0)")
        self._schedule_fit(0)

    def _on_visibility_changed(self, keys: list) -> None:
        self.grid.set_visible(keys)
        _dbg(f"_on_visibility_changed({keys}): scheduling _fit_all via _schedule_fit(0)")
        self._schedule_fit(0)

    def _schedule_fit(self, delay_ms: int = 0) -> None:
        """Route all _fit_all scheduling through ``_resize_timer``.

        Using a bare ``QTimer.singleShot(0, self._fit_all)`` bypasses the
        resize debounce and can fire while the window manager is still
        animating a maximize/move, causing ``fit_to_view`` to measure
        transitional viewport sizes and propagate an inflated width request
        up through the splitter (issue #153).

        Rule: if a *longer* debounce is already counting down (e.g. the
        150 ms from ``resizeEvent``), never shorten it — just let it fire.
        If the timer is idle or ``delay_ms`` is longer than what's left,
        restart it at ``delay_ms``.
        """
        remaining = self._resize_timer.remainingTime() if self._resize_timer.isActive() else -1
        if remaining >= 0 and remaining > delay_ms:
            # A longer debounce is already active — don't shorten it.
            _dbg(
                f"_schedule_fit({delay_ms} ms): deferred — resize debounce has {remaining} ms left"
            )
            return
        _dbg(f"_schedule_fit({delay_ms} ms): starting timer")
        self._resize_timer.start(delay_ms)

    def _fit_all(self) -> None:
        # Re-entrancy guard: QGraphicsView.setTransform / centerOn can trigger
        # internal QAbstractScrollArea geometry updates that propagate back into
        # the Qt event loop and may re-invoke _fit_all while we are still inside
        # grid.fit_all().  The guard prevents that second call from doing anything.
        if self._fitting:
            _dbg("_fit_all: skipped (re-entrant call)")
            return
        if _DBG_INSPECTOR:
            # Capture call origin to identify which timer/path triggered this.
            stack = _traceback.extract_stack()
            caller = stack[-2]  # immediate caller of _fit_all
            _dbg(
                f"_fit_all: called from {caller.filename.split('/')[-1]}:{caller.lineno} ({caller.name})"
            )
            _dbg(
                f"  window geometry: {self.geometry().width()}x{self.geometry().height()}"
                f" | frameGeometry: {self.frameGeometry().width()}x{self.frameGeometry().height()}"
            )
            if hasattr(self, "body_splitter"):
                _dbg(f"  body_splitter sizes before fit: {self.body_splitter.sizes()}")
            # Log visible panel viewport sizes before fitting.
            if hasattr(self, "grid"):
                for key in self.grid.visible():
                    panel = self.grid.panels.get(key)
                    if panel:
                        vp = panel.viewport()
                        sp_h = panel.sizePolicy().horizontalPolicy().name
                        sp_v = panel.sizePolicy().verticalPolicy().name
                        _dbg(
                            f"  panel[{key}]: viewport={vp.width()}x{vp.height()}"
                            f" | panel={panel.width()}x{panel.height()}"
                            f" | fit_scale={panel._fit_scale:.4f}"
                            f" | zoom={panel._zoom:.4f}"
                            f" | sizePolicy={sp_h}/{sp_v}"
                        )
        self._fitting = True
        try:
            self.grid.fit_all()
        finally:
            self._fitting = False
        if _DBG_INSPECTOR:
            if hasattr(self, "body_splitter"):
                _dbg(f"  body_splitter sizes after fit: {self.body_splitter.sizes()}")
            _dbg(
                f"  window geometry after fit: {self.geometry().width()}x{self.geometry().height()}"
            )
        self.overlay.sync_geometry()
        self.overlay.update()

    def _on_focus_changed(self, key: str) -> None:
        if key:
            self.status_label.setText(
                f"Focused: {COMPARATOR_TITLES.get(key, key)} — 0-4 scores its coherence"
            )

    def _on_pixel_hovered(self, key: str, x: int, y: int, bgr) -> None:
        if bgr is None:
            self.pixel_label.setText("Pixel: —")
            return
        b, g, r = bgr
        self.pixel_label.setText(
            f"{COMPARATOR_TITLES.get(key, key)} ({x}, {y})  R={r:3d} G={g:3d} B={b:3d}  #{r:02x}{g:02x}{b:02x}"
        )

    def _on_pixel_pinned(self, key: str, x: int, y: int, bgr) -> None:
        """Pin a probe and report the same pixel in every other panel, which is
        the actual question being asked when probing a comparison."""
        readings = []
        for other in self.grid.visible():
            panel = self.grid.panels[other]
            img = panel.current_image()
            if img is None:
                continue
            h, w = img.shape[:2]
            # Sample the *proportionally* matching pixel, since canvases differ
            # in size; exact pixel coordinates aren't comparable across them.
            src = self.grid.panels[key].current_image()
            if src is None:
                continue
            sh, sw = src.shape[:2]
            ox, oy = min(w - 1, int(x / sw * w)), min(h - 1, int(y / sh * h))
            bb, gg, rr = (int(v) for v in img[oy, ox])
            readings.append(f"{COMPARATOR_TITLES.get(other, other)}({ox},{oy})=({rr},{gg},{bb})")
        self.pixel_label.setText("Pinned: " + "   ".join(readings))

    # -- scoring / persistence ----------------------------------------------

    def _score_focused(self, value: int) -> None:
        key = self.grid.focus_key()
        if key is None:
            self.status_label.setText("No panel focused — press A, S or Tab first.")
            return
        if not self.scoring_panel.score_focused(key, value):
            self.status_label.setText(f"{COMPARATOR_TITLES.get(key, key)} isn't a scored comparator.")

    def _toggle_defect(self, index: int) -> None:
        key = self.scoring_panel.toggle_defect_index(index)
        if key:
            title = dict((k, t) for k, t, _ in DEFECTS).get(key, key)
            self.status_label.setText(f"Toggled defect tag: {title}")

    def _on_scoring_changed(self) -> None:
        self._commit()
        self._update_header()

    def _commit(self) -> None:
        self.session.commit()

    def _save_now(self) -> None:
        self._commit()
        self.session.save()
        self._update_header()
        self.status_label.setText(f"Saved to {self.session.out_path}")

    def _load_evaluation_file(self) -> None:
        if not self._confirm_discard("Load evaluation file"):
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load previous benchmark evaluation",
            self.session.out_path,
            "ASP evaluations (*.json);;All files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return
        self.session.load_evaluation_file(path)
        self.queue_panel.refresh()
        if self.session.current:
            self._load_current()
        else:
            self._update_header()
        self.status_label.setText(f"Resumed evaluation from {self.session.out_path}")

    # -- navigation ----------------------------------------------------------

    def _go_next(self) -> None:
        missing = self.scoring_panel.missing_required()
        if missing:
            QMessageBox.warning(
                self, "Incomplete evaluation",
                "Set both ASP and Simple coherence (0-4) before continuing, or press Skip "
                f"to defer this test.\n\nStill missing: {', '.join(missing)}",
            )
            return
        if self.session.accept() is None:
            self._finish("All datasets rated.")
            return
        self._load_current()

    def _go_skip(self) -> None:
        if self.session.advance(skip=True) is None:
            self._finish("Reached the end of the queue.")
            return
        self._load_current()

    def _go_back(self) -> None:
        if self.session.go_back() is None:
            self.status_label.setText("Nothing to go back to.")
            return
        self._load_current()

    def _on_dataset_selected(self, name: str) -> None:
        self.session.go_to(name)
        self._load_current()

    def open_dataset(self, name: str, record_history: bool = False) -> bool:
        """Jump straight to a dataset by name — the public entry the CLI's
        ``--start-at`` and the FiftyOne plugin's handoff use."""
        if name not in self.session.order:
            return False
        self.session.go_to(name, record_history=record_history)
        self._load_current()
        return True

    def _finish(self, message: str) -> None:
        self.session.save()
        progress = self.session.progress()
        self.status_label.setText(
            f"{message} {progress.rated}/{progress.total} rated. Saved to {self.session.out_path}."
        )
        self.queue_panel.refresh()

    def resizeEvent(self, event) -> None:  # noqa: D102 - Qt override
        super().resizeEvent(event)
        _dbg(
            f"InspectorWindow.resizeEvent: "
            f"old={event.oldSize().width()}x{event.oldSize().height()}"
            f" -> new={event.size().width()}x{event.size().height()}"
            f" | frame={self.frameGeometry().width()}x{self.frameGeometry().height()}"
        )
        if _DBG_INSPECTOR and hasattr(self, "body_splitter"):
            _dbg(f"  body_splitter sizes at resize: {self.body_splitter.sizes()}")
        if _DBG_INSPECTOR:
            # Log the screen geometry to detect if the window has already exceeded it.
            try:
                from PySide6.QtGui import QGuiApplication
                screen = QGuiApplication.primaryScreen()
                if screen:
                    sg = screen.geometry()
                    _dbg(
                        f"  screen geometry: {sg.width()}x{sg.height()}"
                        f" | window exceeds screen width: {self.frameGeometry().width() > sg.width()}"
                    )
            except Exception as _e:  # noqa: BLE001
                _dbg(f"  (could not query screen geometry: {_e})")
        if hasattr(self, "overlay") and self.overlay is not None:
            self.overlay.sync_geometry()
        if hasattr(self, "_resize_timer") and self._resize_timer is not None:
            _dbg("  starting _resize_timer (150 ms debounce -> _fit_all)")
            self._resize_timer.start(150)

    def _confirm_discard(self, title: str) -> bool:
        """Ask before dropping unsaved ratings. True = it is safe to proceed.

        Saving is explicit now, so an unsaved session is the normal state
        rather than an error — this is the only thing standing between a
        rating pass and a closed window.
        """
        self._commit()
        if not self.session.is_dirty:
            return True
        choice = QMessageBox.question(
            self,
            title,
            "You have unsaved ratings. Save them before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if choice == QMessageBox.StandardButton.Save:
            self.session.save()
            return True
        return choice == QMessageBox.StandardButton.Discard

    def closeEvent(self, event) -> None:  # noqa: D102 - Qt override
        if not self._confirm_discard("Close inspector"):
            event.ignore()
            return
        super().closeEvent(event)


# The pre-rebuild name, kept so bench_eval_dispatch.py and its smoke test keep
# working without knowing the window was replaced.
RatingDashboard = InspectorWindow
