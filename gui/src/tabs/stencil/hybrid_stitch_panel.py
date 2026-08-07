import os

import cv2
import numpy as np
from gui.src.constants import (
    DARK_GROUP_STYLE,
    STITCH_THUMB_H,
    STITCH_THUMB_W,
)
from gui.src.styles import apply_shadow_effect
from PySide6.QtCore import (
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QIcon,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..widgets import ColorCorrectionWidget, MeshWarpWidget, SeamPainterWidget
from .control_point_editor import ControlPointEditor, _apply_color_correction, _load_thumb
from .onboarding_wizard import (
    HybridStitchOnboardingWizard,
    hybrid_stitch_onboarding_seen,
    mark_hybrid_stitch_onboarding_seen,
)
from .render_panel import RenderPanel
from .sample_sequences import list_sample_sequences


class _FrameListItem(QListWidgetItem):
    def __init__(self, path: str):
        super().__init__(os.path.basename(path))
        self.setData(Qt.ItemDataRole.UserRole, path)
        self.setToolTip(path)
        pm = _load_thumb(path, STITCH_THUMB_W, STITCH_THUMB_H)

        self.setIcon(QIcon(pm))
        self.setSizeHint(QSize(STITCH_THUMB_W + 4, STITCH_THUMB_H + 8))


class RealHybridStitchPanel(QWidget):
    """
    Human-in-the-loop stitching panel.

    Left sidebar  : frame sequence list with thumbnail icons.
    Right area    : tabbed tool panels (Control Points, Color Correct,
                    Seam Painter, Mesh Warp, Render).
    """

    sequence_accepted = Signal(list)  # List[str] paths in order

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sequence: list[str] = []
        self._homographies: dict[tuple[int, int], np.ndarray] = {}
        self._seam_masks: dict[tuple[int, int], np.ndarray] = {}
        self._corrections: dict[str, dict] = {}
        self._current_pair: tuple[int, int] = (0, 1)
        self._onboarding_wizard: HybridStitchOnboardingWizard | None = None
        self._onboarding_shown_this_session = False
        self._build_ui()
        # First-run only, gated by AppSettings (Phase 6.1). Deliberately NOT
        # a bare QTimer.singleShot(0, ...) in __init__: that only defers to
        # the next event-loop iteration, not to actual visibility -- this
        # panel is built eagerly at StitchTab construction regardless of
        # which tab is active, so a bare singleShot(0, ...) could pop the
        # tour up while the user is looking at a completely different tab.
        # showEvent() is the correct Qt hook for "when this widget actually
        # becomes visible"; self._onboarding_shown_this_session guards
        # against re-firing every time the user switches back to this tab.

    def showEvent(self, event):
        super().showEvent(event)
        if not self._onboarding_shown_this_session:
            self._onboarding_shown_this_session = True
            QTimer.singleShot(0, self._maybe_show_first_run_onboarding)

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # ── Left sidebar: sequence list ───────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(270)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(4)

        # ── Toolbar: title + re-invocable help/tour button ─────────────
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Hybrid Stitch")
        title.setStyleSheet("font-weight:bold; color:#ddd;")
        toolbar.addWidget(title)
        toolbar.addStretch()
        self._sample_btn = QToolButton()
        self._sample_btn.setText("Try a Sample")
        self._sample_btn.setToolTip(
            "Load a bundled synthetic sample sequence to explore every tool tab."
        )
        self._sample_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._sample_btn.setStyleSheet(
            "QToolButton { border-radius:4px; background:#3a3d42; color:#ccc; "
            "padding:3px 8px; } QToolButton:hover { background:#1976D2; color:white; }"
        )
        self._sample_menu = QMenu(self._sample_btn)
        self._populate_sample_menu()
        self._sample_btn.setMenu(self._sample_menu)
        toolbar.addWidget(self._sample_btn)
        self._help_btn = QToolButton()
        self._help_btn.setText("?")
        self._help_btn.setToolTip("Show the guided tour")
        self._help_btn.setFixedSize(24, 24)
        self._help_btn.setStyleSheet(
            "QToolButton { border-radius:12px; background:#3a3d42; color:#ccc; "
            "font-weight:bold; } QToolButton:hover { background:#1976D2; color:white; }"
        )
        self._help_btn.clicked.connect(self._show_onboarding_wizard)
        toolbar.addWidget(self._help_btn)
        sl.addLayout(toolbar)

        seq_group = QGroupBox("Sequence")
        seq_group.setStyleSheet(DARK_GROUP_STYLE)
        seq_l = QVBoxLayout(seq_group)

        self._seq_list = QListWidget()
        self._seq_list.setMinimumHeight(160)
        self._seq_list.setIconSize(QSize(STITCH_THUMB_W, STITCH_THUMB_H))
        self._seq_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._seq_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._seq_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._seq_list.setSpacing(2)
        self._seq_list.model().rowsMoved.connect(self._on_seq_reordered)
        self._seq_list.currentRowChanged.connect(self._on_seq_selection_changed)
        self._seq_list.setStyleSheet(
            "QListWidget { background:#1e1f22; } "
            "QListWidget::item:selected { background:#1976D2; }"
        )
        seq_l.addWidget(self._seq_list)

        for label, slot in [
            ("Add Frames…", self._add_frames),
            ("Remove Selected", self._remove_frame),
            ("Move Up ↑", self._move_up),
            ("Move Down ↓", self._move_down),
            ("Clear All", self._clear_sequence),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(36)
            btn.clicked.connect(slot)
            seq_l.addWidget(btn)

        sl.addWidget(seq_group)

        # Pair selector
        pair_group = QGroupBox("Working Pair")
        pair_group.setStyleSheet(DARK_GROUP_STYLE)
        pair_l = QFormLayout(pair_group)
        self._pair_a_combo = QComboBox()
        self._pair_b_combo = QComboBox()
        self._pair_a_combo.currentIndexChanged.connect(self._on_pair_changed)
        self._pair_b_combo.currentIndexChanged.connect(self._on_pair_changed)
        pair_l.addRow("Frame A:", self._pair_a_combo)
        pair_l.addRow("Frame B:", self._pair_b_combo)

        btn_load_pair = QPushButton("Load Pair →")
        btn_load_pair.setFixedHeight(36)
        btn_load_pair.setStyleSheet(
            "background:#1565C0; color:white; font-weight:bold; padding:4px;"
        )
        btn_load_pair.clicked.connect(self._load_current_pair)
        apply_shadow_effect(btn_load_pair, radius=4, y_offset=2)
        pair_l.addRow("", btn_load_pair)

        btn_accept_h = QPushButton("✔ Accept H")
        btn_accept_h.setFixedHeight(36)
        btn_accept_h.setToolTip("Accept the current homography and seam for this pair.")
        btn_accept_h.setStyleSheet(
            "background:#2E7D32; color:white; font-weight:bold; padding:4px;"
        )
        btn_accept_h.clicked.connect(self._accept_pair)
        apply_shadow_effect(btn_accept_h, radius=4, y_offset=2)
        pair_l.addRow("", btn_accept_h)

        sl.addWidget(pair_group)

        btn_use = QPushButton("✔ Use as Stitch List")
        btn_use.setFixedHeight(36)
        btn_use.setStyleSheet(
            "background:#388E3C; color:white; font-weight:bold; padding:5px;"
        )
        btn_use.clicked.connect(self._emit_sequence)
        apply_shadow_effect(btn_use, radius=6, y_offset=2)
        sl.addWidget(btn_use)

        sl.addStretch()
        root.addWidget(sidebar)

        # ── Right: tool tabs ──────────────────────────────────────────
        self._tools = QTabWidget()
        self._tools.setStyleSheet(
            "QTabWidget::pane { background:#2c2f33; } "
            "QTabBar::tab { background:#3a3d42; color:#ccc; padding:5px 12px; } "
            "QTabBar::tab:selected { background:#1976D2; color:white; }"
        )

        self._cp_editor = ControlPointEditor()
        self._cp_editor.homography_solved.connect(self._on_h_solved)
        self._tools.addTab(self._cp_editor, "Control Points")

        self._cc_widget = ColorCorrectionWidget()
        self._cc_widget.corrections_changed.connect(self._on_cc_changed)
        self._tools.addTab(self._cc_widget, "Color Correct")

        self._seam_widget = SeamPainterWidget()
        self._tools.addTab(self._seam_widget, "Seam Painter")

        self._mesh_widget = MeshWarpWidget()
        self._mesh_widget.warp_applied.connect(self._on_warp_applied)
        self._tools.addTab(self._mesh_widget, "Mesh Warp")

        self._render_panel = RenderPanel()
        self._tools.addTab(self._render_panel, "Render")

        self._tools.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self._tools, 1)

    # ── Sequence management ──────────────────────────────────────────

    def load_paths(self, paths: list[str]):
        """Load an existing path list into the sequence."""
        self._sequence = list(paths)
        self._refresh_list()

    def _add_frames(self):
        start = os.path.dirname(self._sequence[-1]) if self._sequence else ""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Frames",
            start,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        for p in files:
            if p and p not in self._sequence:
                self._sequence.append(p)
        self._refresh_list()

    def _remove_frame(self):
        row = self._seq_list.currentRow()
        if 0 <= row < len(self._sequence):
            self._sequence.pop(row)
            self._seq_list.takeItem(row)
            self._refresh_combos()

    def _move_up(self):
        row = self._seq_list.currentRow()
        if row > 0:
            self._sequence[row], self._sequence[row - 1] = (
                self._sequence[row - 1],
                self._sequence[row],
            )
            self._refresh_list()
            self._seq_list.setCurrentRow(row - 1)

    def _move_down(self):
        row = self._seq_list.currentRow()
        if row < len(self._sequence) - 1:
            self._sequence[row], self._sequence[row + 1] = (
                self._sequence[row + 1],
                self._sequence[row],
            )
            self._refresh_list()
            self._seq_list.setCurrentRow(row + 1)

    def _clear_sequence(self):
        self._sequence.clear()
        self._seq_list.clear()
        self._pair_a_combo.clear()
        self._pair_b_combo.clear()

    def _refresh_list(self):
        self._seq_list.clear()
        for p in self._sequence:
            self._seq_list.addItem(_FrameListItem(p))
        self._refresh_combos()

    def _refresh_combos(self):
        names = [os.path.basename(p) for p in self._sequence]
        self._pair_a_combo.blockSignals(True)
        self._pair_b_combo.blockSignals(True)
        ca, cb = self._pair_a_combo.currentIndex(), self._pair_b_combo.currentIndex()
        self._pair_a_combo.clear()
        self._pair_b_combo.clear()
        self._pair_a_combo.addItems(names)
        self._pair_b_combo.addItems(names)
        n = len(names)
        if n >= 2:
            ia = max(0, min(ca, n - 1))
            ib = max(1, min(cb, n - 1))
            self._pair_a_combo.setCurrentIndex(ia)
            self._pair_b_combo.setCurrentIndex(ib)
            self._current_pair = (ia, ib)
        self._pair_a_combo.blockSignals(False)
        self._pair_b_combo.blockSignals(False)

    def _on_seq_reordered(self, *_args):
        self._sequence = [
            self._seq_list.item(r).data(Qt.ItemDataRole.UserRole)
            for r in range(self._seq_list.count())
        ]
        self._refresh_combos()

    def _on_seq_selection_changed(self, row: int):
        if 0 <= row < len(self._sequence):
            self._pair_a_combo.setCurrentIndex(row)
            self._pair_b_combo.setCurrentIndex(min(row + 1, len(self._sequence) - 1))

    def _on_pair_changed(self, _):
        ia = self._pair_a_combo.currentIndex()
        ib = self._pair_b_combo.currentIndex()
        self._current_pair = (ia, ib)

    def _load_current_pair(self):
        ia = self._pair_a_combo.currentIndex()
        ib = self._pair_b_combo.currentIndex()
        self._current_pair = (ia, ib)
        if ia < 0 or ib < 0 or ia >= len(self._sequence) or ib >= len(self._sequence):
            QMessageBox.warning(self, "Hybrid Stitch", "Select valid frames A and B.")
            return
        if ia == ib:
            QMessageBox.warning(
                self, "Hybrid Stitch", "Frame A and B must be different."
            )
            return
        pa, pb = self._sequence[ia], self._sequence[ib]
        cc_a = self._corrections.get(pa, {})
        cc_b = self._corrections.get(pb, {})
        self._cp_editor.load_pair(pa, pb, cc_a, cc_b)
        self._cc_widget.load_frame(pa, cc_a)
        self._cc_widget.set_adjacent_path(pb)
        key = (ia, ib)
        if key in self._homographies:
            self._refresh_seam_painter(pa, pb, cc_a, cc_b, self._homographies[key])

    def _accept_pair(self):
        H = self._cp_editor.get_homography()
        if H is None:
            QMessageBox.warning(
                self,
                "Hybrid Stitch",
                "No homography solved yet — use Control Points tab.",
            )
            return
        ia, ib = self._current_pair
        key = (ia, ib)
        self._homographies[key] = H
        seam = self._seam_widget.get_seam_mask()
        if seam is not None:
            self._seam_masks[key] = seam
        QMessageBox.information(
            self,
            "Pair Accepted",
            f"H for pair ({ia}→{ib}) saved.  "
            f"{len(self._homographies)} pair(s) in pipeline.",
        )
        self._update_render_panel()

    def _refresh_seam_painter(
        self, pa: str, pb: str, cc_a: dict, cc_b: dict, H: np.ndarray
    ):
        bgr_a = cv2.imread(pa)
        bgr_b = cv2.imread(pb)
        if bgr_a is None or bgr_b is None:
            return
        if cc_a:
            bgr_a = _apply_color_correction(bgr_a, cc_a)
        if cc_b:
            bgr_b = _apply_color_correction(bgr_b, cc_b)
        h, w = bgr_a.shape[:2]
        bgr_b_w = cv2.warpPerspective(bgr_b, H, (w, h))
        self._seam_widget.load_aligned_pair(bgr_a, bgr_b_w)

    def _on_h_solved(self, H: np.ndarray, err: float):
        ia, ib = self._current_pair
        if ia < len(self._sequence) and ib < len(self._sequence):
            pa, pb = self._sequence[ia], self._sequence[ib]
            cc_a = self._corrections.get(pa, {})
            cc_b = self._corrections.get(pb, {})
            self._refresh_seam_painter(pa, pb, cc_a, cc_b, H)

    def _on_cc_changed(self, cc: dict):
        ia = self._pair_a_combo.currentIndex()
        if 0 <= ia < len(self._sequence):
            self._corrections[self._sequence[ia]] = cc

    def _on_warp_applied(self, bgr: np.ndarray):
        pass

    def _on_tab_changed(self, idx: int):
        tab_name = self._tools.tabText(idx)
        if tab_name == "Control Points":
            QTimer.singleShot(50, self._cp_editor.fit_views)
        elif tab_name == "Mesh Warp":
            ia = self._pair_a_combo.currentIndex()
            if 0 <= ia < len(self._sequence):
                bgr = cv2.imread(self._sequence[ia])
                if bgr is not None:
                    cc = self._corrections.get(self._sequence[ia], {})
                    if cc:
                        bgr = _apply_color_correction(bgr, cc)
                    self._mesh_widget.load_image(bgr)
        elif tab_name == "Render":
            self._update_render_panel()

    def _update_render_panel(self):
        self._render_panel.set_pipeline(
            self._sequence,
            self._homographies,
            self._seam_masks,
            self._corrections,
        )

    def _emit_sequence(self):
        if not self._sequence:
            QMessageBox.information(self, "Hybrid Stitch", "Sequence is empty.")
            return
        self.sequence_accepted.emit(list(self._sequence))

    # ── Bundled sample sequences (roadmap Phase 6.3, issue #17) ───────

    def _populate_sample_menu(self) -> None:
        """(Re-)fill the "Try a Sample" menu from the bundled, synthetic,
        procedurally-generated sample sequences (see
        gui/scripts/generate_sample_sequences.py). Safe to call again if
        samples aren't present -- the menu just gets a disabled placeholder
        rather than raising."""
        self._sample_menu.clear()
        samples = list_sample_sequences()
        if not samples:
            action = self._sample_menu.addAction("No bundled samples found")
            action.setEnabled(False)
            return
        for label, paths in samples.items():
            action = self._sample_menu.addAction(label)
            action.triggered.connect(
                lambda _checked=False, p=paths: self._load_sample_sequence(p)
            )

    def _load_sample_sequence(self, paths: list[str]) -> None:
        """Load a bundled sample sequence into the sidebar, exactly like a
        user's own frames -- every tool tab works on it immediately."""
        self.load_paths(list(paths))

    # ── Onboarding wizard (roadmap Phase 6.1) ──────────────────────────

    def _maybe_show_first_run_onboarding(self):
        """Auto-open the guided tour once, the first time this panel runs."""
        if not hybrid_stitch_onboarding_seen():
            self._show_onboarding_wizard()

    def _show_onboarding_wizard(self):
        """(Re-)open the guided tour. Non-modal: never blocks normal use."""
        if self._onboarding_wizard is not None:
            # Disconnect first: close() below triggers finished(Rejected),
            # and without disconnecting, that would route through
            # _on_onboarding_finished and mark the tour "seen" as a side
            # effect of being *superseded* by a new instance, not of the
            # user actually dismissing it. Also deleteLater() rather than
            # just close(), so the superseded instance doesn't leak as a
            # hidden QWizard for the rest of the session.
            self._onboarding_wizard.finished.disconnect(self._on_onboarding_finished)
            self._onboarding_wizard.close()
            self._onboarding_wizard.deleteLater()
        wizard = HybridStitchOnboardingWizard(self, self)
        wizard.setModal(False)
        wizard.finished.connect(self._on_onboarding_finished)
        self._onboarding_wizard = wizard
        wizard.show()

    def _on_onboarding_finished(self, _result: int):
        # Dismissed at any step (Finish, Cancel, or the window's close box)
        # all land here -- mark it seen so it doesn't auto-open again.
        mark_hybrid_stitch_onboarding_seen()
        self._onboarding_wizard = None
