import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
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
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ....constants import (
    DARK_GROUP_STYLE,
    STITCH_THUMB_H,
    STITCH_THUMB_W,
)
from ....styles import apply_shadow_effect
from ..widgets import ColorCorrectionWidget, MeshWarpWidget, SeamPainterWidget
from .control_point_editor import ControlPointEditor, _apply_color_correction, _load_thumb
from .render_panel import RenderPanel


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
        self._sequence: List[str] = []
        self._homographies: Dict[Tuple[int, int], np.ndarray] = {}
        self._seam_masks: Dict[Tuple[int, int], np.ndarray] = {}
        self._corrections: Dict[str, dict] = {}
        self._current_pair: Tuple[int, int] = (0, 1)
        self._build_ui()

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

    def load_paths(self, paths: List[str]):
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
