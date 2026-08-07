"""Sequence Builder sub-tab: UI construction.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gui.src.styles import apply_shadow_effect
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from ._stitch_tab_protocol import _StitchTabHost

    # Type-checking-only base: gives mypy visibility into attributes set by
    # StitchTab.__init__ / sibling mixins (see _stitch_tab_protocol.py).
    # Zero runtime effect -- at runtime this mixin still only inherits object.
    class _Base(_StitchTabHost, QWidget): ...
else:
    _Base = object


class _SeqPanelBuildMixin(_Base):
    def _build_seq_panel(self) -> QWidget:
        from asp_gui.tabs.stencil import SeqBuilderPanel

        panel = SeqBuilderPanel(self)
        root = QVBoxLayout(panel)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Source ────────────────────────────────────────────────────
        src_group = QGroupBox("Source")
        src_form = QFormLayout(src_group)

        anchor_row = QHBoxLayout()
        self._seq_anchor_edit = QLineEdit()
        self._seq_anchor_edit.setPlaceholderText("Pick the base/anchor image…")
        self._seq_anchor_edit.setReadOnly(True)
        anchor_row.addWidget(self._seq_anchor_edit, 1)
        btn_anchor = QPushButton("Open...")
        btn_anchor.clicked.connect(self._seq_browse_anchor)
        anchor_row.addWidget(btn_anchor)
        src_form.addRow("Anchor image:", anchor_row)

        dir_row = QHBoxLayout()
        self._seq_dir_edit = QLineEdit()
        self._seq_dir_edit.setPlaceholderText("Directory of candidate images…")
        self._seq_dir_edit.setReadOnly(True)
        dir_row.addWidget(self._seq_dir_edit, 1)
        btn_dir = QPushButton("Browse…")
        btn_dir.clicked.connect(self._seq_browse_dir)
        dir_row.addWidget(btn_dir)
        src_form.addRow("Candidates dir:", dir_row)

        btn_from_stitch = QPushButton("Use Stitch Frame List as Candidates")
        btn_from_stitch.setToolTip(
            "Populate the candidate pool from the current Stitch tab frame list."
        )
        btn_from_stitch.clicked.connect(self._seq_load_from_stitch)
        src_form.addRow("", btn_from_stitch)

        root.addWidget(src_group)

        # ── Options + run ─────────────────────────────────────────────
        opt_group = QGroupBox("Options")
        opt_layout = QHBoxLayout(opt_group)

        opt_layout.addWidget(QLabel("Min fitness:"))
        self._seq_min_score_spin = QDoubleSpinBox()
        self._seq_min_score_spin.setRange(0.01, 0.99)
        self._seq_min_score_spin.setValue(0.15)
        self._seq_min_score_spin.setDecimals(2)
        self._seq_min_score_spin.setSingleStep(0.05)
        self._seq_min_score_spin.setToolTip(
            "Minimum stitching fitness to extend the chain.\n"
            "Fitness = ORB inlier ratio × displacement quality "
            "(peaks at ~30% of frame diagonal pan, zero for near-duplicates or non-overlapping).\n"
            "Start at 0.15; raise to filter weaker pairs."
        )
        opt_layout.addWidget(self._seq_min_score_spin)

        opt_layout.addSpacing(12)
        opt_layout.addWidget(QLabel("Min sharpness ratio:"))
        self._seq_blur_spin = QDoubleSpinBox()
        self._seq_blur_spin.setRange(0.0, 1.0)
        self._seq_blur_spin.setValue(0.50)
        self._seq_blur_spin.setDecimals(2)
        self._seq_blur_spin.setSingleStep(0.05)
        self._seq_blur_spin.setToolTip(
            "Reject candidates whose Laplacian sharpness is below this fraction "
            "of the anchor's sharpness.\n"
            "0.5 = must be at least 50% as sharp as the anchor (filters motion-blurred frames).\n"
            "Set to 0.0 to disable the sharpness filter."
        )
        opt_layout.addWidget(self._seq_blur_spin)

        opt_layout.addSpacing(12)
        opt_layout.addWidget(QLabel("Min pan %:"))
        self._seq_min_pan_spin = QDoubleSpinBox()
        self._seq_min_pan_spin.setRange(0.01, 0.50)
        self._seq_min_pan_spin.setValue(0.03)
        self._seq_min_pan_spin.setDecimals(2)
        self._seq_min_pan_spin.setSingleStep(0.01)
        self._seq_min_pan_spin.setToolTip(
            "Minimum camera translation as a fraction of the frame diagonal.\n"
            "Below this → near-duplicate (rejected).\n"
            "3% is usually right for anime panning; raise if duplicates appear."
        )
        opt_layout.addWidget(self._seq_min_pan_spin)

        opt_layout.addSpacing(12)
        opt_layout.addWidget(QLabel("Max pan %:"))
        self._seq_max_pan_spin = QDoubleSpinBox()
        self._seq_max_pan_spin.setRange(0.20, 0.99)
        self._seq_max_pan_spin.setValue(0.85)
        self._seq_max_pan_spin.setDecimals(2)
        self._seq_max_pan_spin.setSingleStep(0.05)
        self._seq_max_pan_spin.setToolTip(
            "Maximum camera translation as a fraction of the frame diagonal.\n"
            "Above this → frames don't overlap enough (rejected).\n"
            "85% is usually safe; lower if stitching fails at large offsets."
        )
        opt_layout.addWidget(self._seq_max_pan_spin)

        opt_layout.addStretch()

        self._seq_run_btn = QPushButton("⚡ Build Sequence")
        self._seq_run_btn.setStyleSheet(
            "background:#1976D2; color:white; font-weight:bold; padding:5px 14px;"
        )
        self._seq_run_btn.clicked.connect(self._seq_run)
        opt_layout.addWidget(self._seq_run_btn)

        root.addWidget(opt_group)

        # ── Progress ──────────────────────────────────────────────────
        prog_row = QHBoxLayout()
        self._seq_progress = QProgressBar()
        self._seq_progress.setRange(0, 100)
        self._seq_progress.setTextVisible(True)
        self._seq_progress.hide()
        prog_row.addWidget(self._seq_progress, 1)
        self._seq_status = QLabel("")
        self._seq_status.setStyleSheet("color:#aaa; font-style:italic;")
        prog_row.addWidget(self._seq_status)
        root.addLayout(prog_row)

        # ── Result chain ──────────────────────────────────────────────
        chain_group = QGroupBox(
            "Built Sequence  (drag to reorder · double-click row to replace image)"
        )
        chain_layout = QVBoxLayout(chain_group)

        self._seq_chain_table = QTableWidget(0, 3)
        self._seq_chain_table.setHorizontalHeaderLabels(["Image", "Score to prev.", ""])
        self._seq_chain_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._seq_chain_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self._seq_chain_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self._seq_chain_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._seq_chain_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._seq_chain_table.setAlternatingRowColors(True)
        self._seq_chain_table.verticalHeader().setVisible(False)
        self._seq_chain_table.verticalHeader().setDefaultSectionSize(52)
        self._seq_chain_table.setIconSize(QSize(48, 48))
        self._seq_chain_table.setDragDropMode(QTableWidget.DragDropMode.InternalMove)
        self._seq_chain_table.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._seq_chain_table.setDragEnabled(True)
        self._seq_chain_table.setAcceptDrops(True)
        self._seq_chain_table.setDropIndicatorShown(True)
        self._seq_chain_table.setMinimumHeight(260)
        self._seq_chain_table.setStyleSheet(
            "QTableWidget { background:#2c2f33; alternate-background-color:#36393f; }"
            "QHeaderView::section { background:#1e1f22; color:#ccc; padding:4px; }"
        )
        self._seq_chain_table.cellDoubleClicked.connect(self._seq_replace_row)
        self._seq_chain_table.model().rowsMoved.connect(self._seq_on_rows_moved)
        chain_layout.addWidget(self._seq_chain_table)

        # Edit buttons
        edit_row = QHBoxLayout()
        btn_add_before = QPushButton("Insert Before")
        btn_add_before.setToolTip("Insert a new image before the selected row.")
        btn_add_before.clicked.connect(lambda: self._seq_insert_image(before=True))
        btn_add_after = QPushButton("Insert After")
        btn_add_after.setToolTip("Insert a new image after the selected row.")
        btn_add_after.clicked.connect(lambda: self._seq_insert_image(before=False))
        btn_remove = QPushButton("Remove")
        btn_remove.setToolTip("Remove the selected row from the chain.")
        btn_remove.clicked.connect(self._seq_remove_row)
        btn_up = QPushButton("Up ↑")
        btn_up.setFixedWidth(80)
        btn_up.clicked.connect(self._seq_move_up)
        btn_down = QPushButton("Down ↓")
        btn_down.setFixedWidth(80)
        btn_down.clicked.connect(self._seq_move_down)
        for b in (btn_add_before, btn_add_after, btn_remove, btn_up, btn_down):
            apply_shadow_effect(b, radius=4, y_offset=2)
            edit_row.addWidget(b)
        edit_row.addStretch()

        btn_accept = QPushButton("✔ Use as Stitch List")
        btn_accept.setStyleSheet(
            "background:#388E3C; color:white; font-weight:bold; padding:5px 14px;"
        )
        btn_accept.setToolTip(
            "Load the current sequence into the Stitch tab frame list, replacing any existing frames."
        )
        btn_accept.clicked.connect(self._seq_accept)
        apply_shadow_effect(btn_accept, radius=6, y_offset=2)
        edit_row.addWidget(btn_accept)

        chain_layout.addLayout(edit_row)
        root.addWidget(chain_group)

        return panel


__all__ = ["_SeqPanelBuildMixin"]
