"""Stitch sub-tab UI construction.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change. Kept
as a single ~500-line method rather than split further: the widget tree is
built by one continuous sequence of ``splitter.addWidget``/``layout.addWidget``
calls with heavy local-variable interdependency (left/centre/right panels all
feed into the same ``splitter``, and ``root`` collects both the splitter and
the bottom bar at the end) -- decomposing it into several ``_build_*_row()``
helpers would risk subtly reordering widget construction or reparenting for a
line-count-only benefit. See the settings_window.py split (issue #121/#122)
for the precedent of documenting a large build method as a deliberate
exception rather than forcing an unsafe split.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ....helpers.animation import StitchWorker
from ....styles import apply_shadow_effect
from ....windows.settings.splitter_persistence import persist_splitter
from ._match_editor import _MatchScene, _MatchView


class _StitchPanelBuildMixin:
    def _build_stitch_panel(self) -> QWidget:
        from animation.gui.tabs.animation.stencil import StitchPanel

        panel = StitchPanel(self)
        root = QVBoxLayout(panel)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Main splitter (left │ centre │ right) ─────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # ── LEFT: frame list ──────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(190)
        left.setMaximumWidth(240)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)

        frames_group = QGroupBox("Source Frames")
        frames_group_layout = QVBoxLayout(frames_group)

        # Video mode toggle & Frame counter
        video_row = QHBoxLayout()
        video_row.setContentsMargins(0, 0, 0, 0)
        self._cb_video_mode = QCheckBox("From Video Source")
        self._cb_video_mode.setToolTip(
            "Extract frames from a video file instead of loading images manually.\n"
            "Requires: pip install av (PyAV)"
        )
        self._lbl_frame_count = QLabel("Frames: 0")
        self._lbl_frame_count.setStyleSheet(
            "QLabel { font-weight: bold; color: #00bcd4; }"
        )
        self._lbl_frame_count.setToolTip("Total number of loaded source frames.")

        video_row.addWidget(self._cb_video_mode)
        video_row.addStretch()
        video_row.addWidget(self._lbl_frame_count)
        frames_group_layout.addLayout(video_row)

        # Video input panel (shown only in video mode)
        self._video_input_widget = QWidget()
        _vl = QVBoxLayout(self._video_input_widget)
        _vl.setContentsMargins(0, 0, 0, 0)
        _vl.setSpacing(3)
        self._video_path_edit = QLineEdit()
        self._video_path_edit.setPlaceholderText("Video file path…")
        self._video_path_edit.setToolTip("Path to the video file (MP4, MKV, AVI, etc.)")
        _vl.addWidget(self._video_path_edit)
        _vbrow = QHBoxLayout()
        _btn_browse_video = QPushButton("Browse…")
        _btn_browse_video.setToolTip("Choose a video file.")
        _btn_browse_video.clicked.connect(self._browse_video)
        apply_shadow_effect(_btn_browse_video, radius=4, y_offset=2)
        self._video_n_frames_spin = QSpinBox()
        self._video_n_frames_spin.setRange(2, 200)
        self._video_n_frames_spin.setValue(20)
        self._video_n_frames_spin.setToolTip(
            "Number of frames to extract from the video."
        )
        self._video_n_frames_spin.setPrefix("N: ")
        _vbrow.addWidget(_btn_browse_video)
        _vbrow.addWidget(self._video_n_frames_spin)
        _vl.addLayout(_vbrow)
        self._video_input_widget.setVisible(False)
        frames_group_layout.addWidget(self._video_input_widget)

        self._cb_video_mode.toggled.connect(self._on_video_mode_toggled)

        self._frame_list = QListWidget()
        self._frame_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._frame_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._frame_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._frame_list.setIconSize(QSize(48, 48))
        self._frame_list.setToolTip(
            "Drag rows to reorder.\n"
            "Order = stitching sequence (first = leftmost / topmost)."
        )
        self._frame_list.model().rowsMoved.connect(self._on_rows_reordered)
        self._frame_list.currentRowChanged.connect(self._on_frame_selection_changed)
        frames_group_layout.addWidget(self._frame_list)

        self._btn_add = QPushButton("Add")
        self._btn_add.setToolTip("Add one or more image files to the stitch queue.")
        self._btn_remove = QPushButton("Remove")
        self._btn_remove.setToolTip("Remove the selected frame.")
        self._btn_up = QPushButton("↑ Up")
        self._btn_down = QPushButton("↓ Down")
        btn_grid = QGridLayout()
        btn_grid.setSpacing(4)
        for b in (self._btn_add, self._btn_remove, self._btn_up, self._btn_down):
            apply_shadow_effect(b, radius=4, y_offset=2)
        btn_grid.addWidget(self._btn_add, 0, 0)
        btn_grid.addWidget(self._btn_remove, 0, 1)
        btn_grid.addWidget(self._btn_up, 1, 0)
        btn_grid.addWidget(self._btn_down, 1, 1)

        self._btn_auto_order = QPushButton("⚡ Auto-Order")
        self._btn_auto_order.setToolTip(
            "Find the longest coherent sequence starting from the selected image."
        )
        self._btn_auto_order.clicked.connect(self._auto_order_sequence)
        apply_shadow_effect(self._btn_auto_order, radius=4, y_offset=2)
        btn_grid.addWidget(self._btn_auto_order, 2, 0, 1, 2)
        frames_group_layout.addLayout(btn_grid)

        self._btn_add.clicked.connect(self._add_frames)
        self._btn_remove.clicked.connect(self._remove_selected_frame)
        self._btn_up.clicked.connect(self._move_frame_up)
        self._btn_down.clicked.connect(self._move_frame_down)

        left_layout.addWidget(frames_group)

        # Pair selector
        pair_group = QGroupBox("Preview Pair")
        pair_layout = QFormLayout(pair_group)

        self._pair_combo = QComboBox()
        self._pair_combo.setToolTip(
            "Select the frame pair to display LoFTR matches for."
        )
        self._pair_combo.currentIndexChanged.connect(self._on_pair_changed)
        pair_layout.addRow("Pair:", self._pair_combo)

        self._match_count_label = QLabel("—")
        self._match_count_label.setStyleSheet("color: #aaa; font-size: 10px;")
        pair_layout.addRow("Matches:", self._match_count_label)

        self._conf_thresh_spin = QDoubleSpinBox()
        self._conf_thresh_spin.setRange(0.1, 0.99)
        self._conf_thresh_spin.setValue(0.4)
        self._conf_thresh_spin.setDecimals(2)
        self._conf_thresh_spin.setSingleStep(0.05)
        self._conf_thresh_spin.setToolTip(
            "Minimum LoFTR confidence for a displayed match."
        )
        pair_layout.addRow("Conf. threshold:", self._conf_thresh_spin)

        left_layout.addWidget(pair_group)
        left_layout.addStretch()
        splitter.addWidget(left)

        # ── CENTRE: match preview ─────────────────────────────────────
        centre = QWidget()
        centre_layout = QVBoxLayout(centre)
        centre_layout.setContentsMargins(4, 0, 4, 0)

        self._scene = _MatchScene(self)
        self._scene.affine_updated.connect(self._on_affine_updated)
        self._match_view = _MatchView(self._scene)

        view_toolbar = QHBoxLayout()
        self._btn_compute = QPushButton("Compute Matches")
        self._btn_compute.setToolTip("Run LoFTR on the selected pair.")
        self._btn_compute.setStyleSheet(
            "background:#1976D2; color:white; font-weight:bold; padding:5px 12px;"
        )
        apply_shadow_effect(self._btn_compute, radius=6, y_offset=2)

        self._btn_show_mask = QPushButton("Show Mask")
        self._btn_show_mask.setToolTip(
            "Overlay BiRefNet foreground mask on the left frame."
        )
        apply_shadow_effect(self._btn_show_mask, radius=6, y_offset=2)

        self._btn_reset_anchors = QPushButton("Reset Anchors")
        self._btn_reset_anchors.setToolTip(
            "Discard dragged-anchor overrides for this pair."
        )
        apply_shadow_effect(self._btn_reset_anchors, radius=6, y_offset=2)

        self._btn_fit = QPushButton("⊡ Fit")
        self._btn_fit.setMinimumWidth(80)
        self._btn_fit.setToolTip("Fit the view to the scene.")
        apply_shadow_effect(self._btn_fit, radius=4, y_offset=2)

        view_toolbar.addWidget(self._btn_compute)
        view_toolbar.addWidget(self._btn_show_mask)
        view_toolbar.addWidget(self._btn_reset_anchors)
        view_toolbar.addStretch()
        view_toolbar.addWidget(self._btn_fit)

        self._btn_compute.clicked.connect(self._compute_matches)
        self._btn_show_mask.clicked.connect(self._show_mask)
        self._btn_reset_anchors.clicked.connect(self._reset_anchors)
        self._btn_fit.clicked.connect(self._match_view.fit)

        centre_layout.addLayout(view_toolbar)
        centre_layout.addWidget(self._match_view)

        self._affine_label = QLabel("No manual alignment override active.")
        self._affine_label.setStyleSheet("color: #888; font-size: 10px; padding: 2px;")
        centre_layout.addWidget(self._affine_label)

        splitter.addWidget(centre)

        # ── RIGHT: pipeline config ────────────────────────────────────
        right = QWidget()
        right.setMinimumWidth(230)
        right.setMaximumWidth(290)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)

        pipeline_group = QGroupBox("Pipeline Stages")
        pipeline_form = QFormLayout(pipeline_group)

        self._cb_basic = QCheckBox("BaSiC photometric correction")
        self._cb_basic.setChecked(True)
        self._cb_basic.setToolTip(
            "Remove broadcast dimming and vignettes before matching."
        )
        pipeline_form.addRow(self._cb_basic)

        self._cb_birefnet = QCheckBox("BiRefNet foreground masking")
        self._cb_birefnet.setChecked(True)
        self._cb_birefnet.setToolTip(
            "Detect and exclude anime character regions from LoFTR matching.\n"
            "Strongly recommended — characters move between frames."
        )
        pipeline_form.addRow(self._cb_birefnet)

        self._cb_loftr = QCheckBox("LoFTR dense matching")
        self._cb_loftr.setChecked(True)
        self._cb_loftr.setToolTip(
            "Use LoFTR for subpixel-accurate correspondences.\n"
            "Falls back to template matching if unchecked or unavailable."
        )
        pipeline_form.addRow(self._cb_loftr)

        self._cb_ecc = QCheckBox("ECC sub-pixel refinement")
        self._cb_ecc.setChecked(True)
        self._cb_ecc.setToolTip(
            "Apply ECC after bundle adjustment for sub-pixel accuracy."
        )
        pipeline_form.addRow(self._cb_ecc)

        self._cb_composite_fg = QCheckBox("Composite foreground")
        self._cb_composite_fg.setChecked(True)
        self._cb_composite_fg.setToolTip(
            "Paste the character from the best single frame back onto the\n"
            "median background after stitching."
        )
        pipeline_form.addRow(self._cb_composite_fg)

        right_layout.addWidget(pipeline_group)

        render_group = QGroupBox("Renderer and Quality")
        render_form = QFormLayout(render_group)

        self._renderer_combo = QComboBox()
        self._renderer_combo.addItem("Temporal Median (recommended)", "median")
        self._renderer_combo.addItem("First-Valid Pixel", "first")
        self._renderer_combo.addItem("Sequential Laplacian Blend", "blend")
        self._renderer_combo.setToolTip(
            "median: Overmix-style suppression of MPEG noise and moving foreground.\n"
            "first: Fastest — takes first valid frame per pixel.\n"
            "blend: Sequential Laplacian blend (SCANS-style)."
        )
        render_form.addRow("Renderer:", self._renderer_combo)

        self._bands_spin = QSpinBox()
        self._bands_spin.setRange(1, 8)
        self._bands_spin.setValue(5)
        self._bands_spin.setToolTip(
            "Laplacian pyramid depth for multi-band seam blending."
        )
        render_form.addRow("Pyramid bands:", self._bands_spin)

        right_layout.addWidget(render_group)

        motion_group = QGroupBox("Motion Model and Edge Crop")
        motion_form = QFormLayout(motion_group)

        self._motion_model_combo = QComboBox()
        self._motion_model_combo.addItem("Translation", "translation")
        self._motion_model_combo.addItem("Affine 4-DOF", "affine")
        self._motion_model_combo.setToolTip(
            "translation: Fast homography / translation-only.\n"
            "affine: Full 4-DOF affine — better for rotated panels."
        )
        motion_form.addRow("Motion model:", self._motion_model_combo)

        self._edge_crop_spin = QSpinBox()
        self._edge_crop_spin.setRange(0, 100)
        self._edge_crop_spin.setValue(30)
        self._edge_crop_spin.setToolTip(
            "Pixels to strip from each long edge after final crop.\n"
            "Removes alignment artefacts at panorama borders."
        )
        motion_form.addRow("Edge crop (px):", self._edge_crop_spin)

        right_layout.addWidget(motion_group)

        ckpt_group = QGroupBox("StitchNet Checkpoint (Optional)")
        ckpt_layout = QVBoxLayout(ckpt_group)
        self._ckpt_path = QLineEdit()
        self._ckpt_path.setPlaceholderText("Path to AnimeStitchNet .pth…")
        self._ckpt_path.setToolTip(
            "Trained AnimeStitchNet checkpoint to supplement LoFTR.\n"
            "Leave blank to use LoFTR only."
        )
        btn_ckpt = QPushButton("Browse Checkpoint…")
        btn_ckpt.clicked.connect(self._browse_checkpoint)
        apply_shadow_effect(btn_ckpt, radius=4, y_offset=2)
        ckpt_layout.addWidget(self._ckpt_path)
        ckpt_layout.addWidget(btn_ckpt)
        right_layout.addWidget(ckpt_group)

        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)
        self._output_path = QLineEdit()
        self._output_path.setPlaceholderText("panorama.png")
        self._output_path.setToolTip("Destination path for the stitched panorama.")
        btn_output = QPushButton("Browse Output…")
        btn_output.clicked.connect(self._browse_output)
        apply_shadow_effect(btn_output, radius=4, y_offset=2)
        output_layout.addWidget(self._output_path)
        output_layout.addWidget(btn_output)

        self._cb_save_intermediate = QCheckBox("Save intermediate stage outputs")
        self._cb_save_intermediate.setChecked(False)
        self._cb_save_intermediate.setToolTip(
            "Saves the result of every pipeline stage to a '<output_stem>_stages/' folder.\n"
            "Stages 1–3: normalised frames\n"
            "Stage 4: background masks\n"
            "Stage 5: edge graph (JSON)\n"
            "Stages 6–7: affine matrices (JSON)\n"
            "Stage 8: canvas info (JSON)\n"
            "Stages 9–12: canvas images at each post-processing step\n"
            "Useful for diagnosing misalignments, bad crops, or MFSR artefacts."
        )
        output_layout.addWidget(self._cb_save_intermediate)

        right_layout.addWidget(output_group)

        # HITL group
        hitl_group = QGroupBox("HITL")
        hitl_layout = QVBoxLayout(hitl_group)

        self._cb_hitl_mode = QCheckBox("Human-in-the-loop review")
        self._cb_hitl_mode.setChecked(False)
        self._cb_hitl_mode.setToolTip(
            "Pause the pipeline at key checkpoints for manual review:\n"
            "  Stage 4 — Frame selection: exclude or reorder frames\n"
            "  Stage 5 — Edge graph: toggle or disable matches\n"
            "  Stage 8 — Canvas layout: nudge frame positions\n"
            "  Stage 9 — Render preview: inspect coverage heatmap"
        )
        hitl_layout.addWidget(self._cb_hitl_mode)

        # S88/S92: Session load/save controls
        _sess_row = QHBoxLayout()
        self._btn_load_session = QPushButton("Load")
        self._btn_load_session.setToolTip(
            "Load a saved HITL session file to replay all prior override decisions "
            "non-interactively (no dialogs)."
        )
        self._btn_load_session.clicked.connect(self._on_load_session)
        self._btn_browse_sessions = QPushButton("Browse...")
        self._btn_browse_sessions.setToolTip(
            "Open the HITL Session Browser to inspect, delete, export, "
            "or select a saved session for replay."
        )
        self._btn_browse_sessions.clicked.connect(self._on_browse_sessions)
        self._session_path_label = QLabel("No session loaded")
        self._session_path_label.setWordWrap(True)
        _sess_row.addWidget(self._btn_load_session)
        _sess_row.addWidget(self._btn_browse_sessions)
        _sess_row.addWidget(self._session_path_label, stretch=1)
        hitl_layout.addLayout(_sess_row)
        self._loaded_session_path: str | None = None

        right_layout.addWidget(hitl_group)

        right_layout.addStretch()
        splitter.addWidget(right_scroll)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([180, 1200, 220])

        persist_splitter(splitter, "StitchPanel/main")
        root.addWidget(splitter, stretch=1)

        # ── BOTTOM: progress + log ────────────────────────────────────
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 4, 0, 0)
        bottom_layout.setSpacing(4)

        self._stage_label = QLabel("Ready.")
        self._stage_label.setStyleSheet("color: #aaa; font-size: 10px;")
        bottom_layout.addWidget(self._stage_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, StitchWorker.TOTAL_STAGES)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%v / %m stages")
        bottom_layout.addWidget(self._progress)

        action_row = QHBoxLayout()
        self._btn_stitch = QPushButton("▶  Stitch Panorama")
        self._btn_stitch.setStyleSheet(
            "background:#4CAF50; color:white; font-weight:bold; padding:8px 18px;"
        )
        apply_shadow_effect(self._btn_stitch, radius=8, y_offset=3)
        self._btn_cancel = QPushButton("■  Cancel")
        self._btn_cancel.setStyleSheet(
            "background:#f44336; color:white; font-weight:bold; padding:8px 18px;"
        )
        self._btn_cancel.setEnabled(False)
        self.stitch_worker = None
        self.stitch_thread = None
        apply_shadow_effect(self._btn_cancel, radius=8, y_offset=3)

        self._btn_inspect_edges = QPushButton("⬡ Edges")
        self._btn_inspect_edges.setToolTip(
            "Inspect the LoFTR edge graph from the last stitch run.\n"
            "Only available when 'Save intermediate stage outputs' was enabled."
        )
        self._btn_inspect_edges.setEnabled(False)
        apply_shadow_effect(self._btn_inspect_edges, radius=4, y_offset=2)
        self._btn_inspect_edges.clicked.connect(self._inspect_edges)

        self._btn_inspect_canvas = QPushButton("⬗ Canvas")
        self._btn_inspect_canvas.setToolTip(
            "Inspect the final canvas layout from the last stitch run.\n"
            "Only available when 'Save intermediate stage outputs' was enabled."
        )
        self._btn_inspect_canvas.setEnabled(False)
        apply_shadow_effect(self._btn_inspect_canvas, radius=4, y_offset=2)
        self._btn_inspect_canvas.clicked.connect(self._inspect_canvas)

        self._btn_batch_stitch = QPushButton("⚏ Batch Stitch…")
        self._btn_batch_stitch.setToolTip(
            "Stitch every subdirectory of a chosen root directory (roadmap "
            "§4.1 Option A) -- each subdirectory is treated as one frame "
            "group. Shares the same .stitch_progress.json resume format as "
            "the CLI's `stitch --batch-dir` mode."
        )
        apply_shadow_effect(self._btn_batch_stitch, radius=4, y_offset=2)
        self._btn_batch_stitch.clicked.connect(self._open_batch_stitch_dialog)

        self._btn_stitch.clicked.connect(self._start_stitch)
        self._btn_cancel.clicked.connect(self._cancel_stitch)
        action_row.addWidget(self._btn_stitch)
        action_row.addWidget(self._btn_cancel)
        action_row.addWidget(self._btn_inspect_edges)
        action_row.addWidget(self._btn_inspect_canvas)
        action_row.addWidget(self._btn_batch_stitch)
        action_row.addStretch()
        bottom_layout.addLayout(action_row)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(110)
        self._log.setStyleSheet("font-family: monospace; font-size: 10px;")
        bottom_layout.addWidget(self._log)

        # ── Result preview (§2.11 B+C) ────────────────────────────────
        self._result_group = QGroupBox("Stitch Result")
        self._result_group.setVisible(False)
        result_layout = QVBoxLayout(self._result_group)
        result_layout.setContentsMargins(4, 4, 4, 4)
        result_layout.setSpacing(4)

        result_toolbar = QHBoxLayout()
        self._btn_before_after = QPushButton("◀ Before")
        self._btn_before_after.setCheckable(True)
        self._btn_before_after.setFixedWidth(88)
        self._btn_before_after.setToolTip(
            "Toggle between the first source frame (Before) and the stitched result (After)."
        )
        self._btn_before_after.toggled.connect(self._toggle_before_after)
        self._result_metrics_label = QLabel("")
        self._result_metrics_label.setStyleSheet("color: #aaa; font-size: 10px;")
        result_toolbar.addWidget(self._btn_before_after)
        result_toolbar.addStretch()
        result_toolbar.addWidget(self._result_metrics_label)
        result_layout.addLayout(result_toolbar)

        self._result_preview_label = QLabel()
        self._result_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._result_preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._result_preview_label.setMinimumHeight(100)
        self._result_preview_label.setMaximumHeight(200)
        self._result_preview_label.setStyleSheet(
            "background:#1a1a1a; border:1px solid #333;"
        )
        result_layout.addWidget(self._result_preview_label)
        bottom_layout.addWidget(self._result_group)

        self._metrics_signals.ready.connect(self._on_metrics_ready)

        root.addWidget(bottom)

        return panel


__all__ = ["_StitchPanelBuildMixin"]
