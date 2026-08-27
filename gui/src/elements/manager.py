"""Composed ``StitchTab`` class: state bootstrap, panel wiring, and the
session/tab-config methods shared across every sub-tab.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change. Each
sub-tab's ``_build_*_panel()`` UI-construction method and its handler methods
live in a mixin file alongside it; this module only wires them into the
QTabWidget and owns the state that's genuinely shared (``collect``/
``set_config``/``get_default_config`` touch widgets across every panel).
"""

from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import QThread, QTimer
from PySide6.QtGui import QKeyEvent, QPixmap
from PySide6.QtWidgets import QListWidgetItem, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget

from ..helpers import (
    AdjustWorker,
    AnimClusterWorker,
    CanvasWorker,
    GraphStitchWorker,
    MaskPreviewWorker,
    MatchWorker,
    SequenceBuilderWorker,
    StatsWorker,
    StitchWorker,
)
from gui.src.utils.manager.shortcut_manager import get_registry
from ._adjust_panel import _AdjustPanelMixin
from ._anim_clusters_panel import _AnimClustersPanelMixin
from ._canvas_panel import _CanvasPanelMixin
from ._graph_panel import _GraphPanelMixin
from ._hybrid_panel import _HybridPanelMixin
from ._node_graph_items import _StitchOpNode
from ._seq_panel_build import _SeqPanelBuildMixin
from ._seq_panel_handlers import _SeqPanelHandlersMixin
from ._stats_panel import _StatsPanelMixin
from ._stats_recommendations import _StatsRecommendationsMixin
from ._stitch_execution import _StitchExecutionMixin
from ._stitch_frames import _StitchFramesMixin
from ._stitch_hitl_review import _StitchHitlReviewMixin
from ._stitch_panel_build import _StitchPanelBuildMixin
from ._thumb_workers import _MetricsSignals, _ScansComparisonSignals, _ThumbHub


class StitchTab(
    _StitchPanelBuildMixin,
    _StitchFramesMixin,
    _StitchExecutionMixin,
    _StitchHitlReviewMixin,
    _GraphPanelMixin,
    _AdjustPanelMixin,
    _CanvasPanelMixin,
    _StatsPanelMixin,
    _StatsRecommendationsMixin,
    _AnimClustersPanelMixin,
    _SeqPanelBuildMixin,
    _SeqPanelHandlersMixin,
    _HybridPanelMixin,
    QWidget,
):
    """
    Full image stitching suite focused on anime wallpaper creation.

    Stitch  — intelligent multi-frame panorama stitching with LoFTR matching.
    Adjust  — per-image corrections (tone, color, geometry) before stitching.
    Canvas  — drag-and-drop style layout composer with wallpaper presets.
    """

    def __init__(self):
        super().__init__()

        # ── Stitch state ─────────────────────────────────────────────────
        self._frame_paths: list[str] = []
        self._manual_affines: dict[tuple[int, int], np.ndarray] = {}
        self._current_pair: tuple[int, int] = (0, 1)
        self._last_selected_dir: str = ""
        self._last_stages_dir: str = ""

        self._stitch_thread: QThread | None = None
        self._stitch_worker: StitchWorker | None = None

        # ── Result preview state (§2.11 / 2.6B+C) ───────────────────────
        self._result_pix: QPixmap | None = None
        self._scans_pix: QPixmap | None = None
        self._comparison_metrics: dict[str, str] = {}
        self._metrics_signals = _MetricsSignals()
        self._scans_comparison_signals = _ScansComparisonSignals()
        self._match_thread: QThread | None = None
        self._match_worker: MatchWorker | None = None
        self._mask_thread: QThread | None = None
        self._mask_worker: MaskPreviewWorker | None = None

        # ── Adjust state ──────────────────────────────────────────────────
        self._adj_img_path: str | None = None
        self._adj_flip_h: bool = False
        self._adj_flip_v: bool = False
        self._adj_thread: QThread | None = None
        self._adj_worker: AdjustWorker | None = None

        self._adj_debounce = QTimer()
        self._adj_debounce.setSingleShot(True)
        self._adj_debounce.setInterval(220)
        self._adj_debounce.timeout.connect(self._adj_run_preview)

        # ── Canvas state ──────────────────────────────────────────────────
        self._cv_paths: list[str] = []
        self._cv_bg_color: tuple[int, int, int] = (0, 0, 0)
        self._cv_thread: QThread | None = None
        self._cv_worker: CanvasWorker | None = None

        # ── Graph state ───────────────────────────────────────────────────
        self._graph_thread: QThread | None = None
        self._graph_worker: GraphStitchWorker | None = None
        self._last_selected_op: _StitchOpNode | None = None

        # ── Stats state ───────────────────────────────────────────────────
        self._stats_worker: StatsWorker | None = None
        self._stats_dir_path: str = ""

        # ── Sequence-builder state ────────────────────────────────────────
        self._seq_worker: SequenceBuilderWorker | None = None
        self._seq_anchor_path: str = ""
        self._seq_dir_path: str = ""
        self._seq_chain: list[dict] = []  # current built chain

        # ── Frame-list thumbnail loader ───────────────────────────────────
        self._frame_thumb_hub = _ThumbHub()
        self._frame_thumb_hub.loaded.connect(self._on_frame_thumb_loaded)
        self._frame_item_map: dict[str, QListWidgetItem] = {}

        # ── Canvas thumbnail loader ───────────────────────────────────────
        self._cv_thumb_hub = _ThumbHub()
        self._cv_thumb_hub.loaded.connect(self._on_cv_thumb_loaded)
        self._cv_item_map: dict[str, QListWidgetItem] = {}

        # ── Sequence Builder table thumbnail loader ───────────────────────
        self._seq_thumb_hub = _ThumbHub()
        self._seq_thumb_hub.loaded.connect(self._on_seq_table_thumb_loaded)
        self._seq_table_item_map: dict[str, QTableWidgetItem] = {}

        # ── Statistics thumbnail loaders ──────────────────────────────────
        self._stats_ind_thumb_hub = _ThumbHub()
        self._stats_ind_thumb_hub.loaded.connect(self._on_stats_ind_thumb_loaded)
        self._stats_ind_item_map: dict[str, QTableWidgetItem] = {}

        self._stats_pw_thumb_hub_a = _ThumbHub()
        self._stats_pw_thumb_hub_a.loaded.connect(self._on_stats_pw_thumb_loaded_a)
        self._stats_pw_item_map_a: dict[str, QTableWidgetItem] = {}

        self._stats_pw_thumb_hub_b = _ThumbHub()
        self._stats_pw_thumb_hub_b.loaded.connect(self._on_stats_pw_thumb_loaded_b)
        self._stats_pw_item_map_b: dict[str, QTableWidgetItem] = {}

        # ── Animation Clusters state ───────────────────────────────────────────
        self._anim_cluster_worker: AnimClusterWorker | None = None
        self._anim_cluster_paths: list[str] = []
        self._anim_cluster_dir_path: str = ""
        self._anim_thumb_hub = _ThumbHub()
        self._anim_thumb_hub.loaded.connect(self._on_anim_thumb_loaded)
        self._anim_item_map: dict[str, QTableWidgetItem] = {}

        self._init_ui()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Dispatch configurable Stitch-scope shortcuts through their buttons."""
        reg = get_registry()
        actions = (
            ("stitch.run", "_btn_stitch"),
            ("stitch.cancel", "_btn_cancel"),
            ("stitch.compute_matches", "_btn_compute"),
            ("stitch.generate_scans_comparison", "_btn_generate_scans"),
        )
        for action_id, button_name in actions:
            if reg.matches(event, action_id):
                button = getattr(self, button_name)
                if button.isEnabled():
                    button.click()
                event.accept()
                return
        super().keyPressEvent(event)

    # ======================================================================
    # Top-level UI
    # ======================================================================

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Build each panel and keep a named reference so the main window can
        # promote them to top-level tabs without losing signal/slot bindings.
        self.stitch_panel = self._build_stitch_panel()
        self.graph_panel = self._build_graph_panel()
        self.adjust_panel = self._build_adjust_panel()
        self.canvas_panel = self._build_canvas_panel()
        self.stats_panel = self._build_stats_panel()
        self.seq_builder_panel = self._build_seq_panel()
        self.hybrid_stitch_panel = self._build_hybrid_panel()
        self.anim_clusters_panel = self._build_anim_clusters_panel()

        # Internal tab widget kept for backward-compat and as a fallback parent;
        # the main window may reparent individual panels into top-level tabs.
        self._tab_widget = QTabWidget()
        self._tab_widget.addTab(self.stitch_panel, "Stitch")
        self._tab_widget.addTab(self.graph_panel, "Graph")
        self._tab_widget.addTab(self.adjust_panel, "Adjust")
        self._tab_widget.addTab(self.canvas_panel, "Canvas")
        self._tab_widget.addTab(self.stats_panel, "Statistics")
        self._tab_widget.addTab(self.seq_builder_panel, "Sequence Builder")
        self._tab_widget.addTab(self.hybrid_stitch_panel, "Hybrid Stitch")
        self._tab_widget.addTab(self.anim_clusters_panel, "Animation Clusters")

        root.addWidget(self._tab_widget)

    # ======================================================================
    # Settings persistence
    # ======================================================================

    def collect(self) -> dict:
        return {
            # Stitch
            "frame_paths": list(self._frame_paths),
            "output_path": self._output_path.text(),
            "use_basic": self._cb_basic.isChecked(),
            "use_birefnet": self._cb_birefnet.isChecked(),
            "use_loftr": self._cb_loftr.isChecked(),
            "use_ecc": self._cb_ecc.isChecked(),
            "renderer": self._renderer_combo.currentData(),
            "composite_fg": self._cb_composite_fg.isChecked(),
            "laplacian_bands": self._bands_spin.value(),
            "stitch_net_ckpt": self._ckpt_path.text(),
            "conf_threshold": self._conf_thresh_spin.value(),
            "motion_model": self._motion_model_combo.currentData(),
            "edge_crop": self._edge_crop_spin.value(),
            "save_intermediate": self._cb_save_intermediate.isChecked(),
            # Adjust
            "adj_brightness": self._adj_brightness.value(),
            "adj_contrast": self._adj_contrast.value(),
            "adj_gamma": self._adj_gamma.value(),
            "adj_saturation": self._adj_saturation.value(),
            "adj_hue": self._adj_hue.value(),
            "adj_sharpen": self._adj_sharpen.value(),
            "adj_blur": self._adj_blur.value(),
            "adj_angle": self._adj_angle_spin.value(),
            "adj_crop_idx": self._adj_crop_combo.currentIndex(),
            # Canvas
            "cv_width": self._cv_width_spin.value(),
            "cv_height": self._cv_height_spin.value(),
            "cv_layout": (
                "horizontal"
                if self._cv_radio_h.isChecked()
                else "vertical"
                if self._cv_radio_v.isChecked()
                else "grid"
            ),
            "cv_cols": self._cv_cols_spin.value(),
            "cv_gap": self._cv_gap_spin.value(),
            "cv_bg_color": list(self._cv_bg_color),
            "cv_scale_mode": self._cv_scale_combo.currentData(),
            "cv_paths": list(self._cv_paths),
            # Sub-tab
            "active_subtab_index": self._tab_widget.currentIndex(),
        }

    def set_config(self, cfg: dict):  # noqa: C901
        # Stitch
        if "output_path" in cfg:
            self._output_path.setText(cfg["output_path"])
        if "use_basic" in cfg:
            self._cb_basic.setChecked(cfg["use_basic"])
        if "use_birefnet" in cfg:
            self._cb_birefnet.setChecked(cfg["use_birefnet"])
        if "use_loftr" in cfg:
            self._cb_loftr.setChecked(cfg["use_loftr"])
        if "use_ecc" in cfg:
            self._cb_ecc.setChecked(cfg["use_ecc"])
        if "renderer" in cfg:
            idx = self._renderer_combo.findData(cfg["renderer"])
            if idx >= 0:
                self._renderer_combo.setCurrentIndex(idx)
        if "composite_fg" in cfg:
            self._cb_composite_fg.setChecked(cfg["composite_fg"])
        if "laplacian_bands" in cfg:
            self._bands_spin.setValue(cfg["laplacian_bands"])
        if "stitch_net_ckpt" in cfg:
            self._ckpt_path.setText(cfg["stitch_net_ckpt"])
        if "conf_threshold" in cfg:
            self._conf_thresh_spin.setValue(cfg["conf_threshold"])
        if "motion_model" in cfg:
            idx = self._motion_model_combo.findData(cfg["motion_model"])
            if idx >= 0:
                self._motion_model_combo.setCurrentIndex(idx)
        if "edge_crop" in cfg:
            self._edge_crop_spin.setValue(cfg["edge_crop"])
        if "save_intermediate" in cfg:
            self._cb_save_intermediate.setChecked(cfg["save_intermediate"])
        if "frame_paths" in cfg:
            for p in cfg["frame_paths"]:
                if p and os.path.isfile(p) and p not in self._frame_paths:
                    self._frame_paths.append(p)
                    self._frame_list.addItem(self._make_frame_item(p))
            self._refresh_pair_combo()
        # Adjust
        for key, sl in [
            ("adj_brightness", self._adj_brightness),
            ("adj_contrast", self._adj_contrast),
            ("adj_gamma", self._adj_gamma),
            ("adj_saturation", self._adj_saturation),
            ("adj_hue", self._adj_hue),
            ("adj_sharpen", self._adj_sharpen),
            ("adj_blur", self._adj_blur),
        ]:
            if key in cfg:
                sl.setValue(cfg[key])
        if "adj_angle" in cfg:
            self._adj_angle_spin.setValue(cfg["adj_angle"])
        if "adj_crop_idx" in cfg:
            self._adj_crop_combo.setCurrentIndex(cfg["adj_crop_idx"])
        # Canvas
        if "cv_width" in cfg:
            self._cv_width_spin.setValue(cfg["cv_width"])
        if "cv_height" in cfg:
            self._cv_height_spin.setValue(cfg["cv_height"])
        if "cv_layout" in cfg:
            layout = cfg["cv_layout"]
            if layout == "horizontal":
                self._cv_radio_h.setChecked(True)
            elif layout == "vertical":
                self._cv_radio_v.setChecked(True)
            else:
                self._cv_radio_g.setChecked(True)
        if "cv_cols" in cfg:
            self._cv_cols_spin.setValue(cfg["cv_cols"])
        if "cv_gap" in cfg:
            self._cv_gap_spin.setValue(cfg["cv_gap"])
        if "cv_bg_color" in cfg:
            self._cv_bg_color = tuple(cfg["cv_bg_color"])
            self._cv_update_bg_button()
        if "cv_scale_mode" in cfg:
            idx = self._cv_scale_combo.findData(cfg["cv_scale_mode"])
            if idx >= 0:
                self._cv_scale_combo.setCurrentIndex(idx)
        if "cv_paths" in cfg:
            for p in cfg["cv_paths"]:
                if p and os.path.isfile(p) and p not in self._cv_paths:
                    self._cv_paths.append(p)
                    self._cv_list.addItem(self._make_cv_item(p))
        # Restore active sub-tab
        if "active_subtab_index" in cfg:
            idx = cfg["active_subtab_index"]
            if isinstance(idx, int) and 0 <= idx < self._tab_widget.count():
                self._tab_widget.setCurrentIndex(idx)

    def get_default_config(self) -> dict:
        return self.collect()


__all__ = ["StitchTab"]
