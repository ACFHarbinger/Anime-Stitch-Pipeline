"""Type-checking-only ``Protocol`` describing the ``StitchTab`` host.

``StitchTab`` (see ``manager.py``) is composed out of many per-sub-tab
mixins (``_StitchPanelBuildMixin``, ``_StitchFramesMixin``,
``_StitchExecutionMixin``, ``_StitchHitlReviewMixin``, ``_SeqPanelBuildMixin``,
``_SeqPanelHandlersMixin``, ``_StatsPanelMixin``,
``_StatsRecommendationsMixin``, ``_AnimClustersPanelMixin``,
``_HybridPanelMixin``, etc.). Each mixin references widgets/state that are
actually set by ``StitchTab.__init__``/``_build_*_panel()`` or provided by a
*different* mixin in the composing class -- invisible to mypy when a mixin
file is type-checked in isolation, since ``self`` is only known to be that
mixin's own class there.

This module exists purely to give mypy a complete picture of ``self`` for
those mixins. It has zero runtime effect: the Protocol is only imported
under ``TYPE_CHECKING``, and mixins only inherit from it in the same guard,
so at runtime the MRO/`__mro__` and attribute lookup behave exactly as
before.
"""

from __future__ import annotations

from typing import Any, Protocol

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QWidget,
)

from ..helpers import AnimClusterWorker, StatsWorker, StitchWorker
from ._match_editor import _MatchScene, _MatchView
from ._thumb_workers import _MetricsSignals, _ThumbHub


class _StitchTabHost(Protocol):
    """Attributes/methods a mixin expects to find on ``self``.

    Mirrors what ``StitchTab.__init__``/``_build_*_panel()`` set (manager.py
    and the individual ``_build_*_panel`` mixins) plus the cross-mixin
    methods each mixin borrows from a sibling mixin in the composed class.
    """

    # ── shared state (manager.py __init__) ───────────────────────────────
    _frame_paths: list[str]
    _stitch_worker: StitchWorker | None
    _metrics_signals: _MetricsSignals
    _tab_widget: QTabWidget
    _result_pix: QPixmap | None
    _before_pix: QPixmap | None
    _stats_worker: StatsWorker | None
    _anim_cluster_worker: AnimClusterWorker | None

    _frame_thumb_hub: _ThumbHub
    _frame_item_map: dict[str, QListWidgetItem]
    _cv_thumb_hub: _ThumbHub
    _cv_item_map: dict[str, QListWidgetItem]
    _seq_thumb_hub: _ThumbHub
    _seq_table_item_map: dict[str, QTableWidgetItem]
    _stats_ind_thumb_hub: _ThumbHub
    _stats_ind_item_map: dict[str, QTableWidgetItem]
    _stats_pw_thumb_hub_a: _ThumbHub
    _stats_pw_item_map_a: dict[str, QTableWidgetItem]
    _stats_pw_thumb_hub_b: _ThumbHub
    _stats_pw_item_map_b: dict[str, QTableWidgetItem]
    _anim_thumb_hub: _ThumbHub
    _anim_item_map: dict[str, QTableWidgetItem]

    # ── Stitch panel widgets (_stitch_panel_build.py) ────────────────────
    _frame_list: QListWidget
    _video_input_widget: QWidget
    _affine_label: QLabel
    _btn_add: QPushButton
    _btn_remove: QPushButton
    _btn_up: QPushButton
    _btn_down: QPushButton
    _btn_auto_order: QPushButton
    _pair_combo: QComboBox
    _scene: _MatchScene
    _match_view: _MatchView
    _match_count_label: QLabel
    _btn_before_after: QPushButton
    _btn_inspect_edges: QPushButton
    _btn_inspect_canvas: QPushButton
    _result_group: QGroupBox
    _result_preview_label: QLabel
    _result_metrics_label: QLabel
    _progress: QProgressBar
    _stage_label: QLabel
    _log: QTextEdit

    # ── Sequence-builder panel widgets (_seq_panel_build.py) ─────────────
    _seq_chain_table: QTableWidget
    _seq_progress: QProgressBar
    _seq_run_btn: QPushButton
    _seq_status: QLabel

    # ── Statistics panel widgets (_stats_panel.py) ───────────────────────
    _stats_progress: QProgressBar
    _stats_run_btn: QPushButton
    _stats_status: QLabel

    # ── methods provided by sibling mixins ───────────────────────────────
    def _add_frames(self, *args: Any, **kwargs: Any) -> Any: ...
    def _auto_order_sequence(self, *args: Any, **kwargs: Any) -> Any: ...
    def _browse_checkpoint(self, *args: Any, **kwargs: Any) -> Any: ...
    def _browse_output(self, *args: Any, **kwargs: Any) -> Any: ...
    def _browse_video(self, *args: Any, **kwargs: Any) -> Any: ...
    def _cancel_stitch(self, *args: Any, **kwargs: Any) -> Any: ...
    def _compute_matches(self, *args: Any, **kwargs: Any) -> Any: ...
    def _inspect_canvas(self, *args: Any, **kwargs: Any) -> Any: ...
    def _inspect_edges(self, *args: Any, **kwargs: Any) -> Any: ...
    def _log_append(self, msg: str) -> Any: ...
    def _make_frame_item(self, path: str) -> QListWidgetItem: ...
    def _move_frame_down(self, *args: Any, **kwargs: Any) -> Any: ...
    def _move_frame_up(self, *args: Any, **kwargs: Any) -> Any: ...
    def _on_affine_updated(self, *args: Any, **kwargs: Any) -> Any: ...
    def _on_browse_sessions(self, *args: Any, **kwargs: Any) -> Any: ...
    def _on_frame_selection_changed(self, *args: Any, **kwargs: Any) -> Any: ...
    def _on_load_session(self, *args: Any, **kwargs: Any) -> Any: ...
    def _on_metrics_ready(self, metrics: str) -> None: ...
    def _on_pair_changed(self, idx: int) -> Any: ...
    def _on_rows_reordered(self, *args: Any, **kwargs: Any) -> Any: ...
    def _on_video_mode_toggled(self, *args: Any, **kwargs: Any) -> Any: ...
    def _open_batch_stitch_dialog(self, *args: Any, **kwargs: Any) -> Any: ...
    def _refresh_pair_combo(self, *args: Any, **kwargs: Any) -> Any: ...
    def _remove_selected_frame(self, *args: Any, **kwargs: Any) -> Any: ...
    def _reset_anchors(self, *args: Any, **kwargs: Any) -> Any: ...
    def _seq_accept(self, *args: Any, **kwargs: Any) -> Any: ...
    def _seq_browse_anchor(self, *args: Any, **kwargs: Any) -> Any: ...
    def _seq_browse_dir(self, *args: Any, **kwargs: Any) -> Any: ...
    def _seq_insert_image(self, *args: Any, **kwargs: Any) -> Any: ...
    def _seq_load_from_stitch(self, *args: Any, **kwargs: Any) -> Any: ...
    def _seq_move_down(self, *args: Any, **kwargs: Any) -> Any: ...
    def _seq_move_up(self, *args: Any, **kwargs: Any) -> Any: ...
    def _seq_on_rows_moved(self, *args: Any, **kwargs: Any) -> Any: ...
    def _seq_remove_row(self, *args: Any, **kwargs: Any) -> Any: ...
    def _seq_replace_row(self, *args: Any, **kwargs: Any) -> Any: ...
    def _seq_run(self, *args: Any, **kwargs: Any) -> Any: ...
    def _show_mask(self, *args: Any, **kwargs: Any) -> Any: ...
    def _start_stitch(self, *args: Any, **kwargs: Any) -> Any: ...
    def _stats_build_recommendations(self, *args: Any, **kwargs: Any) -> Any: ...
    def _stats_on_error(self, msg: str) -> Any: ...
    def _toggle_before_after(self, checked: bool) -> None: ...


__all__ = ["_StitchTabHost"]
