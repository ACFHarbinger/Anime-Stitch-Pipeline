"""Stitch sub-tab: checkpoint/video/output browsing and pipeline execution.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PySide6.QtCore import QAbstractAnimation, Qt, QThreadPool, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from ..helpers import StitchWorker
from ._thumb_workers import _MetricsTask, _ScansComparisonTask

if TYPE_CHECKING:
    from ._stitch_tab_protocol import _StitchTabHost

    # Type-checking-only base: gives mypy visibility into attributes set by
    # StitchTab.__init__ / sibling mixins (see _stitch_tab_protocol.py).
    # Zero runtime effect -- at runtime this mixin still only inherits object.
    class _Base(_StitchTabHost, QWidget): ...
else:
    _Base = object


class _StitchExecutionMixin(_Base):
    def _browse_checkpoint(self):
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select StitchNet checkpoint",
            "",
            "PyTorch (*.pth *.pt)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if p:
            self._ckpt_path.setText(p)

    def _on_video_mode_toggled(self, checked: bool):
        """Show/hide video input panel and disable/enable the image frame list."""
        self._video_input_widget.setVisible(checked)
        self._frame_list.setEnabled(not checked)
        self._btn_add.setEnabled(not checked)
        self._btn_remove.setEnabled(not checked)
        self._btn_up.setEnabled(not checked)
        self._btn_down.setEnabled(not checked)
        self._btn_auto_order.setEnabled(not checked)

    def _browse_video(self):
        """Open a file dialog to select a video file."""
        start_dir = self._last_selected_dir or ""
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            start_dir,
            "Video Files (*.mp4 *.mkv *.avi *.mov *.webm *.flv);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if p:
            self._video_path_edit.setText(p)

    def _browse_output(self):
        start_dir = self._last_selected_dir or (
            os.path.dirname(self._frame_paths[-1]) if self._frame_paths else ""
        )
        current_text = self._output_path.text().strip()
        filename = os.path.basename(current_text) if current_text else "panorama.png"
        default_file = os.path.join(start_dir, filename) if start_dir else filename
        p, _ = QFileDialog.getSaveFileName(
            self,
            "Save Panorama As",
            default_file,
            "Images (*.png *.webp *.jpg)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if p:
            self._output_path.setText(p)

    def _start_stitch(self):
        _use_video = self._cb_video_mode.isChecked()
        _video_path = self._video_path_edit.text().strip() if _use_video else ""

        if _use_video:
            if not _video_path or not os.path.isfile(_video_path):
                QMessageBox.warning(
                    self,
                    "Video not found",
                    f"'{_video_path}' does not exist. Select a valid video file.",
                )
                return
        elif len(self._frame_paths) < 2:
            QMessageBox.warning(
                self, "Not enough frames", "Add at least 2 source frames."
            )
            return

        out = self._output_path.text().strip()
        if not out:
            start_dir = (
                os.path.dirname(self._frame_paths[-1]) if self._frame_paths else ""
            )
            default_file = (
                os.path.join(start_dir, "panorama.png") if start_dir else "panorama.png"
            )
            out, _ = QFileDialog.getSaveFileName(
                self,
                "Save Panorama As",
                default_file,
                "Images (*.png *.webp *.jpg)",
                options=QFileDialog.Option.DontUseNativeDialog,
            )
            if not out:
                return
            self._output_path.setText(out)

        self._btn_stitch.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.setValue(0)
        self._log.clear()
        self._stage_label.setText("Initialising pipeline…")
        if _use_video:
            self._log_append(
                f"[Stitch] Starting from video '{os.path.basename(_video_path)}' → '{out}'"
            )
        else:
            self._log_append(
                f"[Stitch] Starting — {len(self._frame_paths)} frames → '{out}'"
            )
        if self._manual_affines:
            self._log_append(
                f"[Stitch] Manual affine overrides active for "
                f"{len(self._manual_affines)} pair(s)."
            )

        pipeline_config = {
            "use_basic": self._cb_basic.isChecked(),
            "use_birefnet": self._cb_birefnet.isChecked(),
            "use_loftr": self._cb_loftr.isChecked(),
            "use_ecc": self._cb_ecc.isChecked(),
            "renderer": self._renderer_combo.currentData(),
            "composite_fg": self._cb_composite_fg.isChecked(),
            "laplacian_bands": self._bands_spin.value(),
            "stitch_net_ckpt": self._ckpt_path.text().strip(),
            "motion_model": self._motion_model_combo.currentData(),
            "edge_crop": self._edge_crop_spin.value(),
            "save_intermediate": self._cb_save_intermediate.isChecked(),
        }

        _hitl = self._cb_hitl_mode.isChecked()
        self._stitch_worker = StitchWorker(
            image_paths=[] if _use_video else list(self._frame_paths),
            output_path=out,
            pipeline_config=pipeline_config,
            manual_affines=dict(self._manual_affines),
            hitl_mode=_hitl,
            video_path=_video_path or None,
            video_n_frames=self._video_n_frames_spin.value() if _use_video else 20,
            session_path=self._loaded_session_path,
        )
        self._last_stages_dir = self._stitch_worker._intermediate_dir
        self._btn_inspect_edges.setEnabled(False)
        self._btn_inspect_canvas.setEnabled(False)
        self._stitch_thread = self._stitch_worker
        self._stitch_worker.sig_stage.connect(self._on_stage)
        self._stitch_worker.sig_log.connect(self._log_append)
        self._stitch_worker.sig_finished.connect(self._on_stitch_finished)
        self._stitch_worker.sig_error.connect(self._on_stitch_error)
        self._stitch_worker.finished.connect(self._on_stitch_thread_done)
        self._stitch_worker.finished.connect(self._stitch_worker.deleteLater)
        if _hitl:
            if _use_video:
                self._stitch_worker.sig_review_video.connect(self._on_hitl_review_video)
            self._stitch_worker.sig_review_frames.connect(self._on_hitl_review_frames)
            self._stitch_worker.sig_review_masks.connect(self._on_hitl_review_masks)
            self._stitch_worker.sig_review_edges.connect(self._on_hitl_review_edges)
            self._stitch_worker.sig_review_canvas.connect(self._on_hitl_review_canvas)
            self._stitch_worker.sig_review_boundaries.connect(
                self._on_hitl_review_boundaries
            )
            self._stitch_worker.sig_review_seams.connect(self._on_hitl_review_seams)
            self._stitch_worker.sig_review_composite.connect(
                self._on_hitl_review_composite
            )
            self._stitch_worker.sig_review_render.connect(self._on_hitl_review_render)
            self._stitch_worker.sig_review_output.connect(self._on_hitl_review_output)
        self._stitch_worker.start()

    def _cancel_stitch(self):
        if self._stitch_worker:
            self._stitch_worker.cancel()
            self._stitch_worker.quit()
            self._stitch_worker.wait()
            self._btn_cancel.setEnabled(False)
        self._stitch_worker = None
        self._stitch_thread = None
        self._log_append("[Stitch] Cancellation requested...")

    @Slot(int, int, str)
    def _on_stage(self, current: int, total: int, label: str):
        self._progress.setValue(current)
        self._stage_label.setText(f"Stage {current}/{total}: {label}")

    @Slot(str)
    def _on_stitch_finished(self, output_path: str):
        self._log_append(f"[Stitch] Complete. Saved to: {output_path}")
        edge_json = os.path.join(self._last_stages_dir, "stage05_edges.json")
        if os.path.isfile(edge_json):
            self._btn_inspect_edges.setEnabled(True)
            self._log_append(
                "[Stitch] Edge graph available — click '⬡ Edges' to inspect."
            )
        canvas_json = os.path.join(self._last_stages_dir, "stage08_canvas_info.json")
        if os.path.isfile(canvas_json):
            self._btn_inspect_canvas.setEnabled(True)
            self._log_append(
                "[Stitch] Canvas layout available — click '⬗ Canvas' to inspect."
            )
        # S88: show autosaved session path
        _sess_info = ""
        if self._stitch_worker and self._stitch_worker.current_session_path:
            _sp = self._stitch_worker.current_session_path
            self._log_append(f"[HITL] Session autosaved: {_sp}")
            _sess_info = f"\n\nHITL session saved to:\n{_sp}"
        self._show_stitch_result(output_path)
        QMessageBox.information(
            self, "Stitch Complete", f"Panorama saved to:\n{output_path}{_sess_info}"
        )

    @Slot(str)
    def _on_stitch_error(self, msg: str):
        self._log_append(f"[Stitch] Error: {msg}")
        if "Cancelled" not in msg:
            QMessageBox.critical(self, "Stitch Error", msg)

    def _on_stitch_thread_done(self):
        self._btn_stitch.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._stitch_worker = None
        self._stitch_thread = None
        self._progress.setValue(0)
        self._stage_label.setText("Ready.")

    # ── Result preview helpers (§2.11 / 2.6B+C) ─────────────────────────

    def _show_stitch_result(self, output_path: str) -> None:
        """Show the ASP result and offer an on-demand SCANS comparison."""
        self._result_pix: QPixmap | None = QPixmap(output_path)
        self._scans_pix = None
        self._comparison_metrics.clear()
        self._result_group.setVisible(True)
        self._btn_before_after.setChecked(False)
        self._btn_before_after.setText("ASP ◀")
        self._btn_before_after.setEnabled(False)
        self._btn_generate_scans.setEnabled(len(self._frame_paths) >= 2)
        self._btn_generate_scans.setText("Generate SCANS Comparison")
        self._result_metrics_label.setText("Computing metrics…")
        self._update_result_preview()
        QThreadPool.globalInstance().start(
            _MetricsTask(output_path, self._metrics_signals)
        )

    def _generate_scans_comparison(self) -> None:
        """Create a SCANS baseline alongside the completed ASP panorama."""
        if not self._result_pix or self._result_pix.isNull() or len(self._frame_paths) < 2:
            return
        output_path = self._output_path.text().strip()
        root, ext = os.path.splitext(output_path)
        scans_output_path = f"{root}_scans{ext or '.png'}"
        self._btn_generate_scans.setEnabled(False)
        self._btn_generate_scans.setText("Generating SCANS…")
        self._result_metrics_label.setText("Generating SCANS comparison…")
        QThreadPool.globalInstance().start(
            _ScansComparisonTask(
                list(self._frame_paths),
                output_path,
                scans_output_path,
                self._scans_comparison_signals,
            )
        )

    @Slot(str, object)
    def _on_scans_comparison_ready(self, output_path: str, metrics: object) -> None:
        scans_pix = QPixmap(output_path)
        if scans_pix.isNull():
            self._on_scans_comparison_failed("OpenCV SCANS did not produce a readable image.")
            return
        self._scans_pix = scans_pix
        self._comparison_metrics = dict(metrics) if isinstance(metrics, dict) else {}
        self._btn_before_after.setEnabled(True)
        self._btn_generate_scans.setText("SCANS comparison ready")
        self._update_result_preview()

    @Slot(str)
    def _on_scans_comparison_failed(self, message: str) -> None:
        self._btn_generate_scans.setEnabled(True)
        self._btn_generate_scans.setText("Retry SCANS Comparison")
        self._result_metrics_label.setText("SCANS comparison unavailable")
        self._log_append(f"[Stitch] SCANS comparison failed: {message}")

    def _toggle_before_after(self, checked: bool) -> None:
        self._btn_before_after.setText("SCANS ▶" if checked else "ASP ◀")
        if self._result_fade.state() == QAbstractAnimation.State.Running:
            self._result_fade.stop()
        self._result_fade.start()

    def _update_result_preview(self) -> None:
        is_scans = self._btn_before_after.isChecked()
        pix = self._scans_pix if is_scans else self._result_pix
        if pix is None or pix.isNull():
            return
        lw = self._result_preview_label.width()
        lh = self._result_preview_label.height()
        scaled = pix.scaled(
            max(1, lw),
            max(1, lh),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._result_preview_label.setPixmap(scaled)
        label = "SCANS" if is_scans else "ASP"
        if label in self._comparison_metrics:
            self._result_metrics_label.setText(f"{label}: {self._comparison_metrics[label]}")

    @Slot(str)
    def _on_metrics_ready(self, metrics: str) -> None:
        self._result_metrics_label.setText(metrics)


__all__ = ["_StitchExecutionMixin"]
