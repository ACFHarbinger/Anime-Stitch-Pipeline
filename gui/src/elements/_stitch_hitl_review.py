"""Stitch sub-tab: HITL checkpoint review dialogs and session management.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change.
"""

from __future__ import annotations

import os

from asp_backend.ingestion.masking import (
    _compute_fg_masks_grounded_sam2,
    _refine_masks_with_clicks,
)
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

from gui.src.components import BatchStitchDialog
from .dialog import (
    BoundaryEditorDialog,
    CanvasInspectorDialog,
    CanvasLayoutInspectorDialog,
    CoverageHeatmapDialog,
    EdgeGraphInspectorDialog,
    EdgeReviewDialog,
    FinalOutputReviewDialog,
    HITLSessionViewerDialog,
    MaskReviewDialog,
    SeamDiagnosticDialog,
    SeamPainterDialog,
    SelectionReviewDialog,
    parse_canvas_json,
    parse_edge_json,
)


class _StitchHitlReviewMixin:
    @Slot(object)
    def _on_hitl_review_video(self, data: dict):
        """Checkpoint 0 pause: review video-extracted frames before pipeline starts (S84)."""
        vname = os.path.basename(data.get("video_path", "video"))
        self._stage_label.setText(f"HITL: Reviewing video frames from '{vname}'…")
        w = self._stitch_worker
        if w is None:
            return
        dlg = SelectionReviewDialog(
            data,
            title=f"Video Frame Review — {vname}",
            parent=self,
        )
        result = dlg.exec()
        if result == QDialog.DialogCode.Accepted:
            w.set_frame_override(dlg.selected_paths())
            w.resume()
        else:
            w.cancel()

    @Slot(object)
    def _on_hitl_review_frames(self, data: dict):
        """Stage 4 pause: show frame selection review dialog."""
        self._stage_label.setText("HITL: Reviewing frame selection…")
        w = self._stitch_worker
        if w is None:
            return
        dlg = SelectionReviewDialog(data, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            selected = dlg.selected_paths()
            if selected:
                w.set_frame_override(selected)
        else:
            w.cancel()
            return
        w.resume()

    @Slot(object)
    def _on_hitl_review_masks(self, data: dict):
        """Stage 4.5 pause: show mask/segmentation review dialog (Issue 10A2)."""
        self._stage_label.setText("HITL: Reviewing segmentation masks…")
        w = self._stitch_worker
        if w is None:
            return

        def _refine_cb(text_prompt: str, pos_clicks, neg_clicks, frame_idx: int):
            """Callback that runs inside _RefinementWorker's thread."""
            frames = data.get("frames", [])
            orig_masks = data.get("bg_masks", [])
            if text_prompt:
                # Grounded SAM-2: re-run segmentation from text prompt
                return _compute_fg_masks_grounded_sam2(
                    frames,
                    text_prompt,
                    birefnet_wrapper=None,
                    use_birefnet=False,
                )
            elif pos_clicks or neg_clicks:
                # Click refinement via live SAM-2 predictor preserved across HITL boundary
                predictor = data.get("sam2_predictor")
                state = data.get("sam2_inference_state")
                _fh = data.get("sam2_frame_h") or (
                    frames[0].shape[0] if frames else 1080
                )
                _fw = data.get("sam2_frame_w") or (
                    frames[0].shape[1] if frames else 1920
                )
                if predictor is not None and state is not None:
                    refined = _refine_masks_with_clicks(
                        predictor,
                        state,
                        pos_clicks=pos_clicks,
                        neg_clicks=neg_clicks,
                        frame_idx=frame_idx,
                        frame_h=_fh,
                        frame_w=_fw,
                    )
                    if refined:
                        return refined
                return list(orig_masks)
            return list(orig_masks)

        dlg = MaskReviewDialog(data, refine_callback=_refine_cb, parent=self)
        dlg.sig_mask_accepted.connect(lambda masks: w.set_mask_override(masks))
        result = dlg.exec()
        if result == QDialog.DialogCode.Accepted:
            w.resume()
        else:
            w.cancel()

    @Slot(object)
    def _on_hitl_review_edges(self, data: dict):
        """Stage 5 pause: show edge graph review dialog."""
        self._stage_label.setText("HITL: Reviewing edge graph…")
        w = self._stitch_worker
        if w is None:
            return
        dlg = EdgeReviewDialog(data=data, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            accepted = dlg.accepted_edges()
            if accepted:
                w.set_edge_override(accepted)
        else:
            w.cancel()
            return
        w.resume()

    @Slot(object)
    def _on_hitl_review_canvas(self, data: dict):
        """Stage 8 pause: show canvas layout inspector with nudge."""
        self._stage_label.setText("HITL: Reviewing canvas layout…")
        w = self._stitch_worker
        if w is None:
            return
        dlg = CanvasInspectorDialog(data=data, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_affines = dlg.adjusted_affines()
            if new_affines:
                w.set_affine_override(new_affines)
        else:
            w.cancel()
            return
        w.resume()

    @Slot(object)
    def _on_hitl_review_boundaries(self, data: dict):
        """Checkpoint 3.5 pause: show seam boundary editor."""
        self._stage_label.setText("HITL: Reviewing seam boundaries…")
        w = self._stitch_worker
        if w is None:
            return

        dlg = BoundaryEditorDialog(data=data, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            w.set_boundary_override(dlg.adjusted_boundaries())
        else:
            w.cancel()
            return
        w.resume()

    @Slot(object)
    def _on_hitl_review_seams(self, data: dict):
        """Checkpoint 4.6 pause: seam registration diagnostic inspector (§2.4A, S95)."""
        n_seams = len(data.get("boundaries", []))
        self._stage_label.setText(f"HITL: Seam inspector — {n_seams} seam(s)…")
        w = self._stitch_worker
        if w is None:
            return

        dlg = SeamDiagnosticDialog(data=data, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            overrides = dlg.get_overrides()
            if overrides:
                w.set_seam_override(overrides)
        else:
            w.cancel()
            return
        w.resume()

    @Slot(object)
    def _on_hitl_review_composite(self, data: dict):
        """Checkpoint 4.5 pause: post-composite seam painter (re-composite loop)."""
        iteration = data.get("iteration", 1)
        self._stage_label.setText(f"HITL: Seam painter — iteration {iteration}…")
        w = self._stitch_worker
        if w is None:
            return

        dlg = SeamPainterDialog(data=data, parent=self)
        result = dlg.exec()
        if result == SeamPainterDialog.RECOMPOSITE:
            mask = dlg.full_resolution_mask()
            if mask is not None:
                w.set_paint_mask(mask)
            # resume without paint_mask key → triggers another loop iteration with mask
            w.resume()
        elif result == QDialog.DialogCode.Accepted:
            # accept current output, no re-composite
            w.resume()
        else:
            w.cancel()

    @Slot(object)
    def _on_hitl_review_render(self, data: dict):
        """Stage 9 pause: show render preview and coverage heatmap."""
        self._stage_label.setText("HITL: Reviewing render…")
        w = self._stitch_worker
        if w is None:
            return
        dlg = CoverageHeatmapDialog(data=data, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            pass  # proceed without override
        else:
            w.set_render_cancel()
        w.resume()

    @Slot(object)
    def _on_hitl_review_output(self, data: dict):
        """Checkpoint 5 pause: final output RLHF quality rating (S87)."""
        self._stage_label.setText("HITL: Rate output quality…")
        w = self._stitch_worker
        if w is None:
            return

        dlg = FinalOutputReviewDialog(data=data, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            fb = dlg.get_feedback()
            if fb is not None:
                w.set_output_feedback(
                    overall_rating=fb["overall_rating"],
                    annotations=fb["annotations"],
                )
        w.resume()

    def _on_load_session(self):
        """Browse for a saved HITL session JSON and set it for the next run (S88)."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load HITL Session",
            str(
                __import__("pathlib").Path.home()
                / ".config"
                / "image-toolkit"
                / "hitl_sessions"
            ),
            "Session files (*.json);;All files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self._loaded_session_path = path
            self._session_path_label.setText(os.path.basename(path))
            self._session_path_label.setToolTip(path)
        else:
            self._loaded_session_path = None
            self._session_path_label.setText("No session loaded")

    def _on_browse_sessions(self):
        """Open the HITL Session Browser; if user selects a session, load it (S92)."""
        dlg = HITLSessionViewerDialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            path = dlg.selected_path()
            if path:
                self._loaded_session_path = path

                self._session_path_label.setText(os.path.basename(path))
                self._session_path_label.setToolTip(path)

    def _inspect_edges(self):
        edge_json = os.path.join(self._last_stages_dir, "stage05_edges.json")
        if not os.path.isfile(edge_json):
            QMessageBox.warning(
                self,
                "No Edge Data",
                "No stage05_edges.json found.\n\n"
                "Enable 'Save intermediate stage outputs' and re-run the stitch.",
            )
            return
        try:
            edges = parse_edge_json(edge_json)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        dlg = EdgeGraphInspectorDialog(
            edges=edges, frame_paths=list(self._frame_paths), parent=self
        )
        dlg.exec()

    def _inspect_canvas(self):
        canvas_json = os.path.join(self._last_stages_dir, "stage08_canvas_info.json")
        if not os.path.isfile(canvas_json):
            QMessageBox.warning(
                self,
                "No Canvas Data",
                "No stage08_canvas_info.json found.\n\n"
                "Enable 'Save intermediate stage outputs' and re-run the stitch.",
            )
            return
        try:
            data = parse_canvas_json(canvas_json)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        dlg = CanvasLayoutInspectorDialog(canvas_data=data, parent=self)
        dlg.exec()

    def _open_batch_stitch_dialog(self):
        """Roadmap §4.1 Option A: stitch every subdirectory of a chosen root
        directory, showing per-item progress. Modeless-in-spirit but run as
        a modal dialog (exec()) since its own worker thread and widgets are
        self-contained -- doesn't share state with the single-run
        StitchWorker/node-graph flow above."""
        dlg = BatchStitchDialog(parent=self)
        dlg.exec()

    def _log_append(self, msg: str):
        self._log.append(msg)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())


__all__ = ["_StitchHitlReviewMixin"]
