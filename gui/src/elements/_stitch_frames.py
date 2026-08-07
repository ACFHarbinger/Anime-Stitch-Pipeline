"""Stitch sub-tab: frame list, pair selection, and LoFTR match preview logic.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import cv2
import numpy as np
from asp_backend import AnimeStitchPipeline
from PySide6.QtCore import Qt, QThreadPool, QTimer, Slot
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QDialog, QListWidgetItem, QMessageBox, QWidget

from ..helpers import MaskPreviewWorker, MatchWorker
from ._thumb_workers import _ThumbTask
from ._thumbnail_file_picker import _ThumbnailFilePicker

if TYPE_CHECKING:
    from ._stitch_tab_protocol import _StitchTabHost

    # Type-checking-only base: gives mypy visibility into attributes set by
    # StitchTab.__init__ / sibling mixins (see _stitch_tab_protocol.py).
    # Zero runtime effect -- at runtime this mixin still only inherits object.
    class _Base(_StitchTabHost, QWidget): ...
else:
    _Base = object


class _StitchFramesMixin(_Base):
    def _make_frame_item(self, path: str) -> QListWidgetItem:
        """Create a QListWidgetItem for the stitch frame list and enqueue thumb load."""
        item = QListWidgetItem(os.path.basename(path))
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        self._frame_item_map[path] = item
        QThreadPool.globalInstance().start(
            _ThumbTask(path, 48, 0, self._frame_thumb_hub)
        )
        return item

    @Slot(str, int, object)
    def _on_frame_thumb_loaded(self, path: str, _generation: int, img: QImage):
        item = self._frame_item_map.get(path)
        if item and not img.isNull():
            item.setIcon(QIcon(QPixmap.fromImage(img)))

    def _add_frames(self):
        start_dir = self._last_selected_dir or (
            os.path.dirname(self._frame_paths[-1]) if self._frame_paths else ""
        )
        dlg = _ThumbnailFilePicker(self, start_dir=start_dir)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if dlg.selected_paths():
            self._last_selected_dir = os.path.dirname(dlg.selected_paths()[0])
        else:
            self._last_selected_dir = dlg._current_dir
        for p in dlg.selected_paths():
            if p and p not in self._frame_paths:
                self._frame_paths.append(p)
                self._frame_list.addItem(self._make_frame_item(p))
        self._refresh_pair_combo()

    def _remove_selected_frame(self):
        row = self._frame_list.currentRow()
        if row < 0:
            return
        self._frame_list.takeItem(row)
        self._frame_paths.pop(row)
        self._manual_affines = {
            k: v
            for k, v in self._manual_affines.items()
            if k[0] < len(self._frame_paths) and k[1] < len(self._frame_paths)
        }
        self._refresh_pair_combo()

    def _move_frame_up(self):
        row = self._frame_list.currentRow()
        if row <= 0:
            return
        self._swap_rows(row, row - 1)
        self._frame_list.setCurrentRow(row - 1)

    def _move_frame_down(self):
        row = self._frame_list.currentRow()
        if row < 0 or row >= self._frame_list.count() - 1:
            return
        self._swap_rows(row, row + 1)
        self._frame_list.setCurrentRow(row + 1)

    def _swap_rows(self, a: int, b: int):
        self._frame_paths[a], self._frame_paths[b] = (
            self._frame_paths[b],
            self._frame_paths[a],
        )
        item_a = self._frame_list.takeItem(a)
        item_b = self._frame_list.takeItem(b - 1)
        self._frame_list.insertItem(b - 1, item_a)
        self._frame_list.insertItem(a, item_b)
        self._manual_affines.clear()
        self._refresh_pair_combo()

    @Slot(object, object, int, int, int, int)
    def _on_rows_reordered(self, *_):
        self._frame_paths = [
            self._frame_list.item(r).data(Qt.ItemDataRole.UserRole)
            for r in range(self._frame_list.count())
        ]
        self._manual_affines.clear()
        self._refresh_pair_combo()

    @Slot(int)
    def _on_frame_selection_changed(self, row: int):
        if row < 0 or len(self._frame_paths) < 2:
            return
        j = min(row + 1, len(self._frame_paths) - 1)
        i = row if j > row else row - 1
        pair_text = f"Frame {i} → {j}"
        idx = self._pair_combo.findText(pair_text)
        if idx >= 0:
            self._pair_combo.setCurrentIndex(idx)

    def _update_frame_counter(self):
        """Update the frame counter label next to the video mode checkbox."""
        count = len(self._frame_paths) if hasattr(self, "_frame_paths") else 0
        if hasattr(self, "_lbl_frame_count"):
            self._lbl_frame_count.setText(f"Frames: {count}")

    def _refresh_pair_combo(self):
        self._update_frame_counter()
        self._pair_combo.blockSignals(True)
        self._pair_combo.clear()
        n = len(self._frame_paths)
        for i in range(n - 1):
            self._pair_combo.addItem(f"Frame {i} → {i + 1}", (i, i + 1))
        for i in range(n - 2):
            self._pair_combo.addItem(f"Frame {i} → {i + 2}  (skip)", (i, i + 2))
        self._pair_combo.blockSignals(False)
        if self._pair_combo.count():
            self._pair_combo.setCurrentIndex(0)
            self._on_pair_changed(0)

    @Slot(int)
    def _on_pair_changed(self, idx: int):
        if idx < 0 or idx >= self._pair_combo.count():
            return
        pair = self._pair_combo.itemData(idx)
        if pair is None:
            return
        self._current_pair = tuple(pair)
        i, j = self._current_pair
        if i >= len(self._frame_paths) or j >= len(self._frame_paths):
            return
        img_a = cv2.imread(self._frame_paths[i])
        img_b = cv2.imread(self._frame_paths[j])
        if img_a is None or img_b is None:
            return
        ha, wa = img_a.shape[:2]
        hb, wb = img_b.shape[:2]
        self._scene.load_pair(img_a, img_b, ha, wa, hb, wb)
        QTimer.singleShot(50, self._match_view.fit)

        if self._current_pair in self._manual_affines:
            self._affine_label.setText(
                f"Manual override active for pair {i}→{j}. Drag anchors to adjust."
            )
            self._affine_label.setStyleSheet(
                "color: #80CBC4; font-size: 10px; padding: 2px;"
            )
        else:
            self._affine_label.setText("No manual alignment override active.")
            self._affine_label.setStyleSheet(
                "color: #888; font-size: 10px; padding: 2px;"
            )

        self._match_count_label.setText("—")

    def _compute_matches(self):
        if len(self._frame_paths) < 2:
            QMessageBox.warning(self, "No frames", "Add at least 2 frames first.")
            return
        i, j = self._current_pair
        if i >= len(self._frame_paths) or j >= len(self._frame_paths):
            return
        if self._match_thread and self._match_thread.isRunning():
            return

        self._btn_compute.setEnabled(False)
        self._log_append(f"[LoFTR] Computing matches for pair {i}→{j}…")

        self._match_worker = MatchWorker(
            self._frame_paths[i],
            self._frame_paths[j],
            conf_thresh=self._conf_thresh_spin.value(),
            use_birefnet=self._cb_birefnet.isChecked(),
        )
        self._match_thread = self._match_worker
        self._match_worker.sig_finished.connect(self._on_matches_ready)
        self._match_worker.sig_error.connect(self._on_match_error)
        self._match_worker.finished.connect(lambda: self._btn_compute.setEnabled(True))
        self._match_worker.finished.connect(self._match_worker.deleteLater)
        self._match_worker.start()

    @Slot(object, object, object)
    def _on_matches_ready(self, pts1, pts2, conf):
        n = len(pts1)
        self._match_count_label.setText(
            f"{n} (conf ≥ {self._conf_thresh_spin.value():.2f})"
        )
        self._log_append(
            f"[LoFTR] {n} matches for pair {self._current_pair[0]}→{self._current_pair[1]}."
        )
        self._scene.show_matches(pts1, pts2, conf)
        QTimer.singleShot(50, self._match_view.fit)

    @Slot(str)
    def _on_match_error(self, msg: str):
        self._log_append(f"[LoFTR] Error: {msg}")

    def _show_mask(self):
        row = self._frame_list.currentRow()
        if row < 0 or row >= len(self._frame_paths):
            QMessageBox.warning(
                self, "No frame selected", "Select a frame in the list first."
            )
            return
        if self._mask_thread and self._mask_thread.isRunning():
            return

        self._btn_show_mask.setEnabled(False)
        self._log_append(f"[BiRefNet] Masking frame {row}…")

        self._mask_worker = MaskPreviewWorker(self._frame_paths[row])
        self._mask_thread = self._mask_worker
        self._mask_worker.sig_finished.connect(self._on_mask_ready)
        self._mask_worker.sig_error.connect(self._on_mask_error)
        self._mask_worker.finished.connect(lambda: self._btn_show_mask.setEnabled(True))
        self._mask_worker.finished.connect(self._mask_worker.deleteLater)
        self._mask_worker.start()

    @Slot(object)
    def _on_mask_ready(self, mask):
        row = self._frame_list.currentRow()
        if row < 0 or row >= len(self._frame_paths):
            return
        img = cv2.imread(self._frame_paths[row])
        if img is not None:
            self._scene.show_mask(img, mask)
        self._log_append("[BiRefNet] Mask overlay applied to left frame.")

    @Slot(str)
    def _on_mask_error(self, msg: str):
        self._log_append(f"[BiRefNet] Error: {msg}")

    @Slot(object)
    def _on_affine_updated(self, M):
        if M is None:
            return
        self._manual_affines[self._current_pair] = M.astype(np.float32)
        i, j = self._current_pair
        tx, ty = float(M[0, 2]), float(M[1, 2])
        scale = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
        angle_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        self._affine_label.setText(
            f"Manual override {i}→{j}:  tx={tx:.1f}  ty={ty:.1f}  "
            f"scale={scale:.3f}  θ={angle_deg:.2f}°"
        )
        self._affine_label.setStyleSheet(
            "color: #80CBC4; font-size: 10px; padding: 2px;"
        )

    def _reset_anchors(self):
        if self._current_pair in self._manual_affines:
            del self._manual_affines[self._current_pair]
        self._affine_label.setText("Manual override cleared — LoFTR will be used.")
        self._affine_label.setStyleSheet("color: #888; font-size: 10px; padding: 2px;")
        self._compute_matches()

    def _auto_order_sequence(self):
        """Reorder the stitch queue using the longest-coherent-path algorithm."""
        if not self._frame_paths:
            return

        ref_idx = self._frame_list.currentRow()
        if ref_idx < 0:
            ref_idx = 0
        ref_path = self._frame_paths[ref_idx]

        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            # We treat the currently loaded frames as the candidate pool
            new_order = AnimeStitchPipeline.find_optimal_sequence(
                ref_path, self._frame_paths, min_inliers=25
            )

            if not new_order:
                QMessageBox.warning(
                    self,
                    "Order Optimizer",
                    "No coherent matches found for the selected frame.",
                )
                return

            # Update state
            self._frame_paths = new_order

            # Refresh list
            self._frame_list.clear()
            for p in self._frame_paths:
                self._frame_list.addItem(os.path.basename(p))

            # Select the original reference in the new list
            try:
                new_ref_idx = self._frame_paths.index(ref_path)
                self._frame_list.setCurrentRow(new_ref_idx)
            except ValueError:
                pass

            QMessageBox.information(
                self,
                "Order Optimizer",
                f"Reordered {len(new_order)} coherent frames.\n"
                f"Sequence length optimized for continuity.",
            )

        except Exception as e:
            QMessageBox.critical(self, "Order Optimizer Error", str(e))
        finally:
            self.setCursor(Qt.CursorShape.ArrowCursor)


__all__ = ["_StitchFramesMixin"]
