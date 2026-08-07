"""Sequence Builder sub-tab: candidate search, chain editing, and acceptance.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change.
"""

from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QThreadPool, Slot
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem, QWidget

from ..helpers import SequenceBuilderWorker
from ._thumb_workers import _ThumbTask

if TYPE_CHECKING:
    from ._stitch_tab_protocol import _StitchTabHost

    # Type-checking-only base: gives mypy visibility into attributes set by
    # StitchTab.__init__ / sibling mixins (see _stitch_tab_protocol.py).
    # Zero runtime effect -- at runtime this mixin still only inherits object.
    class _Base(_StitchTabHost, QWidget): ...
else:
    _Base = object


class _SeqPanelHandlersMixin(_Base):
    @Slot()
    def _seq_browse_anchor(self):
        start = self._seq_dir_path or os.path.dirname(self._seq_anchor_path) or ""
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select Anchor Image",
            start,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if p:
            self._seq_anchor_path = p
            self._seq_anchor_edit.setText(p)
            # Auto-populate dir from anchor location if not already set
            if not self._seq_dir_path:
                self._seq_dir_path = os.path.dirname(p)
                self._seq_dir_edit.setText(self._seq_dir_path)

    @Slot()
    def _seq_browse_dir(self):
        start = self._seq_dir_path or os.path.dirname(self._seq_anchor_path) or ""
        d = QFileDialog.getExistingDirectory(
            self,
            "Select Candidate Directory",
            start,
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if d:
            self._seq_dir_path = d
            self._seq_dir_edit.setText(d)

    @Slot()
    def _seq_load_from_stitch(self):
        if not self._frame_paths:
            QMessageBox.information(
                self, "Sequence Builder", "The Stitch tab frame list is empty."
            )
            return
        # Use selected frame as anchor, rest as candidates
        row = self._frame_list.currentRow()
        anchor = self._frame_paths[row] if row >= 0 else self._frame_paths[0]
        self._seq_anchor_path = anchor
        self._seq_anchor_edit.setText(anchor)
        d = os.path.dirname(anchor)
        self._seq_dir_path = d
        self._seq_dir_edit.setText(d)
        QMessageBox.information(
            self,
            "Sequence Builder",
            f"Anchor set to: {os.path.basename(anchor)}\nCandidates directory: {d}",
        )

    @Slot()
    def _seq_run(self):
        if not self._seq_anchor_path or not os.path.isfile(self._seq_anchor_path):
            QMessageBox.warning(
                self, "Sequence Builder", "Please select a valid anchor image first."
            )
            return

        # Collect candidate paths from the directory
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
        if self._seq_dir_path and os.path.isdir(self._seq_dir_path):
            candidates = sorted(
                str(p)
                for p in pathlib.Path(self._seq_dir_path).iterdir()
                if p.suffix.lower() in exts
            )
        else:
            QMessageBox.warning(
                self, "Sequence Builder", "Please select a candidate directory first."
            )
            return

        if not candidates:
            QMessageBox.warning(
                self,
                "Sequence Builder",
                "No image files found in the selected directory.",
            )
            return

        if self._seq_worker is not None:
            self._seq_worker.cancel()
            self._seq_worker = None

        self._seq_chain_table.setRowCount(0)
        self._seq_chain = []
        self._seq_progress.setValue(0)
        self._seq_progress.show()
        self._seq_run_btn.setEnabled(False)
        n_cand = len(candidates)
        self._seq_status.setText(
            f"Searching {n_cand} candidates for anchor: {os.path.basename(self._seq_anchor_path)}…"
        )

        worker = SequenceBuilderWorker(
            self._seq_anchor_path,
            candidates,
            min_score=self._seq_min_score_spin.value(),
            blur_threshold=self._seq_blur_spin.value(),
            min_pan_ratio=self._seq_min_pan_spin.value(),
            max_pan_ratio=self._seq_max_pan_spin.value(),
        )
        self._seq_worker = worker
        worker.signals.progress.connect(self._seq_on_progress)
        worker.signals.result.connect(self._seq_on_result)
        worker.signals.error.connect(self._seq_on_error)
        QThreadPool.globalInstance().start(worker)

    @Slot(int, int)
    def _seq_on_progress(self, completed: int, total: int):
        self._seq_progress.setMaximum(max(total, 1))
        self._seq_progress.setValue(completed)

    @Slot(list)
    def _seq_on_result(self, chain: list[dict]):
        self._seq_progress.hide()
        self._seq_run_btn.setEnabled(True)
        self._seq_worker = None
        self._seq_chain = chain
        self._seq_status.setText(f"Done. {len(chain)} frame(s) in sequence.")
        self._seq_populate_table(chain)

    def _seq_populate_table(self, chain: list[dict]):
        table = self._seq_chain_table
        table.setRowCount(0)
        self._seq_table_item_map.clear()
        for item in chain:
            r = table.rowCount()
            table.insertRow(r)
            name_item = QTableWidgetItem(item["name"])
            name_item.setData(Qt.ItemDataRole.UserRole, item["path"])
            name_item.setToolTip(item["path"])
            self._seq_table_item_map[item["path"]] = name_item
            QThreadPool.globalInstance().start(
                _ThumbTask(item["path"], 48, 0, self._seq_thumb_hub)
            )
            table.setItem(r, 0, name_item)

            score = item.get("score_to_prev")
            if score is None:
                score_item = QTableWidgetItem("— anchor —")
                score_item.setForeground(QColor("#aaa"))
            else:
                score_item = QTableWidgetItem(f"{score:.3f}")
                if score >= 0.6:
                    score_item.setForeground(QColor("#4CAF50"))
                    score_item.setBackground(QColor("#1b3a1f"))
                elif score >= 0.35:
                    score_item.setForeground(QColor("#FFC107"))
                    score_item.setBackground(QColor("#3a3000"))
                else:
                    score_item.setForeground(QColor("#f44336"))
                    score_item.setBackground(QColor("#3a1010"))
            score_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            table.setItem(r, 1, score_item)

            # "Replace" button cell
            replace_item = QTableWidgetItem("Replace…")
            replace_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            replace_item.setForeground(QColor("#90CAF9"))
            table.setItem(r, 2, replace_item)

    def _seq_chain_from_table(self) -> list[dict]:
        """Read the current table contents back into a chain list."""
        table = self._seq_chain_table
        chain = []
        for r in range(table.rowCount()):
            name_item = table.item(r, 0)
            score_item = table.item(r, 1)
            p = name_item.data(Qt.ItemDataRole.UserRole) if name_item else ""
            score_txt = score_item.text() if score_item else ""
            try:
                s = None if "anchor" in score_txt else float(score_txt)
            except ValueError:
                s = None
            chain.append({"path": p, "name": os.path.basename(p), "score_to_prev": s})
        return chain

    @Slot(int, int)
    def _seq_replace_row(self, row: int, col: int):
        """Double-click on any cell: open file picker to replace that row's image."""
        table = self._seq_chain_table
        name_item = table.item(row, 0)
        if name_item is None:
            return
        current_path = name_item.data(Qt.ItemDataRole.UserRole) or ""
        start = os.path.dirname(current_path) or self._seq_dir_path or ""
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Replace Image",
            start,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not p:
            return
        name_item.setText(os.path.basename(p))
        name_item.setData(Qt.ItemDataRole.UserRole, p)
        name_item.setToolTip(p)
        # Clear score (unknown after manual replacement)
        score_item = table.item(row, 1)
        if score_item:
            score_item.setText("(replaced)")
            score_item.setForeground(QColor("#aaa"))

    @Slot()
    def _seq_insert_image(self, before: bool = True):
        table = self._seq_chain_table
        row = table.currentRow()
        if row < 0:
            row = table.rowCount() if not before else 0
        start = self._seq_dir_path or ""
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Insert Image",
            start,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not p:
            return
        insert_at = row if before else row + 1
        table.insertRow(insert_at)
        name_item = QTableWidgetItem(os.path.basename(p))
        name_item.setData(Qt.ItemDataRole.UserRole, p)
        name_item.setToolTip(p)
        self._seq_table_item_map[p] = name_item
        QThreadPool.globalInstance().start(_ThumbTask(p, 48, 0, self._seq_thumb_hub))
        table.setItem(insert_at, 0, name_item)
        score_item = QTableWidgetItem("(inserted)")
        score_item.setForeground(QColor("#aaa"))
        score_item.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        table.setItem(insert_at, 1, score_item)
        replace_item = QTableWidgetItem("Replace…")
        replace_item.setTextAlignment(
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
        )
        replace_item.setForeground(QColor("#90CAF9"))
        table.setItem(insert_at, 2, replace_item)
        table.setCurrentCell(insert_at, 0)

    @Slot()
    def _seq_remove_row(self):
        row = self._seq_chain_table.currentRow()
        if row >= 0:
            self._seq_chain_table.removeRow(row)

    @Slot()
    def _seq_move_up(self):
        table = self._seq_chain_table
        row = table.currentRow()
        if row <= 0:
            return
        self._seq_swap_rows(row - 1, row)
        table.setCurrentCell(row - 1, 0)

    @Slot()
    def _seq_move_down(self):
        table = self._seq_chain_table
        row = table.currentRow()
        if row < 0 or row >= table.rowCount() - 1:
            return
        self._seq_swap_rows(row, row + 1)
        table.setCurrentCell(row + 1, 0)

    def _seq_swap_rows(self, a: int, b: int):
        table = self._seq_chain_table
        for col in range(table.columnCount()):
            ia = table.takeItem(a, col)
            ib = table.takeItem(b, col)
            if ia:
                table.setItem(b, col, ia)
            if ib:
                table.setItem(a, col, ib)

    @Slot(object, int, int, object, int)
    def _seq_on_rows_moved(self, *_args):
        """QAbstractItemModel.rowsMoved — no extra action needed; table updates itself."""
        pass

    @Slot()
    def _seq_accept(self):
        """Push the current chain table into the Stitch tab frame list."""
        table = self._seq_chain_table
        if table.rowCount() == 0:
            QMessageBox.information(
                self, "Sequence Builder", "The sequence is empty — nothing to load."
            )
            return

        paths = []
        for r in range(table.rowCount()):
            item = table.item(r, 0)
            if item:
                p = item.data(Qt.ItemDataRole.UserRole)
                if p and os.path.isfile(p):
                    paths.append(p)

        if not paths:
            QMessageBox.warning(
                self,
                "Sequence Builder",
                "No valid file paths found in the current sequence.",
            )
            return

        self._frame_paths = paths
        self._frame_item_map.clear()
        self._frame_list.clear()
        for p in self._frame_paths:
            self._frame_list.addItem(self._make_frame_item(p))
        self._refresh_pair_combo()

        self._tab_widget.setCurrentIndex(0)  # switch to Stitch tab
        QMessageBox.information(
            self,
            "Sequence Builder",
            f"Loaded {len(paths)} frame(s) into the Stitch tab.\n"
            "Switch to the Stitch tab to run the pipeline.",
        )

    @Slot(str)
    def _seq_on_error(self, msg: str):
        self._seq_progress.hide()
        self._seq_run_btn.setEnabled(True)
        self._seq_worker = None
        self._seq_status.setText("Error.")
        QMessageBox.critical(self, "Sequence Builder Error", msg)

    @Slot(str, int, object)
    def _on_seq_table_thumb_loaded(self, path: str, _generation: int, img: QImage):
        item = self._seq_table_item_map.get(path)
        if item and not img.isNull():
            item.setIcon(QIcon(QPixmap.fromImage(img)))


__all__ = ["_SeqPanelHandlersMixin"]
