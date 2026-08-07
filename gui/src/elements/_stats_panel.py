"""Statistics sub-tab: per-image / pairwise metrics UI and computation.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change.
"""

from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QSize, Qt, QThreadPool, Slot
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..helpers import StatsWorker
from ._thumb_workers import _ThumbTask

if TYPE_CHECKING:
    from ._stitch_tab_protocol import _StitchTabHost

    # Type-checking-only base: gives mypy visibility into attributes set by
    # StitchTab.__init__ / sibling mixins (see _stitch_tab_protocol.py).
    # Zero runtime effect -- at runtime this mixin still only inherits object.
    class _Base(_StitchTabHost, QWidget): ...
else:
    _Base = object


class _StatsPanelMixin(_Base):
    def _build_stats_panel(self) -> QWidget:
        from asp_gui.tabs.stencil import StatsPanel

        panel = StatsPanel(self)
        root = QVBoxLayout(panel)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Source selector ───────────────────────────────────────────
        src_group = QGroupBox("Image Source")
        src_layout = QHBoxLayout(src_group)

        self._stats_use_frames_btn = QPushButton("Use Stitch Frame List")
        self._stats_use_frames_btn.setToolTip(
            "Analyse the images currently loaded in the Stitch tab."
        )
        self._stats_use_frames_btn.clicked.connect(self._stats_load_from_frames)
        src_layout.addWidget(self._stats_use_frames_btn)

        src_layout.addWidget(QLabel("  or  "))

        self._stats_dir_edit = QLineEdit()
        self._stats_dir_edit.setPlaceholderText("Select a directory of images…")
        self._stats_dir_edit.setReadOnly(True)
        src_layout.addWidget(self._stats_dir_edit, 1)

        btn_browse_stats = QPushButton("Browse…")
        btn_browse_stats.clicked.connect(self._stats_browse_dir)
        src_layout.addWidget(btn_browse_stats)

        root.addWidget(src_group)

        # ── Options row ───────────────────────────────────────────────
        opts_row = QHBoxLayout()
        opts_row.addWidget(QLabel("K neighbors:"))
        self._stats_knn_spin = QSpinBox()
        self._stats_knn_spin.setRange(1, 100)
        self._stats_knn_spin.setValue(20)
        self._stats_knn_spin.setFixedWidth(60)
        self._stats_knn_spin.setToolTip(
            "When a consecutive pair scores below the weak threshold, also compare each "
            "frame against the K nearest frames ahead/behind it to find better matches.\n"
            "Higher values catch periodic pose repetitions further apart (e.g. every 20 frames) "
            "but increase compute time."
        )
        opts_row.addWidget(self._stats_knn_spin)
        opts_row.addStretch()
        root.addLayout(opts_row)

        # ── Run button + progress ─────────────────────────────────────
        run_row = QHBoxLayout()
        self._stats_run_btn = QPushButton("Compute Statistics")
        self._stats_run_btn.setStyleSheet(
            "background:#1976D2; color:white; font-weight:bold; padding:5px 14px;"
        )
        self._stats_run_btn.clicked.connect(self._stats_run)
        run_row.addWidget(self._stats_run_btn)

        self._stats_progress = QProgressBar()
        self._stats_progress.setRange(0, 100)
        self._stats_progress.setValue(0)
        self._stats_progress.setTextVisible(True)
        self._stats_progress.hide()
        run_row.addWidget(self._stats_progress, 1)

        self._stats_status = QLabel("")
        self._stats_status.setStyleSheet("color:#aaa; font-style:italic;")
        run_row.addWidget(self._stats_status)
        run_row.addStretch()
        root.addLayout(run_row)

        # ── Per-image table ───────────────────────────────────────────
        ind_group = QGroupBox("Per-Image Metrics")
        ind_layout = QVBoxLayout(ind_group)

        _IND_COLS = [
            ("Image", "name"),
            ("W", "width"),
            ("H", "height"),
            ("Aspect", "aspect_ratio"),
            ("Brightness", "brightness"),
            ("Contrast", "contrast"),
            ("Sharpness", "sharpness"),
            ("Noise", "noise"),
            ("Saturation", "saturation"),
            ("Dom. Hue °", "dominant_hue"),
            ("Size (KB)", "file_size_kb"),
        ]
        self._stats_ind_cols = _IND_COLS

        self._stats_ind_table = QTableWidget(0, len(_IND_COLS))
        self._stats_ind_table.setHorizontalHeaderLabels([c[0] for c in _IND_COLS])
        self._stats_ind_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for col in range(1, len(_IND_COLS)):
            self._stats_ind_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents
            )
        self._stats_ind_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._stats_ind_table.setAlternatingRowColors(True)
        self._stats_ind_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._stats_ind_table.verticalHeader().setVisible(False)
        self._stats_ind_table.verticalHeader().setDefaultSectionSize(52)
        self._stats_ind_table.setIconSize(QSize(48, 48))
        self._stats_ind_table.setMinimumHeight(220)
        self._stats_ind_table.setStyleSheet(
            "QTableWidget { background:#2c2f33; alternate-background-color:#36393f; }"
            "QHeaderView::section { background:#1e1f22; color:#ccc; padding:4px; }"
        )
        ind_layout.addWidget(self._stats_ind_table)

        # ── Summary row beneath individual table ──────────────────────
        self._stats_ind_summary = QLabel("")
        self._stats_ind_summary.setStyleSheet(
            "color:#aaa; font-size:10px; padding:2px 0;"
        )
        ind_layout.addWidget(self._stats_ind_summary)

        root.addWidget(ind_group)

        # ── Pairwise table ────────────────────────────────────────────
        pw_group = QGroupBox("Pairwise Correlation Metrics")
        pw_layout = QVBoxLayout(pw_group)

        _PW_COLS = [
            ("Frame A", "name_a"),
            ("Frame B", "name_b"),
            ("Hist. Corr.", "hist_corr"),
            ("SSIM", "ssim"),
            ("ORB Inliers", "orb_inliers"),
            ("Mean Diff", "mean_diff"),
            ("Stitch Score", "_score"),
        ]
        self._stats_pw_cols = _PW_COLS

        self._stats_pw_table = QTableWidget(0, len(_PW_COLS))
        self._stats_pw_table.setHorizontalHeaderLabels([c[0] for c in _PW_COLS])
        for col in range(2):
            self._stats_pw_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.Stretch
            )
        for col in range(2, len(_PW_COLS)):
            self._stats_pw_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents
            )
        self._stats_pw_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._stats_pw_table.setAlternatingRowColors(True)
        self._stats_pw_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._stats_pw_table.verticalHeader().setVisible(False)
        self._stats_pw_table.verticalHeader().setDefaultSectionSize(52)
        self._stats_pw_table.setIconSize(QSize(48, 48))
        self._stats_pw_table.setMinimumHeight(200)
        self._stats_pw_table.setStyleSheet(
            "QTableWidget { background:#2c2f33; alternate-background-color:#36393f; }"
            "QHeaderView::section { background:#1e1f22; color:#ccc; padding:4px; }"
        )
        pw_layout.addWidget(self._stats_pw_table)

        self._stats_pw_legend = QLabel(
            "Stitch Score = 0.4 × ORB inliers (norm.) + 0.4 × SSIM + 0.2 × Hist. Corr.  "
            "Higher is better.  Colour: green ≥ 0.6 · yellow ≥ 0.35 · red < 0.35"
        )
        self._stats_pw_legend.setStyleSheet(
            "color:#888; font-size:10px; padding:2px 0;"
        )
        self._stats_pw_legend.setWordWrap(True)
        pw_layout.addWidget(self._stats_pw_legend)

        root.addWidget(pw_group)

        # ── Recommendations ───────────────────────────────────────────
        rec_group = QGroupBox("Stitching Recommendations")
        rec_layout = QVBoxLayout(rec_group)

        self._stats_rec_edit = QTextEdit()
        self._stats_rec_edit.setReadOnly(True)
        self._stats_rec_edit.setMinimumHeight(240)
        self._stats_rec_edit.setPlaceholderText(
            "Run 'Compute Statistics' to generate scenario-based stitching recommendations."
        )
        self._stats_rec_edit.setStyleSheet(
            "QTextEdit { background:#1e1f22; color:#d4d4d4; "
            "border:1px solid #4f545c; border-radius:4px; "
            "font-family: monospace; font-size: 11px; padding: 6px; }"
        )
        rec_layout.addWidget(self._stats_rec_edit)

        root.addWidget(rec_group)

        return panel

    # ── Stats slots ────────────────────────────────────────────────────────

    @Slot()
    def _stats_load_from_frames(self):
        if not self._frame_paths:
            QMessageBox.information(
                self, "Statistics", "No frames loaded in the Stitch tab."
            )
            return
        self._stats_dir_edit.clear()
        self._stats_dir_path = ""
        self._stats_do_run(list(self._frame_paths))

    @Slot()
    def _stats_browse_dir(self):
        start = self._stats_dir_path or (
            os.path.dirname(self._frame_paths[-1]) if self._frame_paths else ""
        )
        d = QFileDialog.getExistingDirectory(
            self,
            "Select Image Directory",
            start,
            QFileDialog.Option.DontUseNativeDialog,
        )
        if not d:
            return
        self._stats_dir_path = d
        self._stats_dir_edit.setText(d)

    @Slot()
    def _stats_run(self):
        if self._stats_dir_path:
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
            paths = sorted(
                str(p)
                for p in pathlib.Path(self._stats_dir_path).iterdir()
                if p.is_file() and p.suffix.lower() in exts
            )
            if not paths:
                QMessageBox.information(
                    self,
                    "Statistics",
                    "No image files found in the selected directory.",
                )
                return
            self._stats_do_run(paths)
        else:
            self._stats_load_from_frames()

    def _stats_do_run(self, paths: list[str]):
        if not paths:
            return

        # Cancel any running worker
        if self._stats_worker is not None:
            self._stats_worker.cancel()
            self._stats_worker: StatsWorker | None = None

        self._stats_ind_rows: list[dict] = []
        self._stats_ind_table.setRowCount(0)
        self._stats_pw_table.setRowCount(0)
        self._stats_ind_summary.setText("")
        self._stats_rec_edit.clear()
        self._stats_progress.setValue(0)
        self._stats_progress.show()
        self._stats_status.setText(f"Analysing {len(paths)} images…")
        self._stats_run_btn.setEnabled(False)

        worker = StatsWorker(paths, knn_window=self._stats_knn_spin.value())
        self._stats_worker = worker
        worker.signals.progress.connect(self._stats_on_progress)
        worker.signals.individual_done.connect(self._stats_on_individual)
        worker.signals.pairwise_done.connect(self._stats_on_pairwise)
        worker.signals.error.connect(self._stats_on_error)
        QThreadPool.globalInstance().start(worker)

    @Slot(int, int)
    def _stats_on_progress(self, completed: int, total: int):
        self._stats_progress.setMaximum(max(total, 1))
        self._stats_progress.setValue(completed)

    @Slot(list)
    def _stats_on_individual(self, rows: list[dict]):
        self._stats_ind_rows = rows
        table = self._stats_ind_table
        table.setRowCount(len(rows))
        cols = self._stats_ind_cols

        self._stats_ind_item_map.clear()
        for r, row in enumerate(rows):
            for c, (_, key) in enumerate(cols):
                val = row.get(key, "")
                item = QTableWidgetItem(str(val) if val != -1 else "achromatic")
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    if c > 0
                    else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                )
                if c == 0:
                    full_path = row.get("path", "")
                    if full_path:
                        item.setData(Qt.ItemDataRole.UserRole, full_path)
                        self._stats_ind_item_map[full_path] = item
                        QThreadPool.globalInstance().start(
                            _ThumbTask(full_path, 48, 0, self._stats_ind_thumb_hub)
                        )

                # Colour-code sharpness column relatively (bottom 15% red,
                # top 40% green) so flat anime content isn't penalised.
                if key == "sharpness":
                    all_sharp = [float(rr.get("sharpness", 0)) for rr in rows]
                    p15 = float(np.percentile(all_sharp, 15))
                    p60 = float(np.percentile(all_sharp, 60))
                    v = float(val) if val else 0.0
                    if v >= p60:
                        item.setForeground(QColor("#4CAF50"))
                    elif v >= p15:
                        item.setForeground(QColor("#FFC107"))
                    else:
                        item.setForeground(QColor("#f44336"))

                # Colour-code noise column (index 7): lower is better
                if key == "noise":
                    v = float(val) if val else 0.0
                    if v <= 5:
                        item.setForeground(QColor("#4CAF50"))
                    elif v <= 15:
                        item.setForeground(QColor("#FFC107"))
                    else:
                        item.setForeground(QColor("#f44336"))

                table.setItem(r, c, item)

        # Summary line
        if rows:
            avg_sharp = np.mean([r.get("sharpness", 0) for r in rows])
            avg_bright = np.mean([r.get("brightness", 0) for r in rows])
            avg_contrast = np.mean([r.get("contrast", 0) for r in rows])
            resolutions = set(
                f"{r['width']}×{r['height']}" for r in rows if r.get("width")
            )
            res_str = (
                ", ".join(sorted(resolutions))
                if len(resolutions) <= 3
                else f"{len(resolutions)} different"
            )
            self._stats_ind_summary.setText(
                f"Count: {len(rows)}  |  Resolutions: {res_str}  |  "
                f"Avg brightness: {avg_bright:.1f}  |  "
                f"Avg contrast: {avg_contrast:.1f}  |  "
                f"Avg sharpness: {avg_sharp:.1f}"
            )

    @Slot(list)
    def _stats_on_pairwise(self, rows: list[dict]):
        table = self._stats_pw_table
        table.setRowCount(len(rows))
        cols = self._stats_pw_cols

        # Normalise ORB inliers across all rows for score
        max_orb = max((r.get("orb_inliers", 0) for r in rows), default=1) or 1

        self._stats_pw_item_map_a.clear()
        self._stats_pw_item_map_b.clear()
        for r, row in enumerate(rows):
            orb_norm = row.get("orb_inliers", 0) / max_orb
            ssim_val = max(0.0, float(row.get("ssim", 0)))
            hist_val = max(0.0, float(row.get("hist_corr", 0)))
            score = round(0.4 * orb_norm + 0.4 * ssim_val + 0.2 * hist_val, 3)
            row["_score"] = score

            for c, (_, key) in enumerate(cols):
                val = row.get(key, "")
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    if c >= 2
                    else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                )
                if c == 0:
                    full_path = row.get("path_a", "")
                    if full_path:
                        item.setData(Qt.ItemDataRole.UserRole, full_path)
                        self._stats_pw_item_map_a[full_path] = item
                        QThreadPool.globalInstance().start(
                            _ThumbTask(full_path, 48, 0, self._stats_pw_thumb_hub_a)
                        )
                elif c == 1:
                    full_path = row.get("path_b", "")
                    if full_path:
                        item.setData(Qt.ItemDataRole.UserRole, full_path)
                        self._stats_pw_item_map_b[full_path] = item
                        QThreadPool.globalInstance().start(
                            _ThumbTask(full_path, 48, 0, self._stats_pw_thumb_hub_b)
                        )

                # Colour-code the Stitch Score column
                if key == "_score":
                    if score >= 0.6:
                        item.setForeground(QColor("#4CAF50"))
                        item.setBackground(QColor("#1b3a1f"))
                    elif score >= 0.35:
                        item.setForeground(QColor("#FFC107"))
                        item.setBackground(QColor("#3a3000"))
                    else:
                        item.setForeground(QColor("#f44336"))
                        item.setBackground(QColor("#3a1010"))

                table.setItem(r, c, item)

        self._stats_progress.hide()
        self._stats_status.setText(f"Done. {len(rows)} pair(s) analysed.")
        self._stats_run_btn.setEnabled(True)
        self._stats_worker = None

        ind_rows = getattr(self, "_stats_ind_rows", [])
        self._stats_rec_edit.setHtml(
            self._stats_build_recommendations(ind_rows, rows, max_orb)
        )

    @Slot(str, int, object)
    def _on_stats_ind_thumb_loaded(self, path: str, _generation: int, img: QImage):
        item = self._stats_ind_item_map.get(path)
        if item and not img.isNull():
            item.setIcon(QIcon(QPixmap.fromImage(img)))

    @Slot(str, int, object)
    def _on_stats_pw_thumb_loaded_a(self, path: str, _generation: int, img: QImage):
        item = self._stats_pw_item_map_a.get(path)
        if item and not img.isNull():
            item.setIcon(QIcon(QPixmap.fromImage(img)))

    @Slot(str, int, object)
    def _on_stats_pw_thumb_loaded_b(self, path: str, _generation: int, img: QImage):
        item = self._stats_pw_item_map_b.get(path)
        if item and not img.isNull():
            item.setIcon(QIcon(QPixmap.fromImage(img)))


__all__ = ["_StatsPanelMixin"]
