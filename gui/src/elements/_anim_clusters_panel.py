"""Animation Clusters sub-tab: animation-phase detection UI and logic.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change.
"""

from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING

from gui.src.constants import ANIM_CLUSTER_COLORS
from PySide6.QtCore import QSize, Qt, QThreadPool, Slot
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QDoubleSpinBox,
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
    QVBoxLayout,
    QWidget,
)

from ..helpers import AnimClusterWorker
from ._thumb_workers import _ThumbTask

if TYPE_CHECKING:
    from ._stitch_tab_protocol import _StitchTabHost

    # Type-checking-only base: gives mypy visibility into attributes set by
    # StitchTab.__init__ / sibling mixins (see _stitch_tab_protocol.py).
    # Zero runtime effect -- at runtime this mixin still only inherits object.
    class _Base(_StitchTabHost, QWidget): ...
else:
    _Base = object


class _AnimClustersPanelMixin(_Base):
    def _build_anim_clusters_panel(self) -> QWidget:
        from asp_gui.tabs.stencil import AnimClustersPanel

        panel = AnimClustersPanel(self)
        root = QVBoxLayout(panel)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Source selector ───────────────────────────────────────────
        src_group = QGroupBox("Image Source")
        src_layout = QHBoxLayout(src_group)

        self._anim_use_frames_btn = QPushButton("Use Stitch Frame List")
        self._anim_use_frames_btn.setToolTip(
            "Analyse the images currently loaded in the Stitch tab."
        )
        self._anim_use_frames_btn.clicked.connect(self._anim_load_from_frames)
        src_layout.addWidget(self._anim_use_frames_btn)

        src_layout.addWidget(QLabel("  or  "))

        self._anim_files_edit = QLineEdit()
        self._anim_files_edit.setPlaceholderText(
            "Select individual files or a directory…"
        )
        self._anim_files_edit.setReadOnly(True)
        src_layout.addWidget(self._anim_files_edit, 1)

        btn_browse_files = QPushButton("Files…")
        btn_browse_files.clicked.connect(self._anim_browse_files)
        src_layout.addWidget(btn_browse_files)

        btn_browse_dir = QPushButton("Dir…")
        btn_browse_dir.clicked.connect(self._anim_browse_dir)
        src_layout.addWidget(btn_browse_dir)

        root.addWidget(src_group)

        # ── Options + run ─────────────────────────────────────────────
        opt_row = QHBoxLayout()

        opt_row.addWidget(QLabel("AC Threshold:"))
        self._anim_threshold_spin = QDoubleSpinBox()
        self._anim_threshold_spin.setRange(0.01, 1.0)
        self._anim_threshold_spin.setSingleStep(0.05)
        self._anim_threshold_spin.setValue(0.25)
        self._anim_threshold_spin.setDecimals(2)
        self._anim_threshold_spin.setToolTip(
            "Fraction of signal power in AC frequencies required to label a pixel as animated.\n"
            "Lower → more sensitive to subtle motion."
        )
        opt_row.addWidget(self._anim_threshold_spin)

        opt_row.addSpacing(8)
        opt_row.addWidget(QLabel("Min anime px:"))
        self._anim_min_px_spin = QSpinBox()
        self._anim_min_px_spin.setRange(10, 50000)
        self._anim_min_px_spin.setSingleStep(100)
        self._anim_min_px_spin.setValue(500)
        self._anim_min_px_spin.setToolTip(
            "Minimum number of animated pixels (at 320-px scale) required to attempt clustering.\n"
            "Increase to suppress noise detections."
        )
        opt_row.addWidget(self._anim_min_px_spin)

        opt_row.addStretch()

        self._anim_run_btn = QPushButton("Detect Phases")
        self._anim_run_btn.setStyleSheet(
            "background:#1976D2; color:white; font-weight:bold; padding:5px 14px;"
        )
        self._anim_run_btn.clicked.connect(lambda: self._anim_do_run())
        opt_row.addWidget(self._anim_run_btn)

        self._anim_progress = QProgressBar()
        self._anim_progress.setRange(0, 100)
        self._anim_progress.setValue(0)
        self._anim_progress.setTextVisible(True)
        self._anim_progress.hide()
        opt_row.addWidget(self._anim_progress, 1)

        root.addLayout(opt_row)

        self._anim_status = QLabel("")
        self._anim_status.setStyleSheet("color:#aaa; font-style:italic;")
        root.addWidget(self._anim_status)

        # ── Result table ──────────────────────────────────────────────
        result_group = QGroupBox("Animation Phases")
        result_layout = QVBoxLayout(result_group)

        self._anim_table = QTableWidget(0, 3)
        self._anim_table.setHorizontalHeaderLabels(["Image", "Filename", "Phase"])
        self._anim_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self._anim_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._anim_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self._anim_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._anim_table.setAlternatingRowColors(False)
        self._anim_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._anim_table.verticalHeader().setVisible(False)
        self._anim_table.verticalHeader().setDefaultSectionSize(52)
        self._anim_table.setIconSize(QSize(48, 48))
        self._anim_table.setMinimumHeight(300)
        self._anim_table.setStyleSheet(
            "QTableWidget { background:#2c2f33; }"
            "QHeaderView::section { background:#1e1f22; color:#ccc; padding:4px; }"
        )
        result_layout.addWidget(self._anim_table)

        self._anim_legend = QLabel(
            "Rows are colour-coded by phase.  "
            "Requires scikit-learn — install with: pip install scikit-learn"
        )
        self._anim_legend.setStyleSheet("color:#888; font-size:10px; padding:2px 0;")
        self._anim_legend.setWordWrap(True)
        result_layout.addWidget(self._anim_legend)

        root.addWidget(result_group)
        return panel

    # ── Animation Clusters slots ────────────────────────────────────────────────

    @Slot()
    def _anim_load_from_frames(self):
        if not self._frame_paths:
            QMessageBox.information(
                self, "Animation Phases", "No frames loaded in the Stitch tab."
            )
            return
        self._anim_files_edit.clear()
        self._anim_cluster_paths = []
        self._anim_cluster_dir_path = ""
        self._anim_do_run(sorted(self._frame_paths))

    @Slot()
    def _anim_browse_files(self):
        start = self._anim_cluster_dir_path or (
            os.path.dirname(self._frame_paths[-1]) if self._frame_paths else ""
        )
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            start,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not paths:
            return
        self._anim_cluster_paths = sorted(paths)
        self._anim_cluster_dir_path = ""
        self._anim_files_edit.setText(f"{len(paths)} file(s) selected")

    @Slot()
    def _anim_browse_dir(self):
        start = self._anim_cluster_dir_path or (
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
        self._anim_cluster_dir_path = d
        self._anim_cluster_paths = []
        self._anim_files_edit.setText(d)

    @Slot()
    def _anim_do_run(self, paths: list[str] | None = None):
        if paths is None:
            if self._anim_cluster_dir_path:
                exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
                paths = sorted(
                    str(p)
                    for p in pathlib.Path(self._anim_cluster_dir_path).iterdir()
                    if p.is_file() and p.suffix.lower() in exts
                )
                if not paths:
                    QMessageBox.information(
                        self,
                        "Animation Phases",
                        "No image files found in the selected directory.",
                    )
                    return
            elif self._anim_cluster_paths:
                paths = self._anim_cluster_paths
            else:
                QMessageBox.information(
                    self,
                    "Animation Phases",
                    "Select images using 'Use Stitch Frame List', 'Files…', or 'Dir…'.",
                )
                return

        if not paths:
            return

        if self._anim_cluster_worker is not None:
            self._anim_cluster_worker.cancel()
            self._anim_cluster_worker: AnimClusterWorker | None = None

        self._anim_table.setRowCount(0)
        self._anim_item_map.clear()
        self._anim_progress.setValue(0)
        self._anim_progress.show()
        self._anim_status.setText(f"Analysing {len(paths)} image(s)…")
        self._anim_run_btn.setEnabled(False)

        worker = AnimClusterWorker(
            paths,
            ac_threshold=self._anim_threshold_spin.value(),
            min_anim_pixels=self._anim_min_px_spin.value(),
        )
        self._anim_cluster_worker = worker
        worker.signals.progress.connect(self._anim_on_progress)
        worker.signals.finished.connect(self._anim_on_finished)
        worker.signals.error.connect(self._anim_on_error)
        QThreadPool.globalInstance().start(worker)

    @Slot(int, int)
    def _anim_on_progress(self, completed: int, total: int):
        self._anim_progress.setMaximum(max(total, 1))
        self._anim_progress.setValue(completed)

    @Slot(list)
    def _anim_on_finished(self, rows: list[dict]):
        self._anim_progress.hide()
        self._anim_cluster_worker = None
        self._anim_run_btn.setEnabled(True)

        n = len(rows)
        if n == 0:
            self._anim_status.setText("No images processed.")
            return

        is_anim = any(r.get("is_animated", False) for r in rows)
        if not is_anim:
            reason = rows[0].get("cluster_name", "Static")
            self._anim_status.setText(f"Result: {reason}")
        else:
            n_phases = len({r["cluster"] for r in rows})
            self._anim_status.setText(
                f"Done. {n} image(s) assigned to {n_phases} animation phase(s)."
            )

        table = self._anim_table
        table.setRowCount(n)
        self._anim_item_map.clear()

        for r, row in enumerate(rows):
            cluster = row.get("cluster", 0)
            is_animated = row.get("is_animated", False)
            if is_animated:
                fg, bg = ANIM_CLUSTER_COLORS[cluster % len(ANIM_CLUSTER_COLORS)]
            else:
                fg, bg = "#aaaaaa", "#2c2f33"

            # Col 0: thumbnail placeholder
            thumb_item = QTableWidgetItem("")
            thumb_item.setData(Qt.ItemDataRole.UserRole, row["path"])
            thumb_item.setBackground(QColor(bg))
            table.setItem(r, 0, thumb_item)
            self._anim_item_map[row["path"]] = thumb_item
            QThreadPool.globalInstance().start(
                _ThumbTask(row["path"], 48, 0, self._anim_thumb_hub)
            )

            # Col 1: filename
            fname_item = QTableWidgetItem(os.path.basename(row["path"]))
            fname_item.setForeground(QColor(fg))
            fname_item.setBackground(QColor(bg))
            fname_item.setToolTip(row["path"])
            fname_item.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            table.setItem(r, 1, fname_item)

            # Col 2: phase name
            phase_item = QTableWidgetItem(row.get("cluster_name", ""))
            phase_item.setForeground(QColor(fg))
            phase_item.setBackground(QColor(bg))
            phase_item.setTextAlignment(
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
            )
            table.setItem(r, 2, phase_item)

    @Slot(str)
    def _anim_on_error(self, msg: str):
        self._anim_progress.hide()
        self._anim_status.setText("Error.")
        self._anim_run_btn.setEnabled(True)
        self._anim_cluster_worker = None
        QMessageBox.critical(self, "Animation Phase Detection Error", msg)

    @Slot(str, int, object)
    def _on_anim_thumb_loaded(self, path: str, _generation: int, img: QImage):
        item = self._anim_item_map.get(path)
        if item and not img.isNull():
            item.setIcon(QIcon(QPixmap.fromImage(img)))


__all__ = ["_AnimClustersPanelMixin"]
