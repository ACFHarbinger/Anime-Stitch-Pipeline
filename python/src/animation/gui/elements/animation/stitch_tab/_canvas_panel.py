"""Canvas sub-tab: manual layout composer UI and logic.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QSize, Qt, QThreadPool, Slot
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ....constants import SIZE_PRESETS
from ....helpers.animation import CanvasWorker
from ....styles import apply_shadow_effect
from ....windows.settings.splitter_persistence import persist_splitter
from ._thumb_workers import _ThumbTask


class _CanvasPanelMixin:
    def _build_canvas_panel(self) -> QWidget:
        from gui.src.tabs.animation.stencil import CanvasPanel

        panel = CanvasPanel(self)
        root_layout = QVBoxLayout(panel)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        # ── Top bar: output size ───────────────────────────────────────
        size_group = QGroupBox("Output Size")
        size_layout = QHBoxLayout(size_group)
        size_layout.setSpacing(6)

        self._cv_preset_combo = QComboBox()
        for label, _ in SIZE_PRESETS:
            self._cv_preset_combo.addItem(label)
        self._cv_preset_combo.currentIndexChanged.connect(self._cv_on_preset_changed)
        size_layout.addWidget(QLabel("Preset:"))
        size_layout.addWidget(self._cv_preset_combo, stretch=1)

        size_layout.addWidget(QLabel("W:"))
        self._cv_width_spin = QSpinBox()
        self._cv_width_spin.setRange(64, 16384)
        self._cv_width_spin.setValue(1920)
        self._cv_width_spin.setSuffix(" px")
        self._cv_width_spin.setFixedWidth(100)
        size_layout.addWidget(self._cv_width_spin)

        size_layout.addWidget(QLabel("H:"))
        self._cv_height_spin = QSpinBox()
        self._cv_height_spin.setRange(64, 16384)
        self._cv_height_spin.setValue(1080)
        self._cv_height_spin.setSuffix(" px")
        self._cv_height_spin.setFixedWidth(100)
        size_layout.addWidget(self._cv_height_spin)

        root_layout.addWidget(size_group)

        # ── Main area: image list │ preview ───────────────────────────
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # Left: image list + controls
        left = QWidget()
        left.setFixedWidth(260)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)

        img_group = QGroupBox("Images")
        img_group_layout = QVBoxLayout(img_group)

        self._cv_list = QListWidget()
        self._cv_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._cv_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._cv_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._cv_list.setToolTip(
            "Drag to reorder. Images are placed left-to-right / top-to-bottom."
        )
        self._cv_list.setIconSize(QSize(48, 48))
        self._cv_list.model().rowsMoved.connect(self._cv_sync_paths)
        img_group_layout.addWidget(self._cv_list)

        cv_btn_grid = QGridLayout()
        cv_btn_grid.setSpacing(4)
        btn_cv_add = QPushButton("Add")
        btn_cv_add.clicked.connect(self._cv_add_images)
        apply_shadow_effect(btn_cv_add, radius=4, y_offset=2)
        btn_cv_remove = QPushButton("Remove")
        btn_cv_remove.clicked.connect(self._cv_remove_selected)
        apply_shadow_effect(btn_cv_remove, radius=4, y_offset=2)
        btn_cv_clear = QPushButton("Clear All")
        btn_cv_clear.clicked.connect(self._cv_clear_all)
        apply_shadow_effect(btn_cv_clear, radius=4, y_offset=2)
        cv_btn_grid.addWidget(btn_cv_add, 0, 0)
        cv_btn_grid.addWidget(btn_cv_remove, 0, 1)
        cv_btn_grid.addWidget(btn_cv_clear, 1, 0, 1, 2)
        img_group_layout.addLayout(cv_btn_grid)
        left_layout.addWidget(img_group)

        # Layout options
        layout_group = QGroupBox("Layout")
        layout_v = QVBoxLayout(layout_group)
        layout_v.setSpacing(4)

        self._cv_layout_bg = QButtonGroup(self)
        self._cv_radio_h = QRadioButton("Horizontal")
        self._cv_radio_v = QRadioButton("Vertical")
        self._cv_radio_g = QRadioButton("Grid")
        self._cv_radio_h.setChecked(True)
        for rb in (self._cv_radio_h, self._cv_radio_v, self._cv_radio_g):
            self._cv_layout_bg.addButton(rb)
            layout_v.addWidget(rb)

        grid_cols_row = QHBoxLayout()
        grid_cols_row.addWidget(QLabel("Columns:"))
        self._cv_cols_spin = QSpinBox()
        self._cv_cols_spin.setRange(1, 20)
        self._cv_cols_spin.setValue(2)
        self._cv_cols_spin.setFixedWidth(70)
        grid_cols_row.addWidget(self._cv_cols_spin)
        grid_cols_row.addStretch()
        layout_v.addLayout(grid_cols_row)

        left_layout.addWidget(layout_group)

        # Style options
        style_group = QGroupBox("Style")
        style_form = QFormLayout(style_group)
        style_form.setSpacing(4)

        gap_row = QHBoxLayout()
        self._cv_gap_spin = QSpinBox()
        self._cv_gap_spin.setRange(0, 200)
        self._cv_gap_spin.setValue(0)
        self._cv_gap_spin.setSuffix(" px")
        self._cv_gap_spin.setFixedWidth(80)
        gap_row.addWidget(self._cv_gap_spin)
        gap_row.addStretch()
        style_form.addRow("Gap:", gap_row)

        self._cv_scale_combo = QComboBox()
        self._cv_scale_combo.addItem("Fit (letterbox)", "fit")
        self._cv_scale_combo.addItem("Fill (center crop)", "fill")
        self._cv_scale_combo.addItem("Stretch", "stretch")
        self._cv_scale_combo.setToolTip(
            "Fit: preserve aspect ratio, add background bars.\n"
            "Fill: fill the cell, crop the excess.\n"
            "Stretch: deform to fill."
        )
        style_form.addRow("Scale:", self._cv_scale_combo)

        bg_row = QHBoxLayout()
        self._cv_bg_btn = QPushButton("  ")
        self._cv_bg_btn.setFixedWidth(40)
        self._cv_bg_btn.setToolTip("Background colour.")
        self._cv_bg_btn.clicked.connect(self._cv_pick_bg_color)
        self._cv_bg_label = QLabel("#000000")
        self._cv_bg_label.setStyleSheet("color: #aaa; font-size: 10px;")
        self._cv_update_bg_button()
        bg_row.addWidget(self._cv_bg_btn)
        bg_row.addWidget(self._cv_bg_label)
        bg_row.addStretch()
        style_form.addRow("Background:", bg_row)

        left_layout.addWidget(style_group)
        left_layout.addStretch()

        main_splitter.addWidget(left)

        # Right: preview
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(4)

        self._cv_preview_label = QLabel("Press 'Preview' to render the canvas.")
        self._cv_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cv_preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._cv_preview_label.setStyleSheet(
            "background: #181818; border: 1px solid #3a3a3a; color: #666;"
        )
        preview_layout.addWidget(self._cv_preview_label)

        self._cv_status_label = QLabel("")
        self._cv_status_label.setStyleSheet("color: #777; font-size: 10px;")
        self._cv_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self._cv_status_label)

        main_splitter.addWidget(preview_widget)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        persist_splitter(main_splitter, "CanvasPanel/main")
        root_layout.addWidget(main_splitter, stretch=1)

        # ── Bottom: action bar ─────────────────────────────────────────
        cv_action_bar = QHBoxLayout()
        self._btn_cv_preview = QPushButton("▶  Preview")
        self._btn_cv_preview.setStyleSheet(
            "background:#1976D2; color:white; font-weight:bold; padding:7px 16px;"
        )
        apply_shadow_effect(self._btn_cv_preview, radius=6, y_offset=2)
        self._btn_cv_preview.clicked.connect(self._cv_run_preview)

        self._btn_cv_export = QPushButton("⬇  Export Canvas…")
        self._btn_cv_export.setStyleSheet(
            "background:#4CAF50; color:white; font-weight:bold; padding:7px 16px;"
        )
        apply_shadow_effect(self._btn_cv_export, radius=6, y_offset=2)
        self._btn_cv_export.clicked.connect(self._cv_export)

        self._cv_export_format = QComboBox()
        self._cv_export_format.addItem("PNG", "png")
        self._cv_export_format.addItem("JPEG", "jpg")
        self._cv_export_format.addItem("WebP", "webp")

        self._cv_progress = QProgressBar()
        self._cv_progress.setRange(0, 0)
        self._cv_progress.setVisible(False)
        self._cv_progress.setFixedWidth(150)
        self._cv_progress.setTextVisible(False)

        cv_action_bar.addWidget(self._btn_cv_preview)
        cv_action_bar.addWidget(self._btn_cv_export)
        cv_action_bar.addWidget(QLabel("Format:"))
        cv_action_bar.addWidget(self._cv_export_format)
        cv_action_bar.addWidget(self._cv_progress)
        cv_action_bar.addStretch()
        root_layout.addLayout(cv_action_bar)

        return panel

    @Slot(int)
    def _cv_on_preset_changed(self, idx: int):
        _, size = SIZE_PRESETS[idx]
        if size:
            self._cv_width_spin.blockSignals(True)
            self._cv_height_spin.blockSignals(True)
            self._cv_width_spin.setValue(size[0])  # pyrefly: ignore [bad-argument-type]
            self._cv_height_spin.setValue(size[1])  # pyrefly: ignore [bad-argument-type]
            self._cv_width_spin.blockSignals(False)
            self._cv_height_spin.blockSignals(False)

    def _cv_add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Images to Canvas",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        for p in paths:
            if p and p not in self._cv_paths:
                self._cv_paths.append(p)
                self._cv_list.addItem(self._make_cv_item(p))

    def _cv_remove_selected(self):
        for item in reversed(self._cv_list.selectedItems()):
            row = self._cv_list.row(item)
            self._cv_list.takeItem(row)
            if row < len(self._cv_paths):
                self._cv_paths.pop(row)

    def _cv_clear_all(self):
        self._cv_list.clear()
        self._cv_paths.clear()
        self._cv_preview_label.setText("Press 'Preview' to render the canvas.")
        self._cv_status_label.setText("")

    def _cv_sync_paths(self, *_):
        self._cv_paths = [
            self._cv_list.item(r).data(Qt.ItemDataRole.UserRole)
            for r in range(self._cv_list.count())
        ]

    def _cv_pick_bg_color(self):
        initial = QColor(*self._cv_bg_color)
        color = QColorDialog.getColor(
            initial,
            self,
            "Background Colour",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if color.isValid():
            self._cv_bg_color = (color.red(), color.green(), color.blue())
            self._cv_update_bg_button()

    def _cv_update_bg_button(self):
        r, g, b = self._cv_bg_color
        self._cv_bg_btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #555;"
        )
        self._cv_bg_label.setText(f"#{r:02X}{g:02X}{b:02X}")

    def _cv_collect_params(self) -> dict:
        if self._cv_radio_h.isChecked():
            layout = "horizontal"
        elif self._cv_radio_v.isChecked():
            layout = "vertical"
        else:
            layout = "grid"
        return {
            "output_w": self._cv_width_spin.value(),
            "output_h": self._cv_height_spin.value(),
            "layout": layout,
            "grid_cols": self._cv_cols_spin.value(),
            "gap": self._cv_gap_spin.value(),
            "bg_color": self._cv_bg_color,
            "scale_mode": self._cv_scale_combo.currentData(),
        }

    def _cv_run_preview(self):
        if not self._cv_paths:
            QMessageBox.warning(
                self, "No images", "Add at least one image to the canvas."
            )
            return
        if self._cv_thread and self._cv_thread.isRunning():
            return

        self._cv_progress.setVisible(True)
        self._cv_status_label.setText("Rendering preview…")
        self._btn_cv_preview.setEnabled(False)
        self._btn_cv_export.setEnabled(False)

        p_cv = self._cv_collect_params()
        self._cv_worker = CanvasWorker(
            images_params=[(path, {}) for path in self._cv_paths],
            layout_mode=p_cv["layout"],
            canvas_w=p_cv["output_w"],
            canvas_h=p_cv["output_h"],
            bg_color=p_cv["bg_color"],
            scale_mode=p_cv["scale_mode"],
            gap=p_cv["gap"],
            preview=True,
        )
        self._cv_thread = self._cv_worker
        self._cv_worker.sig_finished.connect(self._on_cv_preview_ready)
        self._cv_worker.sig_error.connect(self._on_cv_error)
        self._cv_worker.finished.connect(self._on_cv_thread_done)
        self._cv_worker.finished.connect(self._cv_worker.deleteLater)
        self._cv_worker.start()

    @Slot(object)
    def _on_cv_preview_ready(self, qi: QImage):
        px = QPixmap.fromImage(qi)
        lw = self._cv_preview_label.width()
        lh = self._cv_preview_label.height()
        px = px.scaled(
            lw,
            lh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._cv_preview_label.setPixmap(px)
        p = self._cv_collect_params()
        self._cv_status_label.setText(
            f"Canvas: {p['output_w']} × {p['output_h']} px  |  "
            f"{len(self._cv_paths)} image(s)"
        )

    @Slot(str)
    def _on_cv_error(self, msg: str):
        self._cv_status_label.setText(f"Error: {msg}")

    def _on_cv_thread_done(self):
        self._cv_progress.setVisible(False)
        self._btn_cv_preview.setEnabled(True)
        self._btn_cv_export.setEnabled(True)

    def _cv_export(self):
        if not self._cv_paths:
            QMessageBox.warning(
                self, "No images", "Add at least one image to the canvas."
            )
            return
        fmt = self._cv_export_format.currentData()
        p, _ = QFileDialog.getSaveFileName(
            self,
            "Export Canvas",
            f"canvas.{fmt}",
            f"Image (*.{fmt})",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not p:
            return

        self._cv_progress.setVisible(True)
        self._cv_status_label.setText("Exporting full-resolution canvas…")
        self._btn_cv_preview.setEnabled(False)
        self._btn_cv_export.setEnabled(False)

        p_cv = self._cv_collect_params()
        worker = CanvasWorker(
            images_params=[(path, {}) for path in self._cv_paths],
            layout_mode=p_cv["layout"],
            canvas_w=p_cv["output_w"],
            canvas_h=p_cv["output_h"],
            bg_color=p_cv["bg_color"],
            scale_mode=p_cv["scale_mode"],
            gap=p_cv["gap"],
            preview=False,
        )

        def _on_done(qi: QImage):
            qi.save(p)
            self._cv_status_label.setText(f"Exported: {p}")
            self._cv_progress.setVisible(False)
            self._btn_cv_preview.setEnabled(True)
            self._btn_cv_export.setEnabled(True)
            QMessageBox.information(self, "Export Complete", f"Canvas saved to:\n{p}")

        def _on_err(msg: str):
            self._cv_status_label.setText(f"Export error: {msg}")
            self._cv_progress.setVisible(False)
            self._btn_cv_preview.setEnabled(True)
            self._btn_cv_export.setEnabled(True)

        worker.sig_finished.connect(_on_done)
        worker.sig_error.connect(_on_err)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _make_cv_item(self, path: str, label: str = "") -> QListWidgetItem:
        """Create a QListWidgetItem for the canvas list with async thumbnail."""
        item = QListWidgetItem(label or os.path.basename(path))
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        self._cv_item_map[path] = item
        QThreadPool.globalInstance().start(_ThumbTask(path, 48, 0, self._cv_thumb_hub))
        return item

    @Slot(str, int, object)
    def _on_cv_thumb_loaded(self, path: str, _generation: int, img: QImage):
        item = self._cv_item_map.get(path)
        if item and not img.isNull():
            item.setIcon(QIcon(QPixmap.fromImage(img)))


__all__ = ["_CanvasPanelMixin"]
