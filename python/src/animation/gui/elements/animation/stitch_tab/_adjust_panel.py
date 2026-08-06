"""Adjust sub-tab: per-image tone/color/geometric correction UI and logic.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from PIL import Image
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ....constants import CROP_PRESETS
from ....helpers.animation import AdjustWorker
from ....helpers.animation.adjust_worker import _apply_adjustments
from ....styles import apply_shadow_effect


class _AdjustPanelMixin:
    def _build_adjust_panel(self) -> QWidget:
        from gui.src.tabs.animation.stencil import AdjustPanel

        panel = AdjustPanel(self)
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── LEFT: load bar + preview + action buttons ──────────────────
        left = QWidget()
        left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        load_bar = QHBoxLayout()
        self._adj_path_edit = QLineEdit()
        self._adj_path_edit.setPlaceholderText("No image loaded…")
        self._adj_path_edit.setReadOnly(True)
        btn_adj_open = QPushButton("Open…")
        btn_adj_open.setToolTip("Load an image to adjust.")
        btn_adj_open.clicked.connect(self._adj_load_image)
        apply_shadow_effect(btn_adj_open, radius=4, y_offset=2)
        load_bar.addWidget(self._adj_path_edit)
        load_bar.addWidget(btn_adj_open)
        left_layout.addLayout(load_bar)

        # Preview label
        self._adj_preview = QLabel("No image loaded.")
        self._adj_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._adj_preview.setMinimumSize(300, 240)
        self._adj_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._adj_preview.setStyleSheet(
            "background: #181818; border: 1px solid #3a3a3a; color: #666;"
        )
        left_layout.addWidget(self._adj_preview)

        # Status / dims label
        self._adj_status_label = QLabel("")
        self._adj_status_label.setStyleSheet("color: #777; font-size: 10px;")
        self._adj_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self._adj_status_label)

        # Action row
        act_bar = QHBoxLayout()
        self._btn_adj_save = QPushButton("Save As…")
        self._btn_adj_save.setFixedHeight(36)
        self._btn_adj_save.setToolTip("Save the adjusted image at full resolution.")
        self._btn_adj_save.setStyleSheet(
            "background:#1976D2; color:white; font-weight:bold; padding:6px 14px;"
        )
        self._btn_adj_save.setEnabled(False)
        self._btn_adj_save.clicked.connect(self._adj_save)
        apply_shadow_effect(self._btn_adj_save, radius=6, y_offset=2)

        self._btn_adj_to_stitch = QPushButton("→ Add to Stitch")
        self._btn_adj_to_stitch.setFixedHeight(36)
        self._btn_adj_to_stitch.setToolTip(
            "Apply adjustments and add the result to the Stitch queue."
        )
        self._btn_adj_to_stitch.setStyleSheet(
            "background:#388E3C; color:white; font-weight:bold; padding:6px 14px;"
        )
        self._btn_adj_to_stitch.setEnabled(False)
        self._btn_adj_to_stitch.clicked.connect(self._adj_send_to_stitch)
        apply_shadow_effect(self._btn_adj_to_stitch, radius=6, y_offset=2)

        self._btn_adj_to_canvas = QPushButton("→ Add to Canvas")
        self._btn_adj_to_canvas.setFixedHeight(36)
        self._btn_adj_to_canvas.setToolTip(
            "Apply adjustments and add the result to the Canvas queue."
        )
        self._btn_adj_to_canvas.setStyleSheet(
            "background:#6A1B9A; color:white; font-weight:bold; padding:6px 14px;"
        )
        self._btn_adj_to_canvas.setEnabled(False)
        self._btn_adj_to_canvas.clicked.connect(self._adj_send_to_canvas)
        apply_shadow_effect(self._btn_adj_to_canvas, radius=6, y_offset=2)

        act_bar.addWidget(self._btn_adj_save)
        act_bar.addWidget(self._btn_adj_to_stitch)
        act_bar.addWidget(self._btn_adj_to_canvas)
        left_layout.addLayout(act_bar)

        layout.addWidget(left, stretch=1)

        # ── RIGHT: adjustment controls (scrollable) ────────────────────
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFixedWidth(310)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(6)
        right_scroll.setWidget(right)

        # Geometric group
        geo_group = QGroupBox("Geometric")
        geo_layout = QVBoxLayout(geo_group)
        geo_layout.setSpacing(4)

        rot_bar = QHBoxLayout()
        for label, angle in [("↺ 90°", -90), ("↻ 90°", 90), ("↕ 180°", 180)]:
            btn = QPushButton(label)
            btn.setFixedHeight(36)
            btn.clicked.connect(lambda _, a=angle: self._adj_rotate_by(a))
            apply_shadow_effect(btn, radius=3, y_offset=1)
            rot_bar.addWidget(btn)
        geo_layout.addLayout(rot_bar)

        flip_bar = QHBoxLayout()
        btn_flip_h = QPushButton("⟺ Flip H")
        btn_flip_h.setFixedHeight(36)
        btn_flip_h.setCheckable(True)
        btn_flip_h.setToolTip("Mirror image horizontally.")
        btn_flip_h.clicked.connect(lambda checked: self._adj_set_flip(h=checked))
        apply_shadow_effect(btn_flip_h, radius=3, y_offset=1)
        btn_flip_v = QPushButton("⇕ Flip V")
        btn_flip_v.setFixedHeight(36)
        btn_flip_v.setCheckable(True)
        btn_flip_v.setToolTip("Flip image vertically.")
        btn_flip_v.clicked.connect(lambda checked: self._adj_set_flip(v=checked))
        apply_shadow_effect(btn_flip_v, radius=3, y_offset=1)
        self._btn_flip_h = btn_flip_h
        self._btn_flip_v = btn_flip_v
        flip_bar.addWidget(btn_flip_h)
        flip_bar.addWidget(btn_flip_v)
        geo_layout.addLayout(flip_bar)

        angle_form = QFormLayout()
        angle_form.setSpacing(3)
        self._adj_angle_spin = QDoubleSpinBox()
        self._adj_angle_spin.setRange(-180.0, 180.0)
        self._adj_angle_spin.setValue(0.0)
        self._adj_angle_spin.setSuffix("°")
        self._adj_angle_spin.setSingleStep(0.5)
        self._adj_angle_spin.setDecimals(1)
        self._adj_angle_spin.setToolTip("Fine rotation (positive = clockwise).")
        self._adj_angle_spin.valueChanged.connect(self._adj_schedule_preview)
        angle_form.addRow("Fine rotate:", self._adj_angle_spin)
        geo_layout.addLayout(angle_form)

        right_layout.addWidget(geo_group)

        # Crop group
        crop_group = QGroupBox("Crop to Aspect Ratio")
        crop_form = QFormLayout(crop_group)
        crop_form.setSpacing(3)
        self._adj_crop_combo = QComboBox()
        for label, ratio in CROP_PRESETS:
            self._adj_crop_combo.addItem(label, ratio)
        self._adj_crop_combo.setToolTip(
            "Center-crop to this aspect ratio before other adjustments."
        )
        self._adj_crop_combo.currentIndexChanged.connect(self._adj_schedule_preview)
        crop_form.addRow("Preset:", self._adj_crop_combo)
        right_layout.addWidget(crop_group)

        # White Balance group  (fixes yellow/blue tinting between stitched frames)
        wb_group = QGroupBox("White Balance")
        wb_form = QFormLayout(wb_group)
        wb_form.setSpacing(3)
        self._adj_temperature = self._make_slider(
            -100, 100, 0, "Temp (warm→cool)", wb_form
        )
        self._adj_tint = self._make_slider(-100, 100, 0, "Tint (mag→green)", wb_form)
        btn_auto_wb = QPushButton("Auto WB (Gray World)")
        btn_auto_wb.setFixedHeight(36)
        btn_auto_wb.setToolTip(
            "Apply gray-world white balance: corrects dominant colour casts "
            "(e.g. the yellow tinting that appears when stitching frames with "
            "different colour grading)."
        )
        btn_auto_wb.clicked.connect(self._adj_apply_auto_wb)
        apply_shadow_effect(btn_auto_wb, radius=3, y_offset=1)
        wb_form.addRow(btn_auto_wb)
        right_layout.addWidget(wb_group)

        # Tone group
        tone_group = QGroupBox("Tone")
        tone_form = QFormLayout(tone_group)
        tone_form.setSpacing(3)
        self._adj_brightness = self._make_slider(-100, 100, 0, "Brightness", tone_form)
        self._adj_contrast = self._make_slider(-100, 100, 0, "Contrast", tone_form)
        self._adj_gamma = self._make_slider(10, 500, 100, "Gamma ×100", tone_form)
        self._adj_shadows = self._make_slider(-100, 100, 0, "Shadows", tone_form)
        self._adj_highlights = self._make_slider(-100, 100, 0, "Highlights", tone_form)
        right_layout.addWidget(tone_group)

        # Color group
        color_group = QGroupBox("Color")
        color_form = QFormLayout(color_group)
        color_form.setSpacing(3)
        self._adj_saturation = self._make_slider(-100, 100, 0, "Saturation", color_form)
        self._adj_vibrance = self._make_slider(-100, 100, 0, "Vibrance", color_form)
        self._adj_hue = self._make_slider(-180, 180, 0, "Hue shift", color_form)
        right_layout.addWidget(color_group)

        # Detail group
        detail_group = QGroupBox("Detail")
        detail_form = QFormLayout(detail_group)
        detail_form.setSpacing(3)
        self._adj_sharpen = self._make_slider(0, 100, 0, "Sharpen", detail_form)
        self._adj_blur = self._make_slider(0, 50, 0, "Blur", detail_form)
        right_layout.addWidget(detail_group)

        # Reset
        btn_adj_reset = QPushButton("Reset All")
        btn_adj_reset.setFixedHeight(36)
        btn_adj_reset.setToolTip("Reset all adjustments to defaults.")
        btn_adj_reset.clicked.connect(self._adj_reset)
        apply_shadow_effect(btn_adj_reset, radius=4, y_offset=2)
        right_layout.addWidget(btn_adj_reset)
        right_layout.addStretch()

        layout.addWidget(right_scroll)

        return panel

    def _make_slider(
        self, min_val: int, max_val: int, default: int, label: str, form: QFormLayout
    ) -> QSlider:
        """Create an int QSlider + current-value label and add them as a form row."""
        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(min_val, max_val)
        sl.setValue(default)
        lbl = QLabel(str(default))
        lbl.setFixedWidth(36)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl.setStyleSheet("color: #bbb; font-size: 10px;")
        sl.valueChanged.connect(lambda v, _lbl=lbl: _lbl.setText(str(v)))
        sl.valueChanged.connect(self._adj_schedule_preview)
        hl.addWidget(sl)
        hl.addWidget(lbl)
        form.addRow(label + ":", row)
        return sl

    def _adj_apply_auto_wb(self):
        """Apply gray-world auto white balance as a one-shot preset."""
        if not self._adj_img_path:
            return
        # Set temperature and tint to 0, enable auto_wb via a temp param run
        # then disable auto_wb and bake the correction into temp/tint sliders.
        # Simpler: just run a one-shot AdjustWorker with auto_wb=True and
        # save the result to a temp file, then reload it as the adjusted image.
        try:
            img = Image.open(self._adj_img_path)
            result = _apply_adjustments(img, {"auto_wb": True})
            with tempfile.NamedTemporaryFile(
                suffix=os.path.splitext(self._adj_img_path)[1] or ".png",
                delete=False,
            ) as tmp:
                tmp_name = tmp.name
            result.save(tmp_name)
            # Reset temperature/tint sliders then trigger preview from tmp
            for sl in (self._adj_temperature, self._adj_tint):
                sl.blockSignals(True)
                sl.setValue(0)
                sl.blockSignals(False)
            self._adj_img_path = tmp_name
            self._adj_schedule_preview()
        except Exception as e:
            self._adj_status_label.setText(f"Auto WB failed: {e}")

    def _adj_load_image(self):
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not p:
            return
        self._adj_img_path = p
        self._adj_path_edit.setText(p)
        self._btn_adj_save.setEnabled(True)
        self._btn_adj_to_stitch.setEnabled(True)
        self._btn_adj_to_canvas.setEnabled(True)
        self._adj_run_preview()

    def _adj_schedule_preview(self, *_):
        if self._adj_img_path:
            self._adj_debounce.start()

    def _adj_collect_params(self) -> dict:
        return {
            "crop_ar": self._adj_crop_combo.currentData(),
            "rotate": self._adj_angle_spin.value(),
            "flip_h": self._adj_flip_h,
            "flip_v": self._adj_flip_v,
            "temperature": self._adj_temperature.value(),
            "tint": self._adj_tint.value(),
            "brightness": self._adj_brightness.value(),
            "contrast": self._adj_contrast.value(),
            "gamma": self._adj_gamma.value(),
            "shadows": self._adj_shadows.value(),
            "highlights": self._adj_highlights.value(),
            "saturation": self._adj_saturation.value(),
            "vibrance": self._adj_vibrance.value(),
            "hue": self._adj_hue.value(),
            "sharpen": self._adj_sharpen.value(),
            "blur": self._adj_blur.value(),
        }

    def _adj_run_preview(self):
        if not self._adj_img_path:
            return
        if self._adj_thread and self._adj_thread.isRunning():
            self._adj_debounce.start()
            return

        self._adj_status_label.setText("Rendering preview…")
        self._adj_worker = AdjustWorker(
            self._adj_img_path, self._adj_collect_params(), max_size=900
        )
        self._adj_thread = self._adj_worker
        self._adj_worker.sig_finished.connect(self._on_adj_preview_ready)
        self._adj_worker.sig_error.connect(self._on_adj_error)
        self._adj_worker.finished.connect(self._adj_worker.deleteLater)
        self._adj_worker.start()

    @Slot(object)
    def _on_adj_preview_ready(self, qi: QImage):
        w, h = qi.width(), qi.height()
        px = QPixmap.fromImage(qi)
        # Scale to fit the label while preserving aspect ratio
        lw = self._adj_preview.width()
        lh = self._adj_preview.height()
        px = px.scaled(
            lw,
            lh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._adj_preview.setPixmap(px)
        self._adj_status_label.setText(f"Preview: {w} × {h} px")

    @Slot(str)
    def _on_adj_error(self, msg: str):
        self._adj_status_label.setText(f"Error: {msg}")

    def _adj_rotate_by(self, delta: float):
        current = self._adj_angle_spin.value()
        new_val = (current + delta + 180) % 360 - 180
        self._adj_angle_spin.setValue(new_val)

    def _adj_set_flip(self, h: Optional[bool] = None, v: Optional[bool] = None):
        if h is not None:
            self._adj_flip_h = h
        if v is not None:
            self._adj_flip_v = v
        self._adj_schedule_preview()

    def _adj_reset(self):
        for sl, default in [
            (self._adj_temperature, 0),
            (self._adj_tint, 0),
            (self._adj_brightness, 0),
            (self._adj_contrast, 0),
            (self._adj_gamma, 100),
            (self._adj_shadows, 0),
            (self._adj_highlights, 0),
            (self._adj_saturation, 0),
            (self._adj_vibrance, 0),
            (self._adj_hue, 0),
            (self._adj_sharpen, 0),
            (self._adj_blur, 0),
        ]:
            sl.blockSignals(True)
            sl.setValue(default)
            sl.blockSignals(False)

        self._adj_angle_spin.blockSignals(True)
        self._adj_angle_spin.setValue(0.0)
        self._adj_angle_spin.blockSignals(False)

        self._adj_crop_combo.blockSignals(True)
        self._adj_crop_combo.setCurrentIndex(0)
        self._adj_crop_combo.blockSignals(False)

        self._adj_flip_h = False
        self._adj_flip_v = False
        self._btn_flip_h.setChecked(False)
        self._btn_flip_v.setChecked(False)

        self._adj_schedule_preview(None)

    def _adj_save(self):
        if not self._adj_img_path:
            return
        ext = self._adj_img_path.rsplit(".", 1)[-1].lower()
        p, _ = QFileDialog.getSaveFileName(
            self,
            "Save Adjusted Image",
            f"adjusted.{ext}",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not p:
            return
        self._adj_status_label.setText("Saving full-resolution image…")
        worker = AdjustWorker(
            self._adj_img_path, self._adj_collect_params(), max_size=None
        )

        def _on_done(qi: QImage):
            qi.save(p)
            self._adj_status_label.setText(f"Saved: {p}")

        def _on_err(msg: str):
            self._adj_status_label.setText(f"Save error: {msg}")

        worker.sig_finished.connect(_on_done)
        worker.sig_error.connect(_on_err)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _adj_export_to_temp(self) -> Optional[str]:
        """Save the adjusted result to a temp file and return its path."""
        if not self._adj_img_path:
            return None
        with tempfile.NamedTemporaryFile(suffix=".png", prefix="adj_", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            img = Image.open(self._adj_img_path)
            result = _apply_adjustments(img, self._adj_collect_params())
            result.save(tmp_path)
            return tmp_path
        except Exception as e:
            self._adj_status_label.setText(f"Export error: {e}")
            return None

    def _adj_send_to_stitch(self):
        tmp_path = self._adj_export_to_temp()
        if not tmp_path:
            return
        if tmp_path not in self._frame_paths:
            self._frame_paths.append(tmp_path)
            item = self._make_frame_item(tmp_path)
            item.setText(f"[adj] {os.path.basename(self._adj_img_path)}")  # pyrefly: ignore [no-matching-overload]
            self._frame_list.addItem(item)
            self._refresh_pair_combo()
        self._tab_widget.setCurrentIndex(0)

    def _adj_send_to_canvas(self):
        tmp_path = self._adj_export_to_temp()
        if not tmp_path:
            return
        if tmp_path not in self._cv_paths:
            self._cv_paths.append(tmp_path)
            item = self._make_cv_item(
                tmp_path, f"[adj] {os.path.basename(self._adj_img_path)}"  # pyrefly: ignore [no-matching-overload]
            )
            self._cv_list.addItem(item)
        self._tab_widget.setCurrentIndex(2)


__all__ = ["_AdjustPanelMixin"]
