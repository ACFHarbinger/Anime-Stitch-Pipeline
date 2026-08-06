from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import (
    Qt,
)
from PySide6.QtGui import (
    QImage,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ....styles import apply_shadow_effect


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


def _apply_color_correction(bgr: np.ndarray, cc: dict) -> np.ndarray:
    """Apply per-frame color correction dict to a BGR uint8 image."""
    if not cc:
        return bgr
    img = bgr.astype(np.float32)
    # Brightness [-100 .. +100]
    img += float(cc.get("brightness", 0))
    # Contrast  [0.1 .. 3.0, 1.0 = no change]
    img = (img - 127.5) * float(cc.get("contrast", 1.0)) + 127.5
    # Saturation [0.0 .. 3.0, 1.0 = no change]
    sat = float(cc.get("saturation", 1.0))
    if sat != 1.0:
        gray = img.mean(axis=2, keepdims=True)
        img = gray + sat * (img - gray)
    # Gamma [0.2 .. 5.0, 1.0 = no change]
    gamma = float(cc.get("gamma", 1.0))
    if gamma != 1.0:
        img = np.sign(img) * (np.abs(img / 255.0) ** (1.0 / gamma)) * 255.0
    # White balance — temperature offset in blue/red channels [-50 .. +50]
    temp = float(cc.get("temperature", 0))
    img[:, :, 0] -= temp  # blue
    img[:, :, 2] += temp  # red
    return np.clip(img, 0, 255).astype(np.uint8)


class RenderPanel(QWidget):
    """Final composite with chosen blending mode."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sequence: List[str] = []
        self._homographies: Dict[Tuple[int, int], np.ndarray] = {}
        self._seam_masks: Dict[Tuple[int, int], np.ndarray] = {}
        self._corrections: Dict[str, dict] = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        form = QFormLayout()
        self._blend_mode = QComboBox()
        self._blend_mode.addItems(
            ["Seam mask", "Feather (50% overlap)", "Laplacian (5-band)"]
        )
        form.addRow("Blend mode:", self._blend_mode)

        self._use_cc = QCheckBox("Apply color corrections")
        self._use_cc.setChecked(True)
        form.addRow("", self._use_cc)

        self._use_seam = QCheckBox("Use painted seam masks")
        self._use_seam.setChecked(True)
        form.addRow("", self._use_seam)

        root.addLayout(form)

        btn_row = QHBoxLayout()
        self._btn_render = QPushButton("⚡ Render Panorama")
        self._btn_render.setStyleSheet(
            "background:#388E3C; color:white; font-weight:bold; padding:6px 16px;"
        )
        self._btn_render.clicked.connect(self._render)
        apply_shadow_effect(self._btn_render, radius=6, y_offset=2)
        btn_row.addWidget(self._btn_render)

        btn_save = QPushButton("Save…")
        btn_save.clicked.connect(self._save)
        apply_shadow_effect(btn_save, radius=4, y_offset=2)
        btn_row.addWidget(btn_save)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.hide()
        root.addWidget(self._progress)

        self._preview_lbl = QLabel()
        self._preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_lbl.setStyleSheet("background:#111;")
        self._preview_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root.addWidget(self._preview_lbl, 1)

        self._status = QLabel("Set sequence and homographies, then Render.")
        self._status.setStyleSheet("color:#aaa; font-size:10px;")
        root.addWidget(self._status)

        self._result_bgr: Optional[np.ndarray] = None

    def set_pipeline(
        self,
        sequence: List[str],
        homographies: Dict[Tuple[int, int], np.ndarray],
        seam_masks: Dict[Tuple[int, int], np.ndarray],
        corrections: Dict[str, dict],
    ):
        self._sequence = sequence
        self._homographies = homographies
        self._seam_masks = seam_masks
        self._corrections = corrections
        self._status.setText(
            f"Sequence: {len(sequence)} frames, "
            f"{len(homographies)} homographies.  Ready to render."
        )

    def _render(self):
        if len(self._sequence) < 2:
            QMessageBox.warning(self, "Render", "Need ≥2 frames in sequence.")
            return
        self._btn_render.setEnabled(False)
        self._progress.show()
        self._progress.setValue(0)

        try:
            result = self._do_render()
            if result is not None:
                self._result_bgr = result
                qi = _bgr_to_qimage(result)
                pm = QPixmap.fromImage(qi)
                self._preview_lbl.setPixmap(
                    pm.scaled(
                        self._preview_lbl.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                self._status.setText(
                    f"Render complete: {result.shape[1]}×{result.shape[0]} px."
                )
        except Exception as e:
            QMessageBox.critical(self, "Render Error", str(e))
        finally:
            self._btn_render.setEnabled(True)
            self._progress.hide()

    def _do_render(self) -> Optional[np.ndarray]:
        seq = self._sequence
        N = len(seq)

        frames = []
        for i, p in enumerate(seq):
            bgr = cv2.imread(p)
            if bgr is None:
                raise RuntimeError(f"Could not read '{p}'.")
            if self._use_cc.isChecked():
                cc = self._corrections.get(p, {})
                if cc:
                    bgr = _apply_color_correction(bgr, cc)
            frames.append(bgr)
            self._progress.setValue(int((i + 1) / N * 30))
            QApplication.processEvents()

        H, W = frames[0].shape[:2]
        affines = [np.eye(2, 3, dtype=np.float32)]
        for i in range(1, N):
            key = (i - 1, i)
            Hpair = self._homographies.get(key)
            if Hpair is None:
                affines.append(affines[-1].copy())
                continue
            tx = float(Hpair[0, 2])
            ty = float(Hpair[1, 2])
            M = np.eye(2, 3, dtype=np.float32)
            M[0, 2] = affines[-1][0, 2] + tx
            M[1, 2] = affines[-1][1, 2] + ty
            affines.append(M)
            self._progress.setValue(30 + int(i / N * 10))
            QApplication.processEvents()

        all_pts = []
        for _i, (frm, M) in enumerate(zip(frames, affines, strict=False)):
            fh, fw = frm.shape[:2]
            corners = np.array([[0, 0], [fw, 0], [fw, fh], [0, fh]], dtype=np.float32)
            warped_c = corners + M[:, 2]
            all_pts.append(warped_c)
        all_pts = np.vstack(all_pts)
        min_xy = all_pts.min(axis=0)
        max_xy = all_pts.max(axis=0)
        T = -min_xy
        cw = int(np.ceil(max_xy[0] - min_xy[0]))
        ch = int(np.ceil(max_xy[1] - min_xy[1]))
        cw, ch = min(cw, 32768), min(ch, 32768)

        canvas = np.zeros((ch, cw, 3), dtype=np.float64)
        weight = np.zeros((ch, cw), dtype=np.float64)

        blend_mode = self._blend_mode.currentText()

        for i, (frm, M) in enumerate(zip(frames, affines, strict=False)):
            Mt = M.copy()
            Mt[0, 2] += T[0]
            Mt[1, 2] += T[1]
            warped = cv2.warpAffine(frm, Mt, (cw, ch), flags=cv2.INTER_LINEAR)
            valid = (warped.max(axis=2) > 0).astype(np.float64)

            if blend_mode == "Seam mask" and i > 0:
                key = (i - 1, i)
                smask = self._seam_masks.get(key)
                if smask is not None:
                    sm_w = cv2.warpAffine(smask, Mt, (cw, ch), flags=cv2.INTER_NEAREST)
                    alpha = (sm_w > 0).astype(np.float64)
                    canvas += warped.astype(np.float64) * alpha[:, :, np.newaxis]
                    weight += alpha
                    self._progress.setValue(40 + int(i / N * 55))
                    QApplication.processEvents()
                    continue

            canvas += warped.astype(np.float64) * valid[:, :, np.newaxis]
            weight += valid
            self._progress.setValue(40 + int(i / N * 55))
            QApplication.processEvents()

        weight = np.maximum(weight, 1.0)
        result = np.clip(canvas / weight[:, :, np.newaxis], 0, 255).astype(np.uint8)

        gray = result.mean(axis=2)
        rows = np.any(gray > 0, axis=1)
        cols = np.any(gray > 0, axis=0)
        if rows.any() and cols.any():
            r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
            c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
            result = result[r0:r1, c0:c1]

        self._progress.setValue(100)
        return result

    def _save(self):
        if self._result_bgr is None:
            QMessageBox.information(self, "Save", "Render first.")
            return
        p, _ = QFileDialog.getSaveFileName(
            self,
            "Save Panorama",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if p:
            cv2.imwrite(p, self._result_bgr)
