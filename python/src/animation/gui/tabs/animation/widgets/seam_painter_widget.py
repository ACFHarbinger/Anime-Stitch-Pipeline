from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import (
    QPointF,
    Qt,
    QTimer,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ....styles import apply_shadow_effect


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


class _PaintCanvas(QGraphicsView):
    """Paintable canvas showing the overlap blend.  Three brush modes:
    Force-A (red), Force-B (blue), Neutral (eraser).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._overlay: Optional[np.ndarray] = None  # RGBA paint layer, H×W×4
        self._overlay_item: Optional[QGraphicsPixmapItem] = None
        self._brush_mode = "A"  # "A" | "B" | "neutral"
        self._brush_size = 20
        self._painting = False
        self._last_pt: Optional[QPointF] = None
        self._img_w = 1
        self._img_h = 1

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor("#111")))
        self.setMinimumSize(300, 250)

    # ── Public API ───────────────────────────────────────────────────

    def load_blend(self, blend_bgr: np.ndarray):
        """Load the blended composite as background."""
        self._img_h, self._img_w = blend_bgr.shape[:2]
        self._overlay = np.zeros((self._img_h, self._img_w, 4), dtype=np.uint8)
        scene = self._scene
        if self._pix_item:
            scene.removeItem(self._pix_item)
        qi = _bgr_to_qimage(blend_bgr)
        pm = QPixmap.fromImage(qi)
        self._pix_item = QGraphicsPixmapItem(pm)
        scene.addItem(self._pix_item)
        scene.setSceneRect(0, 0, self._img_w, self._img_h)
        # Overlay item
        if self._overlay_item:
            scene.removeItem(self._overlay_item)
        self._overlay_item = QGraphicsPixmapItem()
        self._overlay_item.setZValue(5)
        scene.addItem(self._overlay_item)
        self._update_overlay_pixmap()
        self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        QTimer.singleShot(0, self._fit_image)

    def get_mask_a(self) -> np.ndarray:
        """Returns binary mask where user painted Force-A (red)."""
        if self._overlay is None:
            return np.zeros((self._img_h, self._img_w), dtype=np.uint8)
        return (self._overlay[:, :, 0] > 50).astype(np.uint8) * 255

    def get_mask_b(self) -> np.ndarray:
        if self._overlay is None:
            return np.zeros((self._img_h, self._img_w), dtype=np.uint8)
        return (self._overlay[:, :, 2] > 50).astype(np.uint8) * 255

    def clear_paint(self):
        if self._overlay is not None:
            self._overlay[:] = 0
            self._update_overlay_pixmap()

    def set_brush(self, mode: str, size: int):
        self._brush_mode = mode
        self._brush_size = size

    # ── Paint ────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._overlay is not None:
            self._painting = True
            self._last_pt = self.mapToScene(event.position().toPoint())
            self._paint_at(self._last_pt)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._painting:
            pt = self.mapToScene(event.position().toPoint())
            self._paint_at(pt)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._painting = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def _paint_at(self, scene_pt: QPointF):
        assert self._overlay is not None
        cx = int(scene_pt.x())
        cy = int(scene_pt.y())
        r = self._brush_size
        x0 = max(0, cx - r)
        y0 = max(0, cy - r)
        x1 = min(self._img_w, cx + r)
        y1 = min(self._img_h, cy + r)
        if self._brush_mode == "A":
            self._overlay[y0:y1, x0:x1, 0] = 220  # red
            self._overlay[y0:y1, x0:x1, 2] = 0
            self._overlay[y0:y1, x0:x1, 3] = 120
        elif self._brush_mode == "B":
            self._overlay[y0:y1, x0:x1, 2] = 220  # blue
            self._overlay[y0:y1, x0:x1, 0] = 0
            self._overlay[y0:y1, x0:x1, 3] = 120
        else:
            self._overlay[y0:y1, x0:x1] = 0
        self._update_overlay_pixmap()

    def _update_overlay_pixmap(self):
        if self._overlay is None or self._overlay_item is None:
            return
        rgba = self._overlay
        qi = QImage(
            rgba.data,
            self._img_w,
            self._img_h,
            self._img_w * 4,
            QImage.Format.Format_RGBA8888,
        ).copy()
        self._overlay_item.setPixmap(QPixmap.fromImage(qi))

    def _fit_image(self):
        if self._pix_item is not None:
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, event):
        super().showEvent(event)
        self._fit_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_image()


class SeamPainterWidget(QWidget):
    """
    Shows the blended overlap after alignment.  User paints hard
    constraints; "Compute Seam" finds the min-cost path respecting them.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._warped_a: Optional[np.ndarray] = None
        self._warped_b: Optional[np.ndarray] = None
        self._seam_mask: Optional[np.ndarray] = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Toolbar
        tb = QHBoxLayout()
        tb.addWidget(QLabel("Brush:"))

        self._btn_a = QPushButton("Force A (red)")
        self._btn_a.setCheckable(True)
        self._btn_a.setChecked(True)
        self._btn_a.setStyleSheet("background:#c62828; color:white; font-weight:bold;")
        self._btn_a.clicked.connect(lambda: self._set_brush("A"))
        tb.addWidget(self._btn_a)

        self._btn_b = QPushButton("Force B (blue)")
        self._btn_b.setCheckable(True)
        self._btn_b.setStyleSheet("background:#1565C0; color:white; font-weight:bold;")
        self._btn_b.clicked.connect(lambda: self._set_brush("B"))
        tb.addWidget(self._btn_b)

        self._btn_erase = QPushButton("Erase")
        self._btn_erase.setCheckable(True)
        self._btn_erase.clicked.connect(lambda: self._set_brush("neutral"))
        tb.addWidget(self._btn_erase)

        tb.addWidget(QLabel("Size:"))
        self._size_spin = QSpinBox()
        self._size_spin.setRange(2, 80)
        self._size_spin.setValue(20)
        self._size_spin.valueChanged.connect(self._update_brush)
        tb.addWidget(self._size_spin)

        tb.addStretch()

        btn_blend = QPushButton("Preview Blend")
        btn_blend.clicked.connect(self._preview_blend)
        apply_shadow_effect(btn_blend, radius=4, y_offset=2)
        tb.addWidget(btn_blend)

        btn_seam = QPushButton("🔧 Compute Seam")
        btn_seam.setStyleSheet(
            "background:#1976D2;color:white;font-weight:bold;padding:4px 8px;"
        )
        btn_seam.clicked.connect(self._compute_seam)
        apply_shadow_effect(btn_seam, radius=4, y_offset=2)
        tb.addWidget(btn_seam)

        btn_clear = QPushButton("Clear Paint")
        btn_clear.clicked.connect(self._clear)
        tb.addWidget(btn_clear)

        root.addLayout(tb)

        self._canvas = _PaintCanvas()
        root.addWidget(self._canvas, 1)

        self._status = QLabel(
            "Load two aligned frames and paint hard seam constraints."
        )
        self._status.setStyleSheet("color:#aaa; font-size:10px;")
        root.addWidget(self._status)

    def load_aligned_pair(self, warped_a: np.ndarray, warped_b: np.ndarray):
        self._warped_a = warped_a
        self._warped_b = warped_b
        blend = cv2.addWeighted(warped_a, 0.5, warped_b, 0.5, 0)
        self._canvas.load_blend(blend)
        self._status.setText(
            "Paint Force-A (red) / Force-B (blue) regions, then Compute Seam."
        )

    def get_seam_mask(self) -> Optional[np.ndarray]:
        return self._seam_mask

    def _set_brush(self, mode: str):
        for btn, m in [
            (self._btn_a, "A"),
            (self._btn_b, "B"),
            (self._btn_erase, "neutral"),
        ]:
            btn.setChecked(m == mode)
        self._canvas.set_brush(mode, self._size_spin.value())

    def _update_brush(self):
        for btn, m in [
            (self._btn_a, "A"),
            (self._btn_b, "B"),
            (self._btn_erase, "neutral"),
        ]:
            if btn.isChecked():
                self._canvas.set_brush(m, self._size_spin.value())
                return

    def _clear(self):
        self._canvas.clear_paint()

    def _preview_blend(self):
        if self._warped_a is None or self._warped_b is None:
            return
        blend = cv2.addWeighted(self._warped_a, 0.5, self._warped_b, 0.5, 0)
        self._canvas.load_blend(blend)

    def _compute_seam(self):
        if self._warped_a is None or self._warped_b is None:
            self._status.setText("No warped frames loaded.")
            return

        mask_a = self._canvas.get_mask_a()
        mask_b = self._canvas.get_mask_b()

        diff = (
            cv2.absdiff(self._warped_a, self._warped_b).astype(np.float32).mean(axis=2)
        )
        H, W = diff.shape

        energy = diff.copy()
        energy[mask_a > 0] = 0.0
        energy[mask_b > 0] = 1e9

        horizontal = W > H

        if not horizontal:
            energy = energy.T

        eh, ew = energy.shape
        M = energy.copy()
        for i in range(1, eh):
            for j in range(ew):
                nbrs = [M[i - 1, j]]
                if j > 0:
                    nbrs.append(M[i - 1, j - 1])
                if j < ew - 1:
                    nbrs.append(M[i - 1, j + 1])
                M[i, j] += min(nbrs)

        path = np.zeros(eh, np.int32)
        j = int(np.argmin(M[eh - 1]))
        path[eh - 1] = j
        for i in range(eh - 2, -1, -1):
            nbrs = [j]
            if j > 0:
                nbrs.append(j - 1)
            if j < ew - 1:
                nbrs.append(j + 1)
            j = nbrs[int(np.argmin([M[i, c] for c in nbrs]))]
            path[i] = j

        seam = np.ones((eh, ew), dtype=np.uint8) * 255
        for i in range(eh):
            seam[i, path[i] :] = 0

        if not horizontal:
            seam = seam.T

        self._seam_mask = seam

        result = np.where(seam[:, :, np.newaxis] > 0, self._warped_a, self._warped_b)
        self._canvas.load_blend(result.astype(np.uint8))
        self._status.setText(
            "Seam computed and applied to preview.  "
            "The seam mask will be used during Render."
        )
