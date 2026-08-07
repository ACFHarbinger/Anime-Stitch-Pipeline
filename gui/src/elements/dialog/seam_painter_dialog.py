"""HITL Checkpoint 4.5 — Post-composite seam painter.

Shows the finished composite canvas with an interactive paint brush.  The user
paints over seam regions that look wrong; the painted area becomes a hard cost
barrier (cost=1e6) in the DP seam map, forcing the seam to re-route around the
marked zone.  Clicking "Re-Composite" returns the canvas-space paint mask to
the worker, which re-runs Stage 11 with the mask appended to exclusion_masks.

Clicking "Accept Output" accepts the current composite without changes.
Clicking "Cancel" aborts the pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np
from gui.src.constants.elements import (
    _DEFAULT_BRUSH_PX,
    _MAX_PREVIEW_H,
    _MAX_PREVIEW_W,
    _PAINT_COLOR,
)
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import (
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)


def _bgr_to_pixmap(bgr: np.ndarray, max_w: int, max_h: int) -> tuple[QPixmap, float]:
    h, w = bgr.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale < 1.0:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h2, w2 = rgb.shape[:2]
    qi = QImage(rgb.data, w2, h2, 3 * w2, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qi), scale


class _PaintCanvas(QLabel):
    """QLabel that captures mouse paint strokes on a transparent overlay pixmap."""

    def __init__(self, bg_pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self._bg = bg_pixmap
        self._overlay = QPixmap(bg_pixmap.size())
        self._overlay.fill(Qt.GlobalColor.transparent)
        self._brush_px = _DEFAULT_BRUSH_PX
        self._painting = False
        self._last_pt: QPoint | None = None
        self._erasing = False

        self.setFixedSize(bg_pixmap.size())
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._render()

    def set_brush_size(self, px: int):
        self._brush_px = max(2, px)

    def _render(self):
        composite = QPixmap(self._bg.size())
        p = QPainter(composite)
        p.drawPixmap(0, 0, self._bg)
        p.drawPixmap(0, 0, self._overlay)
        p.end()
        self.setPixmap(composite)

    def _paint_at(self, pt: QPoint, erase: bool = False):
        p = QPainter(self._overlay)
        if erase:
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            p.setBrush(Qt.GlobalColor.transparent)
        else:
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            p.setBrush(_PAINT_COLOR)
        p.setPen(Qt.PenStyle.NoPen)
        r = self._brush_px
        p.drawEllipse(pt, r, r)
        if self._last_pt is not None:
            pen = QPen(_PAINT_COLOR if not erase else Qt.GlobalColor.transparent)
            pen.setWidth(r * 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawLine(self._last_pt, pt)
        p.end()
        self._last_pt = pt
        self._render()

    def mousePressEvent(self, event):
        self._painting = True
        self._erasing = event.button() == Qt.MouseButton.RightButton
        self._last_pt = None
        self._paint_at(event.pos(), self._erasing)

    def mouseMoveEvent(self, event):
        if self._painting:
            self._paint_at(event.pos(), self._erasing)

    def mouseReleaseEvent(self, event):
        self._painting = False
        self._last_pt = None

    def clear_paint(self):
        self._overlay.fill(Qt.GlobalColor.transparent)
        self._render()

    def has_paint(self) -> bool:
        img = self._overlay.toImage().convertToFormat(QImage.Format.Format_ARGB32)
        buf = img.bits().tobytes()  # type: ignore[union-attr]  # QImage.bits() is a buffer-like object with .tobytes() at runtime; PySide6 stubs under-type it  # pyrefly: ignore [missing-attribute]
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(img.height(), img.width(), 4)
        return bool(arr[:, :, 3].any())

    def paint_mask_preview(self) -> np.ndarray:
        """Return uint8 (H, W) mask at preview resolution (alpha channel of overlay)."""
        img = self._overlay.toImage().convertToFormat(QImage.Format.Format_ARGB32)
        buf = img.bits().tobytes()  # type: ignore[union-attr]  # QImage.bits() is a buffer-like object with .tobytes() at runtime; PySide6 stubs under-type it  # pyrefly: ignore [missing-attribute]
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(img.height(), img.width(), 4)
        return arr[:, :, 3].copy()


class SeamPainterDialog(QDialog):
    """Interactive post-composite seam painter (HITL checkpoint 4.5)."""

    # exec() return code for re-composite request
    RECOMPOSITE = 2

    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HITL: Seam Painter — Checkpoint 4.5")
        self.resize(580, 820)

        canvas_bgr: np.ndarray = data["canvas_preview"]
        self._canvas_h: int = int(data["canvas_h"])
        self._canvas_w: int = int(data["canvas_w"])
        iteration: int = int(data.get("iteration", 1))

        pixmap, self._scale = _bgr_to_pixmap(canvas_bgr, _MAX_PREVIEW_W, _MAX_PREVIEW_H)

        self._painter_widget = _PaintCanvas(pixmap)

        tip = QLabel(
            "Left-drag: paint seam barrier  |  Right-drag: erase  |  Brush size: slider"
        )
        tip.setStyleSheet("color: #aaa; font-size: 9px;")
        tip.setAlignment(Qt.AlignmentFlag.AlignCenter)

        iter_label = QLabel(f"Iteration {iteration}")
        iter_label.setStyleSheet("color: #ccc; font-size: 10px;")
        iter_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        brush_label = QLabel("Brush:")
        brush_label.setStyleSheet("color: #ccc; font-size: 10px;")
        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setRange(4, 60)
        self._brush_slider.setValue(_DEFAULT_BRUSH_PX)
        self._brush_slider.setMaximumWidth(120)
        self._brush_slider.valueChanged.connect(self._painter_widget.set_brush_size)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._painter_widget.clear_paint)

        ctrl = QHBoxLayout()
        ctrl.addWidget(iter_label)
        ctrl.addStretch()
        ctrl.addWidget(brush_label)
        ctrl.addWidget(self._brush_slider)
        ctrl.addWidget(btn_clear)

        btn_recomp = QPushButton("Re-Composite")
        btn_recomp.setDefault(True)
        btn_recomp.setStyleSheet("background: #8b2222; color: white; font-weight: bold;")
        btn_recomp.clicked.connect(self._on_recomposite)
        btn_accept = QPushButton("Accept Output")
        btn_accept.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_recomp)
        btn_row.addWidget(btn_accept)
        btn_row.addWidget(btn_cancel)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addWidget(tip)
        root.addLayout(ctrl)
        root.addWidget(self._painter_widget, stretch=1)
        root.addLayout(btn_row)

    def _on_recomposite(self):
        self.done(self.RECOMPOSITE)

    def full_resolution_mask(self) -> np.ndarray | None:
        """Return paint mask upscaled to full canvas resolution, or None if empty."""
        preview_mask = self._painter_widget.paint_mask_preview()
        if not preview_mask.any():
            return None
        if self._scale >= 1.0:
            return (preview_mask > 0).astype(np.uint8) * 255
        full = cv2.resize(
            preview_mask,
            (self._canvas_w, self._canvas_h),
            interpolation=cv2.INTER_NEAREST,
        )
        return (full > 0).astype(np.uint8) * 255
