"""
AnnotationCanvas — a zoomable image view with rubber-band annotation support.

Users hold the left mouse button and drag to draw a flaw-region rectangle.
Existing annotations are rendered as semi-transparent coloured overlays with
a flaw-type label.

Signals
-------
annotation_added(x, y, w, h)  — emitted with normalized [0,1] region coords
                                  when the user completes a drag.
"""

from __future__ import annotations

from typing import List, Tuple

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QRubberBand, QSizePolicy, QWidget
from gui.src.constants.helpers import _FLAW_COLORS

# Flaw type → (R, G, B, alpha 0-255) overlay colour


class _Annotation:
    """One annotation rectangle with its flaw metadata."""

    __slots__ = ("x", "y", "w", "h", "flaw_type", "severity", "description")

    def __init__(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        flaw_type: str = "seam",
        severity: float = 0.5,
        description: str = "",
    ):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.flaw_type = flaw_type
        self.severity = severity
        self.description = description


class AnnotationCanvas(QWidget):
    """
    Displays a (potentially large) image with zoom/pan and annotation overlays.
    """

    annotation_added = Signal(float, float, float, float)  # x, y, w, h (normalized)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self._pixmap: QPixmap | None = None
        self._zoom: float = 1.0
        self._pan_offset: QPoint = QPoint(0, 0)
        self._annotations: List[_Annotation] = []
        self._annotation_mode: bool = False
        self._active_flaw_type: str = "seam"
        self._active_severity: float = 0.5

        # Rubber-band drag state
        self._drag_origin: QPoint | None = None
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------ public

    def set_image(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._zoom = 1.0
        self._pan_offset = QPoint(0, 0)
        self._fit_to_widget()
        self.update()

    def set_annotation_mode(self, enabled: bool) -> None:
        self._annotation_mode = enabled
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )

    def set_active_flaw_type(self, flaw_type: str) -> None:
        self._active_flaw_type = flaw_type

    def set_active_severity(self, severity: float) -> None:
        self._active_severity = severity

    def clear_annotations(self) -> None:
        self._annotations.clear()
        self.update()

    def remove_annotation(self, idx: int) -> None:
        if 0 <= idx < len(self._annotations):
            self._annotations.pop(idx)
            self.update()

    def add_annotation(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        flaw_type: str = "seam",
        severity: float = 0.5,
        description: str = "",
    ) -> None:
        self._annotations.append(
            _Annotation(x, y, w, h, flaw_type, severity, description)
        )
        self.update()

    @property
    def annotations(self) -> List[_Annotation]:
        return list(self._annotations)

    # --------------------------------------------------------- coordinate helpers

    def _fit_to_widget(self) -> None:
        if self._pixmap is None:
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        if pw == 0 or ph == 0 or ww == 0 or wh == 0:
            return
        self._zoom = min(ww / pw, wh / ph)

    def _image_rect(self) -> QRect:
        """Widget-space bounding rect of the displayed (zoomed) image."""
        if self._pixmap is None:
            return QRect()
        pw = int(self._pixmap.width() * self._zoom)
        ph = int(self._pixmap.height() * self._zoom)
        ox = (self.width() - pw) // 2 + self._pan_offset.x()
        oy = (self.height() - ph) // 2 + self._pan_offset.y()
        return QRect(ox, oy, pw, ph)

    def _widget_to_image(self, pt: QPoint) -> Tuple[float, float]:
        """Convert widget-space point to normalized image coordinates [0,1]."""
        r = self._image_rect()
        if r.width() == 0 or r.height() == 0 or self._pixmap is None:
            return 0.0, 0.0
        nx = (pt.x() - r.left()) / r.width()
        ny = (pt.y() - r.top()) / r.height()
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    def _norm_to_widget(self, x: float, y: float) -> QPoint:
        r = self._image_rect()
        return QPoint(int(r.left() + x * r.width()), int(r.top() + y * r.height()))

    # ------------------------------------------------------------------ events

    def resizeEvent(self, event):
        self._fit_to_widget()
        super().resizeEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if self._pixmap is None:
            return
        delta = event.angleDelta().y()
        factor = 1.12 if delta > 0 else (1.0 / 1.12)
        self._zoom = max(0.05, min(20.0, self._zoom * factor))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._annotation_mode and self._pixmap is not None:
                self._drag_origin = event.pos()
                self._rubber_band.setGeometry(
                    QRect(self._drag_origin, self._drag_origin)
                )
                self._rubber_band.show()
            else:
                # Pan mode: store starting point for drag
                self._drag_origin = event.pos()

    def mouseMoveEvent(self, event):
        if self._drag_origin is None:
            return
        if self._annotation_mode:
            self._rubber_band.setGeometry(
                QRect(self._drag_origin, event.pos()).normalized()
            )
        else:
            delta = event.pos() - self._drag_origin
            self._pan_offset += delta
            self._drag_origin = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self._drag_origin is None:
            return
        if self._annotation_mode:
            self._rubber_band.hide()
            rect = QRect(self._drag_origin, event.pos()).normalized()
            if rect.width() > 5 and rect.height() > 5:
                # Convert rubber-band rect to normalized image coordinates
                x0_n, y0_n = self._widget_to_image(rect.topLeft())
                x1_n, y1_n = self._widget_to_image(rect.bottomRight())
                w_n = max(0.001, x1_n - x0_n)
                h_n = max(0.001, y1_n - y0_n)
                ann = _Annotation(
                    x=x0_n,
                    y=y0_n,
                    w=w_n,
                    h=h_n,
                    flaw_type=self._active_flaw_type,
                    severity=self._active_severity,
                )
                self._annotations.append(ann)
                self.update()
                self.annotation_added.emit(x0_n, y0_n, w_n, h_n)
        self._drag_origin = None

    # ------------------------------------------------------------------ paint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._pixmap is None:
            painter.setPen(QColor(140, 140, 140))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Load a stitched image to begin",
            )
            return

        # Image
        r = self._image_rect()
        painter.drawPixmap(r, self._pixmap, self._pixmap.rect())

        # Annotation overlays
        font = QFont("Monospace", 8)
        painter.setFont(font)
        for ann in self._annotations:
            color_tuple = _FLAW_COLORS.get(ann.flaw_type, _FLAW_COLORS["unknown"])
            fill = QColor(*color_tuple)
            border = QColor(color_tuple[0], color_tuple[1], color_tuple[2], 220)

            tl = self._norm_to_widget(ann.x, ann.y)
            br = self._norm_to_widget(ann.x + ann.w, ann.y + ann.h)
            ann_rect = QRect(tl, br)

            painter.fillRect(ann_rect, fill)
            painter.setPen(QPen(border, 2))
            painter.drawRect(ann_rect)

            # Label
            label = f"{ann.flaw_type}  s={ann.severity:.1f}"
            label_rect = QRect(
                ann_rect.left() + 2, ann_rect.top() + 2, ann_rect.width() - 4, 14
            )
            painter.setPen(QColor(255, 255, 255, 230))
            painter.drawText(
                label_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                label,
            )

        painter.end()


__all__ = ["AnnotationCanvas"]
